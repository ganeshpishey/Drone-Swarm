"""Export a self-contained interactive orbital 3D replay viewer."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from relay.runtime.replay_visuals import ReplayData


def _compact_replay(data: ReplayData, stride: int) -> dict[str, Any]:
    if stride < 1:
        raise ValueError("stride must be at least 1")
    first = data.snapshots[0]
    config = data.environment_config
    rows = list(range(0, len(data), stride))
    if rows[-1] != len(data) - 1:
        rows.append(len(data) - 1)
    frames: list[dict[str, Any]] = []
    for index in rows:
        snapshot = data.snapshots[index]
        frames.append(
            {
                "tick": int(data.rows[index]["tick"]),
                "agents": {
                    agent_id: {
                        "x": int(agent["x"]),
                        "y": int(agent["y"]),
                        "z": int(agent.get("z", 0)),
                        "role": str(agent["role"]),
                        "service": str(agent.get("service_phase", "cruise")),
                    }
                    for agent_id, agent in snapshot["agents"].items()
                },
                "incidents": {
                    incident_id: {
                        "x": int(incident["x"]),
                        "y": int(incident["y"]),
                        "kind": str(incident["kind"]),
                        "status": str(incident["status"]),
                    }
                    for incident_id, incident in snapshot["incidents"].items()
                },
            }
        )
    return {
        "title": "Drone Swarm 3D Replay",
        "width": int(config["width"]),
        "height": int(config["height"]),
        "flightLevels": int(config.get("flight_levels", 4)),
        "buildingHeights": first.get("building_heights", first["walls"]),
        "walls": first["walls"],
        "frames": frames,
    }


def export_orbit_viewer(
    data: ReplayData, output: str | Path, *, stride: int = 1
) -> Path:
    """Write an offline HTML viewer with orbit, zoom, and replay controls."""
    output_path = Path(output).resolve()
    if output_path.suffix.lower() != ".html":
        raise ValueError("orbit viewer output must use a .html extension")
    payload = json.dumps(_compact_replay(data, stride), separators=(",", ":"))
    escaped_payload = html.escape(payload).replace("</script", "<\\/script")
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Drone Swarm 3D Replay</title>
<style>
:root {{ color-scheme: dark; font-family: system-ui, sans-serif; background:#10151d; color:#ecf2f8; }}
body {{ margin:0; min-height:100vh; display:grid; grid-template-rows:auto 1fr auto; }}
header {{ padding:12px 18px; background:#182431; border-bottom:1px solid #33485c; }}
h1 {{ margin:0; font-size:20px; }} p {{ margin:4px 0 0; color:#afc0d0; font-size:13px; }}
#scene {{ width:100%; height:calc(100vh - 142px); display:block; background:linear-gradient(#15283b,#0c1118); touch-action:none; }}
#controls {{ padding:10px 18px; display:flex; align-items:center; gap:12px; flex-wrap:wrap; background:#182431; border-top:1px solid #33485c; }}
button {{ background:#2a78b8; color:white; border:0; border-radius:5px; padding:7px 11px; font-weight:600; cursor:pointer; }}
input[type=range] {{ min-width:260px; flex:1; }}
#status {{ color:#c7d7e6; font-variant-numeric:tabular-nums; min-width:180px; }}
.hint {{ color:#afc0d0; font-size:12px; }}
</style></head><body>
<header><h1>Drone Swarm — Interactive 3D Mission Replay</h1><p>Drag to orbit 360°, scroll to zoom, and use the timeline to inspect each mission tick.</p></header>
<canvas id="scene" aria-label="Interactive 3D drone mission replay"></canvas>
<div id="controls"><button id="play" type="button">Play</button><label for="tick">Mission tick</label><input id="tick" type="range" min="0" value="0"><span id="status"></span><span class="hint">Blue: Scout · Green: Medical · Orange: Fire response</span></div>
<script id="replay-data" type="application/json">{escaped_payload}</script>
<script>
const data=JSON.parse(document.getElementById('replay-data').textContent);
const canvas=document.getElementById('scene'),ctx=canvas.getContext('2d'),slider=document.getElementById('tick'),play=document.getElementById('play'),status=document.getElementById('status');
slider.max=data.frames.length-1; let frame=0,playing=false,yaw=-0.7,pitch=0.62,zoom=36,last=null;
const colors={{scout:'#42a5f5',ambulance:'#43c98a',fireman:'#ff9d42'}};
function resize(){{const r=canvas.getBoundingClientRect();canvas.width=Math.max(320,Math.floor(r.width*devicePixelRatio));canvas.height=Math.max(240,Math.floor(r.height*devicePixelRatio));ctx.setTransform(devicePixelRatio,0,0,devicePixelRatio,0,0);draw();}}
function project(x,y,z){{const cx=(data.width-1)/2,cy=(data.height-1)/2;let X=x-cx,Y=y-cy,Z=z;const a=Math.cos(yaw),b=Math.sin(yaw);const rx=X*a-Y*b, ry=X*b+Y*a;const cp=Math.cos(pitch),sp=Math.sin(pitch);const sy=ry*sp+Z*cp, depth=ry*cp-Z*sp;const r=canvas.getBoundingClientRect();return {{x:r.width/2+rx*zoom,y:r.height*.62-sy*zoom,depth}};}}
function line(a,b,color,w=1){{ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.strokeStyle=color;ctx.lineWidth=w;ctx.stroke();}}
function cube(x,y,z,h,color){{const p=[project(x,y,z),project(x+1,y,z),project(x+1,y+1,z),project(x,y+1,z),project(x,y,z+h),project(x+1,y,z+h),project(x+1,y+1,z+h),project(x,y+1,z+h)];ctx.fillStyle=color;ctx.beginPath();[4,5,6,7].forEach((n,i)=>i?ctx.lineTo(p[n].x,p[n].y):ctx.moveTo(p[n].x,p[n].y));ctx.closePath();ctx.fill();ctx.fillStyle='rgba(8,16,24,.5)';for(const f of [[0,1,5,4],[1,2,6,5]]){{ctx.beginPath();f.forEach((n,i)=>i?ctx.lineTo(p[n].x,p[n].y):ctx.moveTo(p[n].x,p[n].y));ctx.closePath();ctx.fill();}}ctx.strokeStyle='rgba(205,225,240,.22)';ctx.lineWidth=1;for(const [a,b] of [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]])line(p[a],p[b],ctx.strokeStyle);}}
function drawDrone(agent,id){{const p=project(agent.x+.5,agent.y+.5,agent.z+.38);const ground=project(agent.x+.5,agent.y+.5,0);line(ground,p,'rgba(210,225,240,.35)',1);ctx.beginPath();ctx.arc(p.x,p.y,7,0,Math.PI*2);ctx.fillStyle=colors[agent.role]||'#ddd';ctx.fill();ctx.strokeStyle='#ffffff';ctx.lineWidth=1.4;ctx.stroke();ctx.fillStyle='#fff';ctx.font='11px system-ui';ctx.textAlign='center';ctx.fillText(id.replace('_0',''),p.x,p.y-11);}}
function draw(){{const r=canvas.getBoundingClientRect();ctx.clearRect(0,0,r.width,r.height);const horizon=project(0,0,0);const end=project(data.width,data.height,0);ctx.fillStyle='#0f1c22';ctx.fillRect(0,0,r.width,r.height);const f=data.frames[frame];const tiles=[];for(let y=0;y<data.height;y++)for(let x=0;x<data.width;x++)tiles.push({{x,y,d:project(x+.5,y+.5,0).depth}});tiles.sort((a,b)=>a.d-b.d);for(const t of tiles){{const h=data.buildingHeights[t.y][t.x];if(h>0)cube(t.x,t.y,0,h,'#5c6772');else{{const p=[project(t.x,t.y,0),project(t.x+1,t.y,0),project(t.x+1,t.y+1,0),project(t.x,t.y+1,0)];ctx.fillStyle=(t.x+t.y)%2?'#263d43':'#2b464a';ctx.beginPath();p.forEach((q,i)=>i?ctx.lineTo(q.x,q.y):ctx.moveTo(q.x,q.y));ctx.closePath();ctx.fill();ctx.strokeStyle='rgba(186,210,221,.16)';ctx.stroke();}}}}
for(const [id,inc] of Object.entries(f.incidents)){{if(inc.status==='resolved')continue;const p=project(inc.x+.5,inc.y+.5,.15);ctx.fillStyle=inc.kind==='fire'?'#ff7043':'#ffd166';ctx.beginPath();ctx.arc(p.x,p.y,6,0,Math.PI*2);ctx.fill();ctx.fillStyle='#131313';ctx.font='bold 10px system-ui';ctx.textAlign='center';ctx.fillText(inc.kind==='fire'?'F':'V',p.x,p.y+3);}}
Object.entries(f.agents).sort((a,b)=>project(a[1].x,a[1].y,a[1].z).depth-project(b[1].x,b[1].y,b[1].z).depth).forEach(([id,a])=>drawDrone(a,id));status.textContent=`Frame ${{frame+1}}/${{data.frames.length}} · Tick ${{f.tick}}`;}}
canvas.addEventListener('pointerdown',e=>{{last={{x:e.clientX,y:e.clientY}};canvas.setPointerCapture(e.pointerId);}});canvas.addEventListener('pointermove',e=>{{if(!last)return;yaw+=(e.clientX-last.x)*.01;pitch=Math.max(.12,Math.min(1.35,pitch+(e.clientY-last.y)*.01));last={{x:e.clientX,y:e.clientY}};draw();}});canvas.addEventListener('pointerup',()=>last=null);canvas.addEventListener('wheel',e=>{{e.preventDefault();zoom=Math.max(15,Math.min(90,zoom-e.deltaY*.025));draw();}},{{passive:false}});slider.addEventListener('input',()=>{{frame=Number(slider.value);draw();}});play.addEventListener('click',()=>{{playing=!playing;play.textContent=playing?'Pause':'Play';}});function animate(){{if(playing){{frame=(frame+1)%data.frames.length;slider.value=frame;draw();}}requestAnimationFrame(animate);}}window.addEventListener('resize',resize);resize();animate();
</script></body></html>"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")
    return output_path
