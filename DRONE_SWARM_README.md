# Three-Drone Urban Search-and-Rescue Swarm

This project is a discrete three-dimensional multi-agent reinforcement-learning
simulation for post-earthquake urban search and rescue. Three drones use MAPPO
to navigate, resolve incidents, and learn selective communication.

## Mission

The environment is a 17 x 17 disaster area with damaged buildings, rubble,
three victims, three fires, and a 240-tick mission limit. Each drone position
is `(x, y, z)`, so agents can move horizontally, ascend, descend, and interact
with an incident at service altitude.

| Drone | Existing role | Responsibility |
|---|---|---|
| Reconnaissance Drone | Scout | Searches a wider local area for victims, fires, and obstacles. |
| Medical-Support Drone | Ambulance | Reaches victims, descends, and performs simulated first-aid service. |
| Fire-Response Drone | Fireman | Reaches fires, descends, and performs simulated fire-suppression service. |

Buildings block lower flight levels. Drones can route around a building or
ascend above it. Building, boundary, and altitude constraints are enforced as
safety rules; RL learns efficient valid routing, altitude choice, service, and
communication.

This is a discrete 3D environment, not a continuous flight-physics simulator.
Its local grid is a processed-sensor abstraction that could later be produced
by RGB, thermal, LiDAR, or depth perception.

## Selective communication

The project learns selective communication instead of unconditional broadcast:

1. **WHEN** a message is useful.
2. **WHO** should receive it.
3. A learned message payload.

For example, a Scout detection of a victim can be routed to the Medical-Support
Drone, while fire information can be routed to the Fire-Response Drone.
Training uses centralized training with decentralized execution (CTDE), meaning
each drone acts from local observations and received messages at evaluation.

## Requirements

- Windows PowerShell or an equivalent Python environment
- Python 3.11.9
- [uv](https://docs.astral.sh/uv/)
- Git, if cloning the project
- Optional NVIDIA GPU for faster training

CPU training works, but longer MAPPO experiments can take several hours.

## First-time setup

Open PowerShell inside the project folder:

```powershell
git clone <your-repository-url> drone-swarm
cd drone-swarm
uv python install 3.11.9
uv sync --all-extras --all-groups
```

Verify installation and 3D environment rules:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_environment.py tests\test_runtime.py tests\test_drone_3d.py -q
```

## Experiment configurations

| File | Condition | Purpose |
|---|---|---|
| `d1_drone_nocomm.yaml` | D1: No communication | Local-observation baseline. |
| `d2_drone_when.yaml` | D2: WHEN | Learns whether communication is useful. |
| `d3_drone_when_who.yaml` | D3: WHEN + WHO | Learns when to send and the correct specialist recipient. |

All three configurations use the same 3D mission, reward structure, and team.

## Run a smoke-training check

Run this first to confirm the complete 3D pipeline works:

```powershell
.\.venv\Scripts\python.exe -m relay.train `
  --config configs\experiments\d1_drone_nocomm.yaml `
  training.total_timesteps=20000 `
  training.checkpoint_interval=10000 `
  execution.device=cpu `
  execution.num_workers=1 `
  execution.envs_per_worker=1 `
  logging.runs_dir=runs
```

Training creates a directory such as:

```text
runs\YYYYMMDDTHHMMSSZ-d1-drone-nocomm-<hash>\
```

It contains checkpoints, metrics, replays, logs, and the resolved config.

## Train an experiment

After the smoke run succeeds, increase the timestep budget. Example D1 CUDA run:

```powershell
.\.venv\Scripts\python.exe -m relay.train `
  --config configs\experiments\d1_drone_nocomm.yaml `
  training.total_timesteps=500000 `
  training.checkpoint_interval=50000 `
  execution.device=cuda `
  execution.num_workers=4 `
  execution.envs_per_worker=4 `
  logging.runs_dir=runs
```

For CPU-only computers, use `execution.device=cpu` with one worker and one
environment. Change only the config path to train D2 or D3.

## Evaluate a checkpoint

Replace `<run-id>` with the folder printed by the training command:

```powershell
.\.venv\Scripts\python.exe -m relay.evaluate `
  --config configs\experiments\d1_drone_nocomm.yaml `
  --checkpoint "runs\<run-id>\checkpoints\latest.pt" `
  execution.device=cpu `
  evaluation.episodes=100 `
  logging.runs_dir=runs
```

Evaluation reports success rate, mean return, and mean episode length. Always
use the same D1, D2, or D3 config that trained the checkpoint.

## Resume training

```powershell
.\.venv\Scripts\python.exe -m relay.train `
  --config configs\experiments\d1_drone_nocomm.yaml `
  --resume "runs\<run-id>\checkpoints\latest.pt" `
  execution.device=cpu `
  execution.num_workers=1 `
  execution.envs_per_worker=1 `
  logging.runs_dir=runs
```

Use the original config and worker layout. Add a larger
`training.total_timesteps` override if you want to extend training.

## View or export a replay

```powershell
$replay = "runs\<evaluation-run-id>\replays\episode_000000.parquet"
.\.venv\Scripts\python.exe -m relay.replay summary $replay
.\.venv\Scripts\python.exe -m relay.replay verify $replay
.\.venv\Scripts\python.exe -m relay.replay view $replay
.\.venv\Scripts\python.exe -m relay.replay export $replay --output output\drone_mission.gif --fps 8 --stride 2
```

The renderer shows raised damaged buildings, rubble, incidents, drone movement,
flight levels, and specialist service actions.

### Interactive 360-degree viewer

Create a browser-based 3D replay viewer with mouse orbit, zoom, timeline, and
playback controls:

```powershell
.\.venv\Scripts\python.exe -m relay.replay orbit $replay `
  --output output\drone_mission_360.html `
  --stride 2
```

Open the resulting HTML file in a modern browser. Drag to rotate the camera,
use the mouse wheel to zoom, and use the timeline to inspect each replay tick.

## Reproducibility

- Keep the checkpoint, `resolved_config.yaml`, and `evaluation.json` for each result.
- Compare policies with the same deterministic evaluation episode count.
- Report success rate together with mean return and mean episode length.
- Commit source, configs, tests, and documentation. Do not commit `.venv` or `runs`.

## Troubleshooting

- **`No module named relay`**: Run from the project root using `.\.venv\Scripts\python.exe`.
- **CUDA unavailable**: use `execution.device=cpu`.
- **No early progress output**: the trainer first collects a rollout; this is normal.
- **Environment mismatch**: use the exact experiment config that made the checkpoint.

## Scope

The current project focuses on learned selective communication and discrete 3D
navigation. Continuous dynamics, wind, battery models, raw camera perception,
and hardware flight control are future extensions.
