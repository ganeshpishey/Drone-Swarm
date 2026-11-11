#!/usr/bin/env bash
# Isolated PX4 SITL launcher for the active Scout/Ambulance cohort.
# It stages Gazebo resources and parameter stores under a unique /tmp run dir.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
CONFIG="$SCRIPT_DIR/active_uavs.json"
PX4_ROOT="${PX4_AUTOPILOT_DIR:-/tmp/PX4-Autopilot}"
PX4_BUILD="$PX4_ROOT/build/px4_sitl_default"
PX4_BIN="$PX4_BUILD/bin/px4"
PX4_ETC="$PX4_BUILD/etc"
PX4_ROOTFS="$PX4_BUILD/rootfs"
PX4_MODELS="$PX4_ROOT/Tools/simulation/gz/models"
PX4_WORLDS="$PX4_ROOT/Tools/simulation/gz/worlds"
PX4_PLUGINS="$PX4_BUILD/src/modules/simulation/gz_plugins"
PX4_SERVER_CONFIG="$PX4_ROOT/src/modules/simulation/gz_bridge/server.config"
MVSDK_PYTHON="$PROJECT_ROOT/.venv-mavsdk/bin/python"
MVSDK_SERVER="$PROJECT_ROOT/.venv-mavsdk/lib/python3.10/site-packages/mavsdk_grpc/bin/mavsdk_server"
PERCEPTION_PYTHON="$PROJECT_ROOT/.venv-perception/bin/python"
PERCEPTION_MODEL="$PROJECT_ROOT/yolo26n.pt"
WORLD_SOURCE="$(/usr/bin/python3 "$SCRIPT_DIR/gazebo_world_identity.py" select \
  --override "${PX4_GZ_WORLD_FILE:-}" \
  --default "$PROJECT_ROOT/worlds/earthquake_landslide_apron_world.sdf")"
WORLD_SOURCE_SHA256="$(/usr/bin/python3 "$SCRIPT_DIR/gazebo_world_identity.py" sha256 "$WORLD_SOURCE")"
WORLD_NAME="earthquake_landslide_world"
BASE_DIR="${TMPDIR:-/tmp}/drone-swarm-active-uavs-${UID}"
CURRENT_LINK="$BASE_DIR/current"
LOCK_FILE="$BASE_DIR/launch.lock"
MODE="${1:-scout-ambulance}"

usage() {
  cat <<'USAGE'
Usage: launch_active_uavs.sh [telemetry-only|scout-only|ambulance-only|scout-ambulance|integrated-safe|preflight-only|stop]

Starts only Scout, only Ambulance, or both. A Scout-plus-Ambulance run also
starts the Day 10 coordinator, direct localhost UDP agents, Scout camera/YOLO,
and read-only sensor monitors. Ctrl+C or stop sends SIGTERM only to owned
process groups. Per-run logs and staged files are retained under /tmp. The
preflight-only mode performs the same checks for the pair and exits without
starting PX4 or Gazebo. telemetry-only starts Scout PX4 and MAVSDK only;
it does not start perception, survey, coordinator, or Ambulance.
USAGE
}

case "$MODE" in
  telemetry-only|scout-only|ambulance-only|scout-ambulance|integrated-safe|preflight-only|stop) ;;
  -h|--help) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac

mkdir -p "$BASE_DIR"

config_rows() {
  /usr/bin/python3 - "$CONFIG" "$MODE" <<'PY'
import json
import math
import os
import sys
from pathlib import Path

config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
mode = sys.argv[2]
expected = {"scout_0", "ambulance_0"}
if set(config) != expected:
    raise SystemExit(f"active_uavs.json must contain exactly {sorted(expected)}")
selected = {
    "telemetry-only": ["scout_0"],
    "scout-only": ["scout_0"],
    "ambulance-only": ["ambulance_0"],
    "scout-ambulance": ["scout_0", "ambulance_0"],
    "integrated-safe": ["scout_0", "ambulance_0"],
    "preflight-only": ["scout_0", "ambulance_0"],
    "stop": [],
}[mode]
scout_pose_override = os.environ.get("PX4_GZ_SCOUT_SPAWN_POSE", "").strip()
if scout_pose_override:
    if mode != "scout-only" or not os.environ.get("PX4_GZ_SCOUT_MODEL_OVERRIDE", "").strip():
        raise SystemExit("PX4_GZ_SCOUT_SPAWN_POSE is permitted only with scout-only mode and PX4_GZ_SCOUT_MODEL_OVERRIDE.")
    pose_fields = scout_pose_override.split(",")
    if len(pose_fields) != 6:
        raise SystemExit("PX4_GZ_SCOUT_SPAWN_POSE must contain six comma-separated numeric values.")
    try:
        pose_values = [float(value) for value in pose_fields]
    except ValueError:
        raise SystemExit("PX4_GZ_SCOUT_SPAWN_POSE must contain six comma-separated numeric values.")
    if not all(math.isfinite(value) for value in pose_values):
        raise SystemExit("PX4_GZ_SCOUT_SPAWN_POSE values must be finite.")
seen_instances, seen_system_ids, seen_udp, seen_grpc = set(), set(), set(), set()
for name in selected:
    item = config[name]
    instance = int(item["instance"])
    system_id = int(item["system_id"])
    mavsdk_port = int(item["mavsdk_port"])
    grpc_port = int(item["mavsdk_server_port"])
    gcs_port = int(item["px4_gcs_udp_port"])
    if system_id != instance + 1:
        raise SystemExit(f"{name}: system_id must equal instance + 1")
    if instance in seen_instances or system_id in seen_system_ids:
        raise SystemExit(f"duplicate PX4 instance or system ID in active config")
    if mavsdk_port in seen_udp or grpc_port in seen_grpc:
        raise SystemExit(f"duplicate MAVSDK endpoint in active config")
    seen_instances.add(instance)
    seen_system_ids.add(system_id)
    seen_udp.add(mavsdk_port)
    seen_grpc.add(grpc_port)
    pose = scout_pose_override if name == "scout_0" and scout_pose_override else str(item["spawn_pose"])
    fields = pose.split(",")
    if len(fields) != 6:
        raise SystemExit(f"{name}: spawn_pose must contain six comma-separated values")
    print("\t".join((name, str(instance), str(system_id), str(gcs_port),
                     str(mavsdk_port), str(grpc_port), str(item["model_name"]),
                     pose, str(item["lidar_topic"]))))
PY
}

stop_run() {
  local run_dir manifest role instance pid workdir exe pgid cwd
  if [[ ! -L "$CURRENT_LINK" ]]; then
    echo "No active Scout/Ambulance run is recorded at $CURRENT_LINK." >&2
    return 1
  fi
  run_dir="$(readlink -f -- "$CURRENT_LINK")"
  case "$run_dir" in
    "$BASE_DIR"/run-*) ;;
    *) echo "Refusing to stop a run outside $BASE_DIR: $run_dir" >&2; return 1 ;;
  esac
  manifest="$run_dir/processes.tsv"
  if [[ ! -f "$manifest" ]]; then
    echo "Run manifest is missing; refusing to signal processes: $run_dir" >&2
    return 1
  fi
  # Validate every recorded process group before signalling any of them. PX4
  # may exit before Gazebo; in that case validate the surviving group members
  # using the run marker inherited from PX4 instead of trusting a stale PGID.
  while IFS=$'\t' read -r role instance pid workdir; do
    [[ "$pid" =~ ^[0-9]+$ ]] || continue
    if ! kill -0 -- "-$pid" 2>/dev/null; then
      continue
    fi

    if [[ -e "/proc/$pid/exe" ]]; then
      exe="$(readlink -f -- "/proc/$pid/exe" 2>/dev/null || true)"
      pgid="$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ' || true)"
      cwd="$(readlink -f -- "/proc/$pid/cwd" 2>/dev/null || true)"
      if [[ "$exe" != "$(readlink -f -- "$PX4_BIN")" || "$pgid" != "$pid" || "$cwd" != "$(readlink -f -- "$workdir")" ]]; then
        echo "Refusing to signal PID $pid: process identity no longer matches $role instance $instance." >&2
        return 1
      fi
    elif ! /usr/bin/python3 - "$pid" "$run_dir" "$workdir" <<'PY'
import os
import sys
from pathlib import Path

pgid = int(sys.argv[1])
run_dir = Path(sys.argv[2]).resolve()
workdir = Path(sys.argv[3]).resolve()
members = []
for entry in Path("/proc").iterdir():
    if not entry.name.isdigit():
        continue
    try:
        stat = (entry / "stat").read_text()
        fields = stat[stat.rfind(")") + 2 :].split()
        if int(fields[2]) != pgid:
            continue
        cwd = (entry / "cwd").resolve(strict=True)
        argv = (entry / "cmdline").read_bytes().split(b"\0")
        environ = (entry / "environ").read_bytes().split(b"\0")
        executable = (entry / "exe").resolve(strict=True)
    except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
        continue

    marker = b"DRONE_SWARM_ACTIVE_RUN_DIR=" + os.fsencode(run_dir)
    in_workdir = cwd == workdir or workdir in cwd.parents
    is_px4 = executable.name == "px4"
    staged_world = os.fsencode(run_dir / "gz" / "worlds")
    is_gz_sim = any(
        arg.startswith(b"gz sim ") and staged_world in arg for arg in argv if arg
    )
    if marker not in environ or not in_workdir or not (is_px4 or is_gz_sim):
        print(f"unverified process in recorded group: pid={entry.name} exe={executable} cwd={cwd}", file=sys.stderr)
        raise SystemExit(1)
    members.append(entry.name)

if not members:
    raise SystemExit(1)
PY
    then
      echo "Refusing to signal process group $pid: surviving members could not be verified as $role instance $instance." >&2
      return 1
    fi
  done <"$manifest"
  local aux_manifest="$run_dir/auxiliary_processes.tsv"
  if [[ -f "$aux_manifest" ]]; then
    while IFS=$'\t' read -r component pid expected_exe expected_script workdir; do
      [[ "$pid" =~ ^[0-9]+$ ]] || continue
      kill -0 -- "-$pid" 2>/dev/null || continue
      if ! /usr/bin/python3 - "$pid" "$run_dir" "$expected_exe" "$expected_script" "$workdir" <<'PY'
import os
import sys
from pathlib import Path

leader_pid = int(sys.argv[1])
run_dir = Path(sys.argv[2]).resolve()
expected_exe = Path(sys.argv[3]).resolve()
expected_script = os.fsencode(str(Path(sys.argv[4]).resolve()))
workdir = Path(sys.argv[5]).resolve()
try:
    leader = Path(f"/proc/{leader_pid}")
    actual_exe = (leader / "exe").resolve(strict=True)
    pgid = int((leader / "stat").read_text().split(")", 1)[1].split()[2])
    argv = (leader / "cmdline").read_bytes().split(b"\0")
    environ = (leader / "environ").read_bytes().split(b"\0")
except (OSError, ValueError, IndexError):
    raise SystemExit(1)
marker = b"DRONE_SWARM_ACTIVE_RUN_DIR=" + os.fsencode(run_dir)
actual_cwd = (leader / "cwd").resolve(strict=True)
if (actual_exe != expected_exe or pgid != leader_pid or expected_script not in argv
        or marker not in environ or actual_cwd != workdir):
    raise SystemExit(1)
members = 0
for entry in Path("/proc").iterdir():
    if not entry.name.isdigit():
        continue
    try:
        fields = (entry / "stat").read_text().split(")", 1)[1].split()
        if int(fields[2]) != leader_pid:
            continue
        member_env = (entry / "environ").read_bytes().split(b"\0")
        member_cwd = (entry / "cwd").resolve(strict=True)
    except (OSError, ValueError, IndexError):
        continue
    if marker not in member_env or member_cwd != workdir:
        raise SystemExit(1)
    members += 1
if not members:
    raise SystemExit(1)
PY
      then
        echo "Refusing to signal auxiliary process group $pid ($component): identity/run ownership did not verify." >&2
        return 1
      fi
    done <"$aux_manifest"
  fi
  : >"$run_dir/stop.requested"
  if [[ -f "$aux_manifest" ]]; then
    while IFS=$'\t' read -r component pid _expected_exe _expected_script _workdir; do
      [[ "$pid" =~ ^[0-9]+$ ]] || continue
      if kill -0 -- "-$pid" 2>/dev/null; then
        echo "Sending SIGTERM to owned process group $pid ($component)."
        kill -TERM -- "-$pid" 2>/dev/null || true
      fi
    done <"$aux_manifest"
  fi
  while IFS=$'\t' read -r role instance pid workdir; do
    [[ "$pid" =~ ^[0-9]+$ ]] || continue
    kill -0 -- "-$pid" 2>/dev/null || continue
    echo "Sending SIGTERM to owned process group $pid ($role, instance $instance)."
    kill -TERM -- "-$pid" 2>/dev/null || true
  done <"$manifest"

  local attempt remaining=0
  for attempt in $(seq 1 20); do
    remaining=0
    while IFS=$'\t' read -r _role _instance pid _workdir; do
      [[ "$pid" =~ ^[0-9]+$ ]] || continue
      if kill -0 -- "-$pid" 2>/dev/null; then remaining=1; fi
    done <"$manifest"
    if [[ -f "$aux_manifest" ]]; then
      while IFS=$'\t' read -r _component pid _expected_exe _expected_script _workdir; do
        [[ "$pid" =~ ^[0-9]+$ ]] || continue
        if kill -0 -- "-$pid" 2>/dev/null; then remaining=1; fi
      done <"$aux_manifest"
    fi
    [[ "$remaining" == "0" ]] && break
    sleep 1
  done
  if [[ "$remaining" != "0" ]]; then
    echo "Owned process groups remain after SIGTERM; no SIGKILL was sent. Survivors:" >&2
    while IFS=$'\t' read -r _role _instance pid _workdir; do
      [[ "$pid" =~ ^[0-9]+$ ]] || continue
      if kill -0 -- "-$pid" 2>/dev/null; then
        ps -ww -o pid=,ppid=,pgid=,stat=,wchan:24=,args= -g "$pid" >&2
      fi
    done <"$manifest"
    if [[ -f "$aux_manifest" ]]; then
      while IFS=$'\t' read -r component pid _expected_exe _expected_script _workdir; do
        [[ "$pid" =~ ^[0-9]+$ ]] || continue
        if kill -0 -- "-$pid" 2>/dev/null; then
          echo "Surviving auxiliary process group $pid ($component):" >&2
          ps -ww -o pid=,ppid=,pgid=,stat=,wchan:24=,args= -g "$pid" >&2
        fi
      done <"$aux_manifest"
    fi
    echo "Logs and manifest remain at $run_dir." >&2
    return 1
  fi
  rm -f -- "$CURRENT_LINK"
  echo "Scout/Ambulance PX4 and Gazebo processes stopped; logs retained at $run_dir."
}

if [[ "$MODE" == "stop" ]]; then
  stop_run
  exit $?
fi

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "Another active-UAV launcher holds the run lock: $LOCK_FILE" >&2
  exit 1
fi
if [[ -e "$CURRENT_LINK" || -L "$CURRENT_LINK" ]]; then
  echo "An active run is already recorded. Stop it with: $0 stop" >&2
  exit 1
fi

if [[ ! -x "$PX4_BIN" || ! -d "$PX4_ETC" || ! -d "$PX4_ROOTFS" ]]; then
  echo "PX4 SITL binary/ETC/rootfs not found under $PX4_ROOT; nothing launched." >&2
  exit 1
fi
if [[ ! -d "$PX4_MODELS/x500" || ! -d "$PX4_MODELS/x500_base" || ! -d "$PX4_MODELS/mono_cam" ]]; then
  echo "Required stock X500 model resources are missing under $PX4_MODELS." >&2
  exit 1
fi
if [[ ! -d "$PX4_PLUGINS" || ! -f "$PX4_SERVER_CONFIG" ]]; then
  echo "PX4 Gazebo plugin/server configuration is missing; nothing launched." >&2
  exit 1
fi
if [[ ! -f "$WORLD_SOURCE" || ! -d "$PROJECT_ROOT/worlds/earthquake_landslide_assets" ]]; then
  echo "Selected project world or its asset directory is missing; nothing launched." >&2
  exit 1
fi
if ! command -v gz >/dev/null 2>&1 || ! command -v setsid >/dev/null 2>&1; then
  echo "Required executable missing (gz or setsid); nothing launched." >&2
  exit 1
fi

if [[ "$MODE" == "telemetry-only" ]]; then
  if [[ ! -x "$MVSDK_PYTHON" ]] || ! "$MVSDK_PYTHON" -c 'import mavsdk_grpc' >/dev/null 2>&1; then
    echo "Telemetry-only mode requires a working .venv-mavsdk; nothing launched." >&2
    exit 1
  fi
elif [[ "$MODE" == "scout-ambulance" || "$MODE" == "integrated-safe" || "$MODE" == "preflight-only" ]]; then
  if [[ ! -x "$MVSDK_PYTHON" || ! -x "$PERCEPTION_PYTHON" || ! -r "$PERCEPTION_MODEL" ]]; then
    echo "Two-UAV runtime requires .venv-mavsdk, .venv-perception and $PERCEPTION_MODEL; nothing launched." >&2
    exit 1
  fi
  if ! "$MVSDK_PYTHON" -c 'import mavsdk_grpc' >/dev/null 2>&1; then
    echo "MAVSDK-Python runtime is unavailable in $MVSDK_PYTHON; nothing launched." >&2
    exit 1
  fi
  if ! "$PERCEPTION_PYTHON" -c 'import sys; sys.path.insert(0, "/usr/lib/python3/dist-packages"); import gi, numpy, ultralytics; from gz.transport13 import Node; from gz.msgs10.image_pb2 import Image' >/dev/null 2>&1; then
    echo "Scout Gazebo-image/YOLO dependencies are unavailable in $PERCEPTION_PYTHON; nothing launched." >&2
    exit 1
  fi
fi

if pgrep -x px4 >/dev/null 2>&1; then
  echo "A PX4 process already exists. No process was stopped or signaled." >&2
  exit 1
fi
if pgrep -x gz-sim-server >/dev/null 2>&1 || pgrep -x gz-sim >/dev/null 2>&1 || pgrep -x gz >/dev/null 2>&1; then
  echo "A Gazebo process already exists. No process was stopped or signaled." >&2
  exit 1
fi

mapfile -t VEHICLES < <(config_rows)
if ((${#VEHICLES[@]} == 0)); then
  echo "No vehicles selected." >&2
  exit 1
fi

declare -A PORT_TYPES=()
for row in "${VEHICLES[@]}"; do
  IFS=$'\t' read -r role instance _system_id gcs_port mavsdk_port grpc_port _model _pose _topic <<<"$row"
  PORT_TYPES["udp:$gcs_port"]=UDP
  PORT_TYPES["udp:$mavsdk_port"]=UDP
  PORT_TYPES["udp:$((14580 + instance))"]=UDP
  PORT_TYPES["tcp:$grpc_port"]=TCP
done
PORT_TYPES["udp:14550"]=UDP
PORT_TYPES["udp:5600"]=UDP
if [[ "$MODE" == "scout-ambulance" || "$MODE" == "integrated-safe" || "$MODE" == "preflight-only" ]]; then
  PORT_TYPES["udp:8765"]=UDP
  PORT_TYPES["udp:8766"]=UDP
  PORT_TYPES["udp:8767"]=UDP
fi
if ! /usr/bin/python3 - "${!PORT_TYPES[@]}" <<'PY'
import socket
import sys
import errno

errors = []
uncertain = []
for token in sys.argv[1:]:
    kind, port_text = token.split(":", 1)
    family = socket.AF_INET
    sock_type = socket.SOCK_DGRAM if kind == "udp" else socket.SOCK_STREAM
    try:
        sock = socket.socket(family, sock_type)
    except OSError as error:
        uncertain.append(f"{kind.upper()} {port_text}: cannot create verification socket: {error}")
        continue
    try:
        sock.bind(("0.0.0.0", int(port_text)))
    except OSError as error:
        detail = f"{kind.upper()} {port_text}: {error}"
        if error.errno == errno.EADDRINUSE:
            errors.append(detail)
        else:
            uncertain.append(detail)
    finally:
        if sock is not None:
            sock.close()
if errors or uncertain:
    if errors:
        print("Port conflict(s); no simulator was launched:", file=sys.stderr)
        print("\n".join(errors), file=sys.stderr)
    if uncertain:
        print("Could not verify port availability; no simulator was launched:", file=sys.stderr)
        print("\n".join(uncertain), file=sys.stderr)
    raise SystemExit(1)
print("All selected UDP/TCP ports are available.")
PY
then
  exit 1
fi

for row in "${VEHICLES[@]}"; do
  IFS=$'\t' read -r role instance _sid _gcs _mavsdk _grpc model _pose _topic <<<"$row"
  for store in parameters.bson parameters_backup.bson; do
    if [[ ! -r "$PX4_ROOTFS/$instance/$store" ]]; then
      echo "Missing original parameter store $PX4_ROOTFS/$instance/$store; refusing to invent defaults." >&2
      exit 1
    fi
  done
  if [[ ! -f "$PROJECT_ROOT/worlds/disaster_assets/models/$model/model.sdf" ]]; then
    echo "Configured model is missing: $PROJECT_ROOT/worlds/disaster_assets/models/$model/model.sdf" >&2
    exit 1
  fi
done

SELECTED_MODEL_SDFS=()
for row in "${VEHICLES[@]}"; do
  IFS=$'\t' read -r _role _instance _sid _gcs _mavsdk _grpc model _pose _topic <<<"$row"
  SELECTED_MODEL_SDFS+=("$PROJECT_ROOT/worlds/disaster_assets/models/$model/model.sdf")
done
if ! /usr/bin/python3 - "$WORLD_SOURCE" "$PROJECT_ROOT/worlds" "$PROJECT_ROOT/worlds/disaster_assets/models" "$PX4_MODELS" "$PX4_WORLDS" "${SELECTED_MODEL_SDFS[@]}" <<'PY'
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

world = Path(sys.argv[1])
search_roots = [Path(p) for p in sys.argv[2:6]]

def parse(path):
    try:
        return ET.parse(path).getroot()
    except (OSError, ET.ParseError) as exc:
        raise SystemExit(f"Invalid or unreadable SDF {path}: {exc}")

def resolve_uri(uri):
    value = uri.strip()
    relative = value.removeprefix("model://") if value.startswith("model://") else value
    candidates = [root / relative for root in search_roots]
    for path in candidates:
        if path.is_file() or (path.is_dir() and (path / "model.sdf").is_file()):
            return path
    return None

for sdf_path in [world, *(Path(x) for x in sys.argv[6:])]:
    root = parse(sdf_path)
    for uri in root.findall(".//uri"):
        text = uri.text or ""
        if text and not text.startswith(("file://", "https://", "http://")) and resolve_uri(text) is None:
            raise SystemExit(f"SDF URI cannot be resolved from configured asset paths: {sdf_path}: {text}")
print("World and selected model SDF XML plus referenced model/file URIs validated.")
PY
then
  echo "World/model SDF or referenced asset validation failed; nothing launched." >&2
  exit 1
fi

if [[ "$MODE" == "preflight-only" ]]; then
  echo "Selected world source: $WORLD_SOURCE"
  echo "Selected world SHA256: $WORLD_SOURCE_SHA256"
  echo "Preflight passed. No PX4 or Gazebo process was started."
  exit 0
fi

RUN_DIR="$(mktemp -d "$BASE_DIR/run-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")"
STAGE_MODELS="$RUN_DIR/gz/models"
STAGE_WORLDS="$RUN_DIR/gz/worlds"
mkdir -p "$STAGE_MODELS" "$STAGE_WORLDS" "$RUN_DIR/instances" "$RUN_DIR/logs" "$RUN_DIR/runtime"

SCOUT_SDF_CALIBRATION_REPORT="$RUN_DIR/runtime/scout_sdf_calibration_report.json"
if [[ "$MODE" == "scout-only" || "$MODE" == "scout-ambulance" || "$MODE" == "integrated-safe" ]]; then
  echo "Generating run-specific Scout SDF calibration report."
  "$PERCEPTION_PYTHON" "$SCRIPT_DIR/scout_calibration_sync.py" \
    --vehicle-sdf "$PROJECT_ROOT/worlds/disaster_assets/models/x500_mono_cam_down_lidar/model.sdf" \
    --camera-model-sdf "$PX4_MODELS/mono_cam/model.sdf" \
    --base-model-sdf "$PX4_MODELS/x500_base/model.sdf" \
    --calibration-json "$SCRIPT_DIR/scout_camera_calibration.json" \
    --output "$SCOUT_SDF_CALIBRATION_REPORT"
  [[ -s "$SCOUT_SDF_CALIBRATION_REPORT" ]] || {
    echo "Scout SDF calibration report was not generated." >&2
    exit 1
  }
fi

install -m 644 "$WORLD_SOURCE" "$STAGE_WORLDS/$WORLD_NAME.sdf"
/usr/bin/python3 "$SCRIPT_DIR/gazebo_world_identity.py" verify-staged \
  --source "$WORLD_SOURCE" \
  --staged "$STAGE_WORLDS/$WORLD_NAME.sdf" \
  --record "$RUN_DIR/world_identity.json"
ln -s "$PROJECT_ROOT/worlds/earthquake_landslide_assets" "$STAGE_WORLDS/earthquake_landslide_assets"

stage_model() {
  local name="$1" source="$2"
  if [[ ! -f "$source/model.sdf" || -e "$STAGE_MODELS/$name" ]]; then
    echo "Missing or duplicate staged model '$name' ($source)." >&2
    return 1
  fi
  if [[ "$name" == "x500_mono_cam_down_lidar" && -n "${PX4_GZ_SCOUT_MODEL_OVERRIDE:-}" ]]; then
    local override_root override_sdf model_name
    if [[ "$MODE" != "scout-only" ]]; then
      echo "PX4_GZ_SCOUT_MODEL_OVERRIDE is permitted only with scout-only mode." >&2
      return 1
    fi
    override_root="$(realpath -e -- "$PX4_GZ_SCOUT_MODEL_OVERRIDE")"
    override_sdf="$override_root/model.sdf"
    if [[ ! -d "$override_root" || ! -r "$override_sdf" ]]; then
      echo "Scout diagnostic model override must be a readable model directory: $override_root" >&2
      return 1
    fi
    model_name="$(/usr/bin/python3 - "$override_sdf" <<'PY'
import sys
import xml.etree.ElementTree as ET
root = ET.parse(sys.argv[1]).getroot()
model = root.find("model")
print(model.get("name", "") if model is not None else "")
PY
)"
    if [[ "$model_name" != "$name" ]] || ! grep -q 'name="gz::sim::systems::PosePublisher"' "$override_sdf"; then
      echo "Scout diagnostic override must be $name with a Gazebo PosePublisher plugin." >&2
      return 1
    fi
    cp -a -- "$override_root" "$STAGE_MODELS/$name"
    printf 'diagnostic_scout_model_source=%s\ndiagnostic_scout_model_source_sha256=%s\n' \
      "$override_root" "$(sha256sum "$override_sdf" | awk '{print $1}')" \
      >"$RUN_DIR/logs/scout_pose_publisher_model.txt"
    echo "Staged temporary PosePublisher-instrumented Scout model copy: $override_root"
    return 0
  fi
  ln -s "$source" "$STAGE_MODELS/$name"
}

stage_particle_model() {
  local name="$1" source_dir="$2" image_name="$3" source_uri="$4"
  local staged_dir="$STAGE_MODELS/$name"
  local source_image="$source_dir/materials/textures/$image_name"
  local staged_image="$staged_dir/materials/textures/$image_name"

  if [[ ! -f "$source_dir/model.sdf" || ! -r "$source_image" || -e "$staged_dir" || -L "$staged_dir" ]]; then
    echo "Missing particle model/texture or duplicate staged model '$name' ($source_dir)." >&2
    return 1
  fi

  # Keep the project model tree untouched. This copy belongs to this unique run.
  cp -a -- "$source_dir" "$staged_dir"
  if [[ ! -r "$staged_image" ]]; then
    echo "Staged particle texture is missing or unreadable: $staged_image" >&2
    return 1
  fi

  /usr/bin/python3 - "$source_dir" "$staged_dir" "$source_uri" "$image_name" <<'PY' | tee -a "$RUN_DIR/logs/particle-texture-staging.log"
import hashlib
import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

source_dir = Path(sys.argv[1])
staged_dir = Path(sys.argv[2])
source_uri = sys.argv[3]
image_name = sys.argv[4]
source_sdf = source_dir / "model.sdf"
staged_sdf = staged_dir / "model.sdf"
source_image = source_dir / "materials" / "textures" / image_name
staged_image = (staged_dir / "materials" / "textures" / image_name).resolve(strict=True)

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

if not source_image.is_file() or not os.access(source_image, os.R_OK):
    raise SystemExit(f"Source texture is missing or unreadable: {source_image}")

original_text = source_sdf.read_text(encoding="utf-8")
match = re.compile(
    r"(<color_range_image>\s*)" + re.escape(source_uri) +
    r"(\s*</color_range_image>)"
)
updated_text, replacements = match.subn(
    lambda found: found.group(1) + str(staged_image) + found.group(2),
    original_text,
)
if replacements != 1:
    raise SystemExit(
        f"Expected exactly one {source_uri!r} color_range_image in {source_sdf}; "
        f"found {replacements}"
    )
staged_sdf.write_text(updated_text, encoding="utf-8")

source_xml = ET.parse(source_sdf).getroot()
staged_xml = ET.parse(staged_sdf).getroot()
source_nodes = source_xml.findall(".//color_range_image")
staged_nodes = staged_xml.findall(".//color_range_image")
if len(source_nodes) != 1 or (source_nodes[0].text or "").strip() != source_uri:
    raise SystemExit(f"Unexpected source texture reference in {source_sdf}")
if len(staged_nodes) != 1 or (staged_nodes[0].text or "").strip() != str(staged_image):
    raise SystemExit(f"Staged SDF did not retain the intended absolute texture path: {staged_sdf}")

# Confirm the model copy is identical except for its one intended SDF URI.
def entries(root):
    result = {}
    for path in root.rglob("*"):
        rel = path.relative_to(root).as_posix()
        if path.is_symlink():
            result[rel] = ("symlink", os.readlink(path))
        elif path.is_dir():
            result[rel] = ("directory", "")
        elif path.is_file():
            result[rel] = ("file", path)
        else:
            result[rel] = ("other", "")
    return result

source_entries = entries(source_dir)
staged_entries = entries(staged_dir)
if source_entries.keys() != staged_entries.keys():
    raise SystemExit(f"Staged model tree contents differ: {source_dir} vs {staged_dir}")
for rel, (kind, value) in source_entries.items():
    staged_kind, staged_value = staged_entries[rel]
    if kind != staged_kind:
        raise SystemExit(f"Staged model entry type changed: {rel}")
    if rel == "model.sdf":
        if kind != "file":
            raise SystemExit("model.sdf must be a regular file")
        restored = updated_text.replace(str(staged_image), source_uri, 1)
        if restored != original_text:
            raise SystemExit("Staged model.sdf has changes beyond its texture URI")
    elif kind == "file" and sha256(value) != sha256(staged_value):
        raise SystemExit(f"Staged model file differs from source: {rel}")
    elif kind == "symlink" and value != staged_value:
        raise SystemExit(f"Staged model symlink differs from source: {rel}")

source_hash = sha256(source_image)
staged_hash = sha256(staged_image)
if source_hash != staged_hash:
    raise SystemExit(f"Staged texture hash mismatch: {source_image} vs {staged_image}")
print(f"model={source_dir.name}")
print(f"staged_model={staged_dir}")
print(f"texture_uri={staged_image}")
print(f"source_texture_sha256={source_hash}")
print(f"staged_texture_sha256={staged_hash}")
print(f"source_sdf_sha256={sha256(source_sdf)}")
print(f"staged_sdf_sha256={sha256(staged_sdf)}")
print("model_copy_check=PASS (only the requested color_range_image URI differs)")
PY
}

stage_model x500 "$PX4_MODELS/x500"
stage_model x500_base "$PX4_MODELS/x500_base"
stage_model mono_cam "$PX4_MODELS/mono_cam"
for name in standing_person_small rescue_randy_sitting_small survivor_female survivor_male; do
  stage_model "$name" "$PROJECT_ROOT/worlds/disaster_assets/models/$name"
done
stage_particle_model fire_only "$PROJECT_ROOT/worlds/disaster_assets/models/fire_only" \
  flame_life.png materials/textures/flame_life.png
stage_particle_model smoke_only "$PROJECT_ROOT/worlds/disaster_assets/models/smoke_only" \
  smoke_life.png materials/textures/smoke_life.png
for row in "${VEHICLES[@]}"; do
  IFS=$'\t' read -r _role _instance _sid _gcs _mavsdk _grpc model _pose _topic <<<"$row"
  stage_model "$model" "$PROJECT_ROOT/worlds/disaster_assets/models/$model"
done

WORLD_Q="$(printf '%q' "$STAGE_WORLDS")"
MODELS_Q="$(printf '%q' "$STAGE_MODELS")"
PLUGINS_Q="$(printf '%q' "$PX4_PLUGINS")"
SERVER_CONFIG_Q="$(printf '%q' "$PX4_SERVER_CONFIG")"
PX4_MODELS_Q="$(printf '%q' "$PX4_MODELS")"
PX4_WORLDS_Q="$(printf '%q' "$PX4_WORLDS")"
cat >"$RUN_DIR/gz_env.sh" <<ENV
export PX4_GZ_MODELS=$MODELS_Q
export PX4_GZ_WORLDS=$WORLD_Q
export PX4_GZ_PLUGINS=$PLUGINS_Q
export PX4_GZ_SERVER_CONFIG=$SERVER_CONFIG_Q
export GZ_SIM_RESOURCE_PATH="$MODELS_Q/fire_only:$MODELS_Q/smoke_only:$MODELS_Q:$WORLD_Q:$PX4_MODELS_Q:$PX4_WORLDS_Q:\${GZ_SIM_RESOURCE_PATH:-}"
export GZ_SIM_SYSTEM_PLUGIN_PATH="$PLUGINS_Q:\${GZ_SIM_SYSTEM_PLUGIN_PATH:-}"
export GZ_SIM_SERVER_CONFIG_PATH=$SERVER_CONFIG_Q
ENV

: >"$RUN_DIR/processes.tsv"
: >"$RUN_DIR/auxiliary_processes.tsv"
sha256sum "$PX4_ROOTFS/0/parameters.bson" "$PX4_ROOTFS/0/parameters_backup.bson" \
  "$PX4_ROOTFS/1/parameters.bson" "$PX4_ROOTFS/1/parameters_backup.bson" \
  >"$RUN_DIR/original_parameter_hashes.before"
ln -s "$RUN_DIR" "$CURRENT_LINK"
PIDS=()
AUX_PIDS=()
cleanup_owned() {
  local rc=$? pid attempt remaining=0
  trap - EXIT INT TERM ERR
  set +e
  for pid in "${PIDS[@]:-}"; do
    if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 -- "-$pid" 2>/dev/null; then
      kill -TERM -- "-$pid" 2>/dev/null
    fi
  done
  for pid in "${AUX_PIDS[@]:-}"; do
    if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 -- "-$pid" 2>/dev/null; then
      kill -TERM -- "-$pid" 2>/dev/null
    fi
  done
  for attempt in $(seq 1 20); do
    remaining=0
    for pid in "${PIDS[@]:-}"; do
      if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 -- "-$pid" 2>/dev/null; then remaining=1; fi
    done
    for pid in "${AUX_PIDS[@]:-}"; do
      if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 -- "-$pid" 2>/dev/null; then remaining=1; fi
    done
    [[ "$remaining" == "0" ]] && break
    sleep 1
  done
  if [[ "$remaining" == "0" ]]; then
    [[ -L "$CURRENT_LINK" && "$(readlink -f -- "$CURRENT_LINK")" == "$RUN_DIR" ]] && rm -f -- "$CURRENT_LINK"
  else
    echo "Owned process group did not exit after SIGTERM; no SIGKILL will be sent." >&2
    echo "Keeping the launcher lock and run record while waiting for owned processes to exit: $RUN_DIR" >&2
    while [[ "$remaining" != "0" ]]; do
      for pid in "${PIDS[@]:-}"; do
        if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 -- "-$pid" 2>/dev/null; then
          ps -ww -o pid=,ppid=,pgid=,stat=,wchan:24=,args= -g "$pid" >&2
        fi
      done
      while IFS=$'\t' read -r component pid _exe _script _workdir; do
        [[ "$pid" =~ ^[0-9]+$ ]] || continue
        if kill -0 -- "-$pid" 2>/dev/null; then
          echo "Owned auxiliary group $pid ($component) still running:" >&2
          ps -ww -o pid=,ppid=,pgid=,stat=,wchan:24=,args= -g "$pid" >&2
        fi
      done <"$RUN_DIR/auxiliary_processes.tsv"
      sleep 5
      remaining=0
      for pid in "${PIDS[@]:-}"; do
        if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 -- "-$pid" 2>/dev/null; then remaining=1; fi
      done
      for pid in "${AUX_PIDS[@]:-}"; do
        if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 -- "-$pid" 2>/dev/null; then remaining=1; fi
      done
    done
  fi
  if ! sha256sum -c "$RUN_DIR/original_parameter_hashes.before" >"$RUN_DIR/original_parameter_hashes.after" 2>&1; then
    echo "Original PX4 parameter hash changed; evidence retained at $RUN_DIR" >&2
    rc=1
  fi
  echo "Run stopped. Separate logs and isolated parameter copies are retained at $RUN_DIR" >&2
  exit "$rc"
}
trap cleanup_owned EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'cleanup_owned; exit 1' ERR

start_instance() {
  local row="$1" world_creator="$2"
  local role instance sid gcs_port mavsdk_port grpc_port model pose topic workdir pid
  IFS=$'\t' read -r role instance sid gcs_port mavsdk_port grpc_port model pose topic <<<"$row"
  workdir="$RUN_DIR/instances/$instance"
  mkdir -p "$workdir"
  cp -a "$PX4_ROOTFS/$instance/parameters.bson" "$workdir/parameters.bson"
  cp -a "$PX4_ROOTFS/$instance/parameters_backup.bson" "$workdir/parameters_backup.bson"
  cp -a "$RUN_DIR/gz_env.sh" "$workdir/gz_env.sh"
  {
    printf 'role=%s\ninstance=%s\nsystem_id=%s\nmodel=%s\nspawn_pose=%s\n' "$role" "$instance" "$sid" "$model" "$pose"
    printf 'px4_gcs_udp_port=%s\nmavsdk_udp_port=%s\nmavsdk_grpc_port=%s\nlidar_topic=%s\n' "$gcs_port" "$mavsdk_port" "$grpc_port" "$topic"
  } >"$RUN_DIR/logs/$role.config.txt"

  local -a env_args=(
    "PX4_GZ_WORLD=$WORLD_NAME"
    "PX4_SYS_AUTOSTART=4001"
    "PX4_PARAM_SDLOG_PROFILE=129"
    "PX4_PARAM_SDLOG_MODE=0"
    "PX4_GZ_MODEL_POSE=$pose"
    "PX4_SIM_MODEL=gz_$model"
    "PX4_GZ_MODELS=$STAGE_MODELS"
    "PX4_GZ_WORLDS=$STAGE_WORLDS"
    "PX4_GZ_PLUGINS=$PX4_PLUGINS"
    "PX4_GZ_SERVER_CONFIG=$PX4_SERVER_CONFIG"
    "GZ_SIM_RESOURCE_PATH=$STAGE_MODELS/fire_only:$STAGE_MODELS/smoke_only:$STAGE_MODELS:$STAGE_WORLDS:$PX4_MODELS:$PX4_WORLDS:${GZ_SIM_RESOURCE_PATH:-}"
    "GZ_SIM_SYSTEM_PLUGIN_PATH=$PX4_PLUGINS:${GZ_SIM_SYSTEM_PLUGIN_PATH:-}"
    "GZ_SIM_SERVER_CONFIG_PATH=$PX4_SERVER_CONFIG"
    "HEADLESS=1"
    "DRONE_SWARM_ACTIVE_RUN_DIR=$RUN_DIR"
  )
  if [[ "$world_creator" == "0" ]]; then
    env_args+=("PX4_GZ_STANDALONE=1" "PX4_GZ_NO_FOLLOW=1")
  fi
  # Keep the launcher's flock descriptor private so orphaned simulator
  # children cannot retain the run lock after the launcher exits.
  setsid env "${env_args[@]}" "$PX4_BIN" -d "$PX4_ETC" -i "$instance" -w "$workdir" \
    >"$RUN_DIR/logs/$role.px4.log" 2>&1 9>&- &
  pid=$!
  PIDS+=("$pid")
  printf '%s\t%s\t%s\t%s\n' "$role" "$instance" "$pid" "$workdir" >>"$RUN_DIR/processes.tsv"
  echo "Started $role: PX4 instance=$instance system_id=$sid PID=$pid; MAVSDK UDP=$mavsdk_port gRPC=$grpc_port."

  wait_for_instance "$role" "$pid" "$instance" "$model"

  # Start an owned MAVSDK server for this PX4 vehicle.
  local mavsdk_log="$RUN_DIR/logs/$role.mavsdk.log"
  if [[ ! -x "$MVSDK_SERVER" ]]; then
    echo "MAVSDK server executable missing: $MVSDK_SERVER" >&2
    return 1
  fi
  setsid env "DRONE_SWARM_ACTIVE_RUN_DIR=$RUN_DIR" \
    "$MVSDK_SERVER" -p "$grpc_port" --sysid 245 --compid 190 \
    "udpin://0.0.0.0:$mavsdk_port" >"$mavsdk_log" 2>&1 9>&- &
  local mavsdk_pid=$!
  AUX_PIDS+=("$mavsdk_pid")
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "$role-mavsdk" "$mavsdk_pid" "$MVSDK_SERVER" "$MVSDK_SERVER" "$PROJECT_ROOT" \
    >>"$RUN_DIR/auxiliary_processes.tsv"
  echo "Started $role MAVSDK server PID=$mavsdk_pid UDP=$mavsdk_port gRPC=$grpc_port."

  local attempt
  for ((attempt=1; attempt<=15; attempt++)); do
    if grep -q 'Server set to listen on' "$mavsdk_log"; then
      echo "$role MAVSDK gRPC server ready on port $grpc_port."
      break
    fi
    if ! kill -0 "$mavsdk_pid" 2>/dev/null; then
      echo "$role MAVSDK server exited; inspect $mavsdk_log" >&2
      return 1
    fi
    sleep 1
  done
  if ! grep -q 'Server set to listen on' "$mavsdk_log"; then
    echo "Timed out waiting for $role MAVSDK server; inspect $mavsdk_log" >&2
    return 1
  fi

}

start_auxiliary() {
  local component="$1" python="$2" script="$3" log="$4"
  shift 4
  local pid
  setsid env "DRONE_SWARM_ACTIVE_RUN_DIR=$RUN_DIR" PYTHONUNBUFFERED=1 "$python" "$script" "$@" \
    >"$log" 2>&1 9>&- &
  pid=$!
  AUX_PIDS+=("$pid")
  printf '%s\t%s\t%s\t%s\t%s\n' "$component" "$pid" \
    "$(readlink -f -- "$python")" "$(readlink -f -- "$script")" "$(pwd -P)" \
    >>"$RUN_DIR/auxiliary_processes.tsv"
  echo "Started runtime component $component PID=$pid; log=$log"
}

wait_auxiliary_ready() {
  local component="$1" pid="$2" log="$3" pattern="$4" attempt
  for ((attempt=1; attempt<=150; attempt++)); do
    if grep -q -- "$pattern" "$log"; then
      echo "$component reported ready."
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "$component exited during startup; inspect $log" >&2
      return 1
    fi
    sleep 1
  done
  echo "Timed out waiting for $component; inspect $log" >&2
  return 1
}

start_scout_only_survey() {
  [[ "$MODE" == "scout-only" ]] || return 0
  local runtime="$RUN_DIR/runtime"
  local survey_script="$SCRIPT_DIR/scout_victim_survey.py"
  local detector_script="$SCRIPT_DIR/scout_live_person_detection.py"
  local survey_supervisor="$runtime/scout_survey_supervisor.sh"
  local perception_output="$RUN_DIR/perception-waypoint"
  mkdir -p "$runtime" "$perception_output"
  # Use this run's staged Gazebo resources.
  source "$RUN_DIR/gz_env.sh"

  # Start the camera/LiDAR detector before the Scout begins moving.
  start_auxiliary scout-waypoint-detector "$PERCEPTION_PYTHON" "$detector_script" \
    "$RUN_DIR/logs/scout-waypoint-detector.log" \
    --input-source gazebo-image \
    --gazebo-world "$WORLD_NAME" \
    --gazebo-image-topic "/world/$WORLD_NAME/model/x500_mono_cam_down_lidar_0/link/camera_link/sensor/imager/image" \
    --lidar-topic /scout/lidar/points \
    --sdf-calibration-report "$SCOUT_SDF_CALIBRATION_REPORT" \
    --model "$PERCEPTION_MODEL" \
    --output-dir "$perception_output" \
    --interval 1 --max-inferences 0 --imgsz 960 --conf 0.25

  local detector_pid="${AUX_PIDS[-1]}"
  wait_auxiliary_ready scout-waypoint-detector "$detector_pid" \
    "$RUN_DIR/logs/scout-waypoint-detector.log" \
    'Listening for Scout Gazebo Image messages'

  cat >"$survey_supervisor" <<'SURVEY'
#!/usr/bin/env bash
set -Eeuo pipefail
python="$1"
survey="$2"
runtime="$3"
detector_pid="$4"
echo "Starting bounded Scout-only waypoint survey with live camera recording."
if "$python" "$survey" \
  --north 0.7 --east 1.9 --altitude 2.96 --yaw-deg 20 \
  --minimum-altitude 2.5 --max-altitude 4.0 \
  --hold-seconds 35 \
  --skip-frame-capture \
  --pose-log "$runtime/scout_survey_pose.jsonl" \
  --telemetry-log "$runtime/scout_survey_telemetry.jsonl"; then
  survey_status=0
else
  survey_status=$?
fi
echo "Scout survey exited with status $survey_status."
echo "Live detector remains active until the simulation is stopped."
echo "SCOUT_SURVEY_PROCESS_EXITED; inspect survey log and perception output."
while :; do sleep 3600; done
SURVEY
  chmod 700 "$survey_supervisor"
  start_auxiliary scout-survey /bin/bash "$survey_supervisor" \
    "$RUN_DIR/logs/scout-survey.log" \
    "$MVSDK_PYTHON" "$survey_script" "$runtime" "$detector_pid"
  local survey_pid="${AUX_PIDS[-1]}"
  sleep 2
  if ! kill -0 "$survey_pid" 2>/dev/null; then
    echo "Scout-only survey failed during startup; inspect $RUN_DIR/logs/scout-survey.log" >&2
    return 1
  fi
  echo "Scout-only survey supervisor started; bounded flight and RTL are enabled."
}

start_day10_runtime() {
  [[ "$MODE" == "scout-ambulance" || "$MODE" == "integrated-safe" ]] || return 0
  local runtime="$RUN_DIR/runtime"
  local coordinator_script="$SCRIPT_DIR/day10_coordinator_service.py"
  local scout_bridge_script="$SCRIPT_DIR/day10_scout_manet_bridge.py"
  local ambulance_agent_script="$SCRIPT_DIR/day10_vehicle_agent.py"
  local perception_script="$SCRIPT_DIR/scout_live_person_detection.py"
  local coordinator_log="$RUN_DIR/logs/coordinator.log"
  local perception_output="$RUN_DIR/perception"
  local standby_coordinator_args=()
  local standby_agent_args=()
  if [[ "${ENABLE_AMBULANCE_STANDBY_ACCEPTANCE:-0}" == "1" ]]; then
    standby_coordinator_args+=(--allow-ambulance-standby-ack)
    standby_agent_args+=(--accept-ambulance-standby)
    echo "Ambulance standby ACK enabled; flight execution remains disabled."
  fi
  mkdir -p "$runtime" "$perception_output"
  cat >"$runtime/coordinator_peers.json" <<'JSON'
{
  "scout_0": {"host": "127.0.0.1", "port": 8766},
  "ambulance_0": {"host": "127.0.0.1", "port": 8767}
}
JSON
  cat >"$runtime/scout_peer.json" <<'JSON'
{"coordinator": {"host": "127.0.0.1", "port": 8765}}
JSON
  cp -- "$runtime/scout_peer.json" "$runtime/ambulance_peer.json"
  # All service children use this run's staged Gazebo resources and run marker.
  # shellcheck disable=SC1091
  source "$RUN_DIR/gz_env.sh"
  start_auxiliary coordinator "$MVSDK_PYTHON" "$coordinator_script" "$coordinator_log" \
    --drones "$CONFIG" --peers "$runtime/coordinator_peers.json" \
    --bind 127.0.0.1 --port 8765 --state "$runtime/coordinator_state.json" \
    --reports-dir "$PROJECT_ROOT/reports" --control-file "$runtime/control.jsonl" \
    --outbound-minutes 1 --task-minutes 1 --return-minutes 1 \
    --telemetry-stream-timeout-s 15 \
    --scout-lidar-topic /scout/lidar/points --scout-lidar-frame lidar_link \
    --scout-lidar-horizontal-samples 180 --scout-lidar-vertical-samples 16 \
    --camera-topic "/world/$WORLD_NAME/model/x500_mono_cam_down_lidar_0/link/camera_link/sensor/imager/image" \
    --gazebo-world "$WORLD_NAME" --perception-log "$perception_output/detections.jsonl" \
    --perception-model "$PERCEPTION_MODEL" \
    --manet-event-log "$runtime/scout_manet.jsonl" \
    --manet-event-log "$runtime/ambulance_manet.jsonl" \
    --component-log "$RUN_DIR/logs/scout_camera_yolo.log" \
    --component-log "$RUN_DIR/logs/scout_manet_bridge.log" \
    --component-log "$RUN_DIR/logs/ambulance_manet_agent.log" \
    "${standby_coordinator_args[@]}"
  local coordinator_pid="${AUX_PIDS[-1]}"
  wait_auxiliary_ready coordinator "$coordinator_pid" "$coordinator_log" '\[COORDINATOR ONLINE\]' 

  start_auxiliary scout-manet-bridge "$MVSDK_PYTHON" "$scout_bridge_script" \
    "$RUN_DIR/logs/scout_manet_bridge.log" \
    --input "$perception_output/detections.jsonl" --bind 127.0.0.1 --port 8766 \
    --coordinator-host 127.0.0.1 --coordinator-port 8765 \
    --event-log "$runtime/scout_manet.jsonl"
  local scout_bridge_pid="${AUX_PIDS[-1]}"
  wait_auxiliary_ready scout-manet-bridge "$scout_bridge_pid" \
    "$RUN_DIR/logs/scout_manet_bridge.log" 'Scout report bridge listening'

  start_auxiliary ambulance-manet-agent "$MVSDK_PYTHON" "$ambulance_agent_script" \
    "$RUN_DIR/logs/ambulance_manet_agent.log" \
    --vehicle ambulance_0 --bind 127.0.0.1 --port 8767 \
    --coordinator-host 127.0.0.1 --coordinator-port 8765 \
    --mission-file "$runtime/ambulance_proposal.json" \
    --state-file "$runtime/ambulance_agent_state.json" \
    --heartbeat-file "$runtime/ambulance_heartbeat.json" \
    --event-log "$runtime/ambulance_manet.jsonl" \
    "${standby_agent_args[@]}"
  local ambulance_agent_pid="${AUX_PIDS[-1]}"
  wait_auxiliary_ready ambulance-manet-agent "$ambulance_agent_pid" \
    "$RUN_DIR/logs/ambulance_manet_agent.log" 'MANET agent listening'

  start_auxiliary scout-camera-yolo "$PERCEPTION_PYTHON" "$perception_script" \
    "$RUN_DIR/logs/scout_camera_yolo.log" \
    --input-source gazebo-image --gazebo-world "$WORLD_NAME" \
    --gazebo-image-topic "/world/$WORLD_NAME/model/x500_mono_cam_down_lidar_0/link/camera_link/sensor/imager/image" \
    --lidar-topic /scout/lidar/points --model "$PERCEPTION_MODEL" \
    --output-dir "$perception_output" --interval 5 --imgsz 960 --conf 0.25 \
    --pose-log "$runtime/coordinator_navigation_history.json" \
    --sdf-calibration-report "$SCOUT_SDF_CALIBRATION_REPORT" \
    ${SCOUT_VERIFIED_CALIBRATION:+--calibration "$SCOUT_VERIFIED_CALIBRATION"}
  local perception_pid="${AUX_PIDS[-1]}"
  wait_auxiliary_ready scout-camera-yolo "$perception_pid" \
    "$RUN_DIR/logs/scout_camera_yolo.log" 'Listening for Scout Gazebo Image messages'

  if [[ "$MODE" == "integrated-safe" ]]; then
    echo "Integrated-safe mode: Scout survey skipped; no flight commands will be issued."
    return 0
  fi

  # Run the bounded Scout survey alongside live camera/YOLO.
  # The supervisor remains alive after a successful survey so the launcher's
  # auxiliary-process monitor does not mistake normal completion for a crash.
  local survey_script="$SCRIPT_DIR/scout_victim_survey.py"
  local survey_supervisor="$runtime/scout_survey_supervisor.sh"
  cat >"$survey_supervisor" <<'SURVEY'
#!/usr/bin/env bash
set -Eeuo pipefail
python="$1"
survey="$2"
runtime="$3"
echo "Starting bounded Scout survey; live camera receiver remains with YOLO."
"$python" "$survey" \
  --north 0.7 --east 1.9 --altitude 2.96 --yaw-deg 20 \
  --minimum-altitude 2.5 --max-altitude 4.0 \
  --hold-seconds 35 \
  --skip-frame-capture \
  --pose-log "$runtime/scout_survey_pose.jsonl" \
  --telemetry-log "$runtime/scout_survey_telemetry.jsonl"
echo "SCOUT_SURVEY_PROCESS_EXITED; check survey log for landing confirmation."
while :; do sleep 3600; done
SURVEY
  chmod 700 "$survey_supervisor"
  start_auxiliary scout-survey /bin/bash "$survey_supervisor" \
    "$RUN_DIR/logs/scout-survey.log" \
    "$MVSDK_PYTHON" "$survey_script" "$runtime"
  local survey_pid="${AUX_PIDS[-1]}"
  sleep 2
  if ! kill -0 "$survey_pid" 2>/dev/null; then
    echo "Scout survey supervisor failed during startup; inspect $RUN_DIR/logs/scout-survey.log" >&2
    return 1
  fi
  echo "Scout survey supervisor started; bounded flight and RTL are enabled."
}

wait_for_instance() {
  local role="$1" pid="$2" instance="$3" model="$4" attempt
  for ((attempt=1; attempt<=120; attempt++)); do
    if grep -q "\\[gz_bridge\\] world: $WORLD_NAME, model: ${model}_${instance}" "$RUN_DIR/logs/$role.px4.log"; then
      echo "$role connected to Gazebo (instance=$instance, system_id=$((instance + 1)))."
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "$role PX4 exited; inspect $RUN_DIR/logs/$role.px4.log" >&2
      return 1
    fi
    sleep 1
  done
  echo "Timed out waiting for $role; inspect $RUN_DIR/logs/$role.px4.log" >&2
  return 1
}

echo "Active UAV launch mode: $MODE"
echo "World: $WORLD_SOURCE (staged privately as $WORLD_NAME)"
echo "Temporary parameter copies and per-vehicle logs: $RUN_DIR"
if [[ "$MODE" == "telemetry-only" || "$MODE" == "scout-only" ]]; then
  start_instance "${VEHICLES[0]}" 1
elif [[ "$MODE" == "ambulance-only" ]]; then
  start_instance "${VEHICLES[0]}" 1
else
  start_instance "${VEHICLES[0]}" 1
  start_instance "${VEHICLES[1]}" 0
fi

# Apply and verify the battery model on every Scout + Ambulance launch.
# The launcher uses isolated PX4 parameter copies, so reapply settings here.
if [[ "$MODE" == "scout-ambulance" ]]; then
  BATTERY_PYTHON="$PROJECT_ROOT/.venv-px4/bin/python"
  if [[ ! -x "$BATTERY_PYTHON" ]]; then
    echo "Battery configuration requires .venv-px4/bin/python." >&2
    exit 1
  fi

  echo "Applying battery model: 2400-second drain interval, 50% reserve."
  env -u PYTHONPATH -u PYTHONHOME \
    "$BATTERY_PYTHON" "$SCRIPT_DIR/set_swarm_battery_drain.py" \
    --config "$CONFIG" --drain-seconds 2400 --reserve-percent 50 \
    2>&1 | tee "$RUN_DIR/logs/battery_config.log"
fi

start_day10_runtime
if [[ "$MODE" == "scout-only" ]]; then
  start_scout_only_survey
fi

for row in "${VEHICLES[@]}"; do
  IFS=$'\t' read -r role instance sid gcs_port mavsdk_port grpc_port model pose topic <<<"$row"
  echo "ACTIVE role=$role instance=$instance system_id=$sid model=$model pose=$pose mavsdk_udp=$mavsdk_port mavsdk_grpc=$grpc_port lidar=$topic"
done
echo "Press Ctrl+C here, or run '$0 stop' in another terminal to stop only this run."

while :; do
  if [[ -f "$RUN_DIR/stop.requested" ]]; then
    echo "Recorded stop request received; launcher is exiting normally."
    exit 0
  fi
  for pid in "${PIDS[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "A PX4 process exited unexpectedly; stopping this run." >&2
      exit 1
    fi
  done
  for pid in "${AUX_PIDS[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "An owned Day 10 runtime component exited unexpectedly; stopping this run." >&2
      exit 1
    fi
  done
  sleep 1
done
