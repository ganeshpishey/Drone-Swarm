#!/usr/bin/env bash
set -euo pipefail

PX4_ROOT="${PX4_AUTOPILOT_DIR:-/tmp/PX4-Autopilot}"
PX4_BIN="$PX4_ROOT/build/px4_sitl_default/bin/px4"
RUN_DIR="${TMPDIR:-/tmp}/drone-swarm-px4-multi-${UID}"
GCS_HEARTBEAT_PID_FILE="$RUN_DIR/fleet_mavlink_heartbeat.pid"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
WORLD_NAME="earthquake_landslide_world"
if [[ "${1:-}" == "fireman-only" ]]; then
  DEFAULT_WORLD_FILE="$PROJECT_ROOT/worlds/lidar_obstacle_validation_world.sdf"
else
  DEFAULT_WORLD_FILE="$PROJECT_ROOT/worlds/earthquake_landslide_apron_world.sdf"
fi
WORLD_FILE="${PX4_GZ_WORLD_FILE:-$DEFAULT_WORLD_FILE}"
PX4_WORLDS_DIR="$PX4_ROOT/Tools/simulation/gz/worlds"
PX4_MODELS_DIR="$PX4_ROOT/Tools/simulation/gz/models"
PX4_MODEL_NAME="x500"
PX4_SCOUT_MODEL_NAME="x500_mono_cam_down_lidar"
PX4_AMBULANCE_MODEL_NAME="x500_ambulance_lidar"
# Allow a stock-X500 Fireman-only baseline to isolate PX4/Gazebo sensor health
# from the custom LiDAR model. Normal swarm launches keep the custom model.
PX4_FIREMAN_MODEL_NAME="${PX4_GZ_FIREMAN_MODEL:-x500_fireman_lidar}"
DISASTER_WORLD_MODELS_DIR="${DISASTER_WORLD_MODELS_DIR:-${HOME}/disaster_world/models}"

if [[ "${1:-}" == "stop" ]]; then
  if [[ -f "$GCS_HEARTBEAT_PID_FILE" ]]; then
    heartbeat_pid="$(cat "$GCS_HEARTBEAT_PID_FILE" 2>/dev/null || true)"
    if [[ "$heartbeat_pid" =~ ^[0-9]+$ ]] && kill -0 "$heartbeat_pid" 2>/dev/null; then
      kill -TERM "$heartbeat_pid" 2>/dev/null || true
    fi
    rm -f "$GCS_HEARTBEAT_PID_FILE"
  fi
  pkill -TERM -x px4 2>/dev/null || true
  pkill -x px4-gz_bridge 2>/dev/null || true
  pkill -TERM -x gz-sim-server 2>/dev/null || true
  pkill -TERM -x gz-sim 2>/dev/null || true
  pkill -x gz-gui 2>/dev/null || true
  pkill -TERM -f '[g]z sim' 2>/dev/null || true
  for _ in $(seq 1 5); do
    if ! pgrep -x px4 >/dev/null && ! pgrep -x px4-gz_bridge >/dev/null \
      && ! pgrep -x gz-sim-server >/dev/null && ! pgrep -x gz-sim >/dev/null \
      && ! pgrep -f '[g]z sim' >/dev/null; then
      break
    fi
    sleep 1
  done
  # Force-stop any simulator processes that ignored SIGTERM.
  pkill -KILL -x px4 2>/dev/null || true
  pkill -KILL -x px4-gz_bridge 2>/dev/null || true
  pkill -KILL -x gz-sim-server 2>/dev/null || true
  pkill -KILL -x gz-sim 2>/dev/null || true
  pkill -KILL -f '[g]z sim' 2>/dev/null || true
  if pgrep -a -x px4 || pgrep -a -x px4-gz_bridge || pgrep -a -x gz-sim-server \
    || pgrep -a -x gz-sim || pgrep -af '[g]z sim'; then
    echo "Some PX4/Gazebo processes are still running; see the PIDs above." >&2
    exit 1
  fi
  echo "Stopped PX4 SITL and Gazebo Sim processes."
  exit 0
fi

if [[ -n "${1:-}" && "${1:-}" != "scout-only" && "${1:-}" != "fireman-only" ]]; then
  echo "Usage: $0 [scout-only|fireman-only|stop]" >&2
  exit 2
fi
SCOUT_ONLY=0
if [[ "${1:-}" == "scout-only" ]]; then
  SCOUT_ONLY=1
fi
FIREMAN_ONLY=0
if [[ "${1:-}" == "fireman-only" ]]; then
  FIREMAN_ONLY=1
fi
FIREMAN_INSTANCE="${PX4_GZ_FIREMAN_INSTANCE:-2}"
if [[ ! "$FIREMAN_INSTANCE" =~ ^[0-2]$ ]]; then
  echo "PX4_GZ_FIREMAN_INSTANCE must be 0, 1, or 2 (got '$FIREMAN_INSTANCE')." >&2
  exit 2
fi
if [[ "$FIREMAN_ONLY" != "1" && "$FIREMAN_INSTANCE" != "2" ]]; then
  echo "PX4_GZ_FIREMAN_INSTANCE is only supported in fireman-only mode." >&2
  exit 2
fi
SENSOR_DIAGNOSTIC="${PX4_GZ_SENSOR_DIAGNOSTIC:-0}"
if [[ "$SENSOR_DIAGNOSTIC" != "0" && "$SENSOR_DIAGNOSTIC" != "1" ]]; then
  echo "PX4_GZ_SENSOR_DIAGNOSTIC must be 0 or 1 (got '$SENSOR_DIAGNOSTIC')." >&2
  exit 2
fi
if [[ "$SENSOR_DIAGNOSTIC" == "1" && "$FIREMAN_ONLY" != "1" ]]; then
  echo "PX4_GZ_SENSOR_DIAGNOSTIC=1 requires fireman-only mode." >&2
  exit 2
fi
if [[ "$SENSOR_DIAGNOSTIC" == "1" ]]; then
  declare -A heartbeat_candidates=()
  if [[ -f "$GCS_HEARTBEAT_PID_FILE" ]]; then
    recorded_pid="$(cat "$GCS_HEARTBEAT_PID_FILE" 2>/dev/null || true)"
    if [[ "$recorded_pid" =~ ^[0-9]+$ ]]; then
      heartbeat_candidates["$recorded_pid"]=1
    fi
    rm -f "$GCS_HEARTBEAT_PID_FILE"
  fi
  while read -r heartbeat_pid; do
    [[ "$heartbeat_pid" =~ ^[0-9]+$ ]] && heartbeat_candidates["$heartbeat_pid"]=1
  done < <(pgrep -f -- 'integration/mavsdk_controller/fleet_mavlink_heartbeat.py' 2>/dev/null || true)

  stopped_helper=0
  for heartbeat_pid in "${!heartbeat_candidates[@]}"; do
    [[ -r "/proc/$heartbeat_pid/cmdline" ]] || continue
    heartbeat_cmdline="$(tr '\0' ' ' <"/proc/$heartbeat_pid/cmdline" 2>/dev/null || true)"
    heartbeat_cwd="$(readlink -f "/proc/$heartbeat_pid/cwd" 2>/dev/null || true)"
    [[ "$heartbeat_cwd" == "$PROJECT_ROOT" ]] || continue
    [[ "$heartbeat_cmdline" == *"integration/mavsdk_controller/fleet_mavlink_heartbeat.py"* ]] || continue
    kill -TERM "$heartbeat_pid" 2>/dev/null || true
    for _ in $(seq 1 20); do
      kill -0 "$heartbeat_pid" 2>/dev/null || break
      sleep 0.1
    done
    if kill -0 "$heartbeat_pid" 2>/dev/null; then
      echo "Could not stop the prior Drone-Swarm heartbeat helper (PID $heartbeat_pid)." >&2
      exit 2
    fi
    stopped_helper=1
  done
  if [[ "$stopped_helper" == "1" ]]; then
    echo "Stopped the previous Drone-Swarm GCS heartbeat helper for the sensor diagnostic."
  fi
fi
if [[ "$SENSOR_DIAGNOSTIC" == "1" ]] \
   && command -v ss >/dev/null 2>&1 \
   && ss -H -lun 'sport = :14550' 2>/dev/null | grep -q .; then
  echo "UDP 14550 is already in use; stop its GCS listener before sensor-diagnostic mode." >&2
  exit 2
fi
HEADLESS="${PX4_GZ_HEADLESS:-0}"
if [[ "$FIREMAN_ONLY" == "1" && -z "${PX4_GZ_HEADLESS+x}" ]]; then
  HEADLESS=1
fi

if [[ ! -x "$PX4_BIN" ]]; then
  echo "PX4 binary not found: $PX4_BIN" >&2
  echo "Set PX4_AUTOPILOT_DIR to your PX4-Autopilot checkout." >&2
  exit 1
fi

if [[ ! -f "$WORLD_FILE" ]]; then
  echo "Gazebo world not found: $WORLD_FILE" >&2
  exit 1
fi

if [[ "$(basename -- "$WORLD_FILE")" == "earthquake_landslide_apron_world.sdf" ]]; then
  for model in standing_person rescue_randy_sitting; do
    if [[ ! -f "$DISASTER_WORLD_MODELS_DIR/$model/model.sdf" ]]; then
      echo "Required local model wrapper not found: $DISASTER_WORLD_MODELS_DIR/$model/model.sdf" >&2
      echo "Set DISASTER_WORLD_MODELS_DIR to the directory containing disaster_world/models." >&2
      exit 1
    fi
  done
fi

if [[ ! -d "$PX4_WORLDS_DIR" ]]; then
  echo "PX4 Gazebo world directory not found: $PX4_WORLDS_DIR" >&2
  exit 1
fi

if [[ ! -d "$PX4_MODELS_DIR/x500" || ! -d "$PX4_MODELS_DIR/x500_base" ]]; then
  echo "PX4 X500 model files not found under: $PX4_MODELS_DIR" >&2
  exit 1
fi

if pgrep -x px4 >/dev/null; then
  echo "A PX4 SITL process is already running." >&2
  echo "Stop the existing single-drone PX4 session and Gazebo before this swarm." >&2
  exit 1
fi

if command -v gz >/dev/null 2>&1; then
  while IFS= read -r clock_topic; do
    [[ -n "$clock_topic" ]] || continue
    topic_info="$(gz topic -i -t "$clock_topic" 2>/dev/null || true)"
    # Gazebo Transport can list a clock topic with subscribers but no live
    # publisher after the server has stopped. Only a clock publisher means an
    # active world that would conflict with this launch.
    if grep -q '^Publishers ' <<<"$topic_info"; then
      existing_world="${clock_topic#/world/}"
      existing_world="${existing_world%/clock}"
      echo "Gazebo world '$existing_world' is already running." >&2
      echo "Close/stop that Gazebo server before starting $WORLD_NAME." >&2
      exit 1
    fi
  done < <(gz topic -l 2>/dev/null | grep -E '^/world/[^/]+/clock$' || true)
fi

# PX4 opens the Gazebo GUI as soon as instance 0 starts, before the other
# vehicles have joined. Preserve any pre-existing standalone GUI windows, and
# remember the PIDs of GUI windows created by this launch so they can be
# replaced after all three models are present.
GUI_PIDS_BEFORE=" $( { pgrep -f '[g]z sim -g' 2>/dev/null || true; } | tr '\n' ' ') "

# PX4's gz_env.sh resets PX4_GZ_WORLDS to the PX4 checkout's own directory.
# Stage the project-owned world there so PX4 can load it by world name.
install -m 644 "$WORLD_FILE" "$PX4_WORLDS_DIR/$WORLD_NAME.sdf"
install -d "$PX4_WORLDS_DIR/earthquake_landslide_assets"
cp -a "$PROJECT_ROOT/worlds/earthquake_landslide_assets/." \
  "$PX4_WORLDS_DIR/earthquake_landslide_assets/"
install -d "$PX4_MODELS_DIR/survivor_female" "$PX4_MODELS_DIR/survivor_male"
cp -a "$PROJECT_ROOT/worlds/disaster_assets/models/survivor_female/." \
  "$PX4_MODELS_DIR/survivor_female/"
cp -a "$PROJECT_ROOT/worlds/disaster_assets/models/survivor_male/." \
  "$PX4_MODELS_DIR/survivor_male/"
install -d "$PX4_MODELS_DIR/$PX4_SCOUT_MODEL_NAME"
cp -a "$PROJECT_ROOT/worlds/disaster_assets/models/$PX4_SCOUT_MODEL_NAME/." \
  "$PX4_MODELS_DIR/$PX4_SCOUT_MODEL_NAME/"
for model in "$PX4_AMBULANCE_MODEL_NAME" "$PX4_FIREMAN_MODEL_NAME"; do
  source_model="$PROJECT_ROOT/worlds/disaster_assets/models/$model"
  if [[ -d "$source_model" ]]; then
    install -d "$PX4_MODELS_DIR/$model"
    cp -a "$source_model/." "$PX4_MODELS_DIR/$model/"
  elif [[ ! -d "$PX4_MODELS_DIR/$model" ]]; then
    echo "Model '$model' is neither a project model nor present in PX4: $PX4_MODELS_DIR/$model" >&2
    exit 1
  fi
done
for model in standing_person_small rescue_randy_sitting_small; do
  install -d "$PX4_MODELS_DIR/$model"
  cp -a "$PROJECT_ROOT/worlds/disaster_assets/models/$model/." \
    "$PX4_MODELS_DIR/$model/"
done
for model in fire_only smoke_only; do
  install -d "$PX4_MODELS_DIR/$model"
  cp -a "$PROJECT_ROOT/worlds/disaster_assets/models/$model/." \
    "$PX4_MODELS_DIR/$model/"
done
# ParticleEmitter's color-range image path is resolved against Gazebo resource
# paths rather than the model SDF directory. Add each self-contained effect
# folder so its relative materials/textures paths resolve from any cwd.
export GZ_SIM_RESOURCE_PATH="${GZ_SIM_RESOURCE_PATH:+${GZ_SIM_RESOURCE_PATH}:}$PX4_MODELS_DIR/fire_only:$PX4_MODELS_DIR/smoke_only"
if [[ "$(basename -- "$WORLD_FILE")" == "earthquake_landslide_apron_world.sdf" ]]; then
  for model in standing_person rescue_randy_sitting; do
    source_model="$DISASTER_WORLD_MODELS_DIR/$model"
    staged_model="$PX4_MODELS_DIR/$model"
    if [[ -L "$staged_model" ]]; then
      if [[ "$(readlink -f -- "$staged_model")" != "$(readlink -f -- "$source_model")" ]]; then
        echo "Gazebo model path already links elsewhere: $staged_model" >&2
        exit 1
      fi
    elif [[ -e "$staged_model" ]]; then
      echo "Gazebo model path already exists; refusing to overwrite it: $staged_model" >&2
      exit 1
    else
      ln -s -- "$source_model" "$staged_model"
    fi
  done
fi

mkdir -p "$RUN_DIR"
pids=()
stop_started_instances() {
  for pid in "${pids[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
  pkill -x px4-gz_bridge 2>/dev/null || true
  # A failed first PX4 launch can leave the Gazebo server it started behind.
  # The preflight check above guarantees no Gazebo world existed before this
  # launcher started, so these are the server processes owned by this run.
  pkill -x gz-sim-server 2>/dev/null || true
  pkill -x gz-sim 2>/dev/null || true
  pkill -x gz-gui 2>/dev/null || true
  pkill -f '[g]z sim' 2>/dev/null || true
}
trap stop_started_instances ERR

start_instance() {
  local instance="$1" pose="$2" role="$3" standalone="${4:-0}"
  local model_name="$PX4_MODEL_NAME"
  case "$role" in
    scout_0) model_name="$PX4_SCOUT_MODEL_NAME" ;;
    ambulance_0) model_name="$PX4_AMBULANCE_MODEL_NAME" ;;
    fireman_0) model_name="$PX4_FIREMAN_MODEL_NAME" ;;
  esac
  local instance_dir="$PX4_ROOT/build/px4_sitl_default/swarm_instance_$instance"
  local log_file="$RUN_DIR/${role}.log"
  mkdir -p "$instance_dir"
  : >"$log_file"

  if [[ "$standalone" == "1" ]]; then
    (
      cd "$instance_dir"
      exec env PX4_GZ_STANDALONE=1 PX4_GZ_NO_FOLLOW=1 PX4_GZ_WORLD="$WORLD_NAME" \
        PX4_SYS_AUTOSTART=4001 PX4_GZ_MODEL_POSE="$pose" \
        PX4_SIM_MODEL="gz_$model_name" \
        "$PX4_BIN" -i "$instance" \
        > >(stdbuf -oL tr '\r' '\n' | stdbuf -oL awk '!/pxh>/' >>"$log_file") 2>&1
    ) >"$log_file" 2>&1 &
  else
    (
      cd "$instance_dir"
      exec env PX4_GZ_WORLD="$WORLD_NAME" PX4_SYS_AUTOSTART=4001 \
        PX4_GZ_MODEL_POSE="$pose" PX4_SIM_MODEL="gz_$model_name" \
        "$PX4_BIN" -i "$instance" \
        > >(stdbuf -oL tr '\r' '\n' | stdbuf -oL awk '!/pxh>/' >>"$log_file") 2>&1
    ) >"$log_file" 2>&1 &
  fi
  local pid=$!
  pids+=("$pid")
  instance_pids+=("$pid")
  instance_roles+=("$role")
  instance_ids+=("$instance")
  echo "Started PX4 for $role (instance $instance, PID $pid)."
}

wait_for_instance() {
  local role="$1" pid="$2" instance="$3" model_name="$4"
  local log_file="$RUN_DIR/${role}.log"
  for _ in $(seq 1 120); do
    # PX4's readiness banner varies between builds and may never print when
    # optional GCS/power preflight checks fail. The Gazebo bridge world line
    # confirms that this PX4 instance spawned and connected its vehicle.
    if grep -q "\\[gz_bridge\\] world: $WORLD_NAME, model: ${model_name}_${instance}" "$log_file"; then
      echo "$role spawned and connected (instance $instance, system ID $((instance + 1)))."
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "$role PX4 exited; see $log_file" >&2
      grep -aE 'INFO|WARN|ERROR|Error|Failed|Gazebo|gz_bridge|Ready|Spawning|Preflight' \
        "$log_file" | tail -n 25 | cut -c1-240 >&2 || true
      return 1
    fi
    sleep 1
  done

  echo "Timed out waiting for $role; see $log_file" >&2
  grep -aE 'INFO|WARN|ERROR|Error|Failed|Gazebo|gz_bridge|Ready|Spawning|Preflight' \
    "$log_file" | tail -n 25 | cut -c1-240 >&2 || true
  return 1
}

# Instance 0 starts Gazebo. Give it a moment to create the server, then start
# both standalone PX4 instances before waiting for any one vehicle. This way a
# slow join from one PX4 instance cannot prevent the remaining drones spawning.
instance_pids=()
instance_roles=()
instance_ids=()
scout_spawn="-75,-60,1.70,0,0,0"
ambulance_spawn="-67,-60,1.70,0,0,0"
fireman_spawn="-75,-52,1.70,0,0,0"
if [[ "$(basename -- "$WORLD_FILE")" == "earthquake_landslide_apron_world.sdf" ]]; then
  # The compact world has a dedicated, obstacle-cleared launch corner.
  scout_spawn="-76,-61,1.70,0,0,0"
  ambulance_spawn="-72,-61,1.70,0,0,0"
  fireman_spawn="-76,-57,1.70,0,0,0"
elif [[ "$(basename -- "$WORLD_FILE")" == "lidar_validation_world.sdf" \
   || "$(basename -- "$WORLD_FILE")" == "lidar_obstacle_validation_world.sdf" ]]; then
  fireman_spawn="0,0,1.70,0,0,0"
fi
scout_spawn="${PX4_GZ_SCOUT_SPAWN:-$scout_spawn}"
ambulance_spawn="${PX4_GZ_AMBULANCE_SPAWN:-$ambulance_spawn}"
fireman_spawn="${PX4_GZ_FIREMAN_SPAWN:-$fireman_spawn}"
if [[ "$FIREMAN_ONLY" == "1" ]] && \
   [[ "$(basename -- "$WORLD_FILE")" == "lidar_validation_world.sdf" \
      || "$(basename -- "$WORLD_FILE")" == "lidar_obstacle_validation_world.sdf" ]]; then
  IFS=, read -r _fireman_x _fireman_y fireman_z _fireman_roll _fireman_pitch _fireman_yaw \
    <<<"$fireman_spawn"
  if ! awk -v z="$fireman_z" 'BEGIN { exit !(z + 0 >= 1.0) }'; then
    echo "Fireman LiDAR validation spawn altitude must be at least 1.0 m (got $fireman_z)." >&2
    echo "Leave PX4_GZ_FIREMAN_SPAWN unset to use the safe default: $fireman_spawn" >&2
    exit 2
  fi
fi
if [[ "$FIREMAN_ONLY" == "1" ]]; then
  # The default preserves the fleet identity. An alternate instance supports
  # a one-vehicle A/B check against instance-specific PX4 SITL sensor faults.
  start_instance "$FIREMAN_INSTANCE" "$fireman_spawn" fireman_0 0
else
  start_instance 0 "$scout_spawn" scout_0 0
fi
if [[ "$SCOUT_ONLY" == "0" && "$FIREMAN_ONLY" == "0" ]]; then
  sleep 5
  start_instance 1 "$ambulance_spawn" ambulance_0 1
  start_instance 2 "$fireman_spawn" fireman_0 1
fi

failed_roles=()
for index in "${!instance_roles[@]}"; do
  role="${instance_roles[$index]}"
  pid="${instance_pids[$index]}"
  instance="${instance_ids[$index]}"
  expected_model="$PX4_MODEL_NAME"
  case "$role" in
    scout_0) expected_model="$PX4_SCOUT_MODEL_NAME" ;;
    ambulance_0) expected_model="$PX4_AMBULANCE_MODEL_NAME" ;;
    fireman_0) expected_model="$PX4_FIREMAN_MODEL_NAME" ;;
  esac
  if ! wait_for_instance "$role" "$pid" "$instance" "$expected_model"; then
    failed_roles+=("$role")
  fi
done

if ((${#failed_roles[@]})); then
  echo "PX4 did not confirm Gazebo connection for: ${failed_roles[*]}" >&2
  echo "Stopping this partial launch. Logs: $RUN_DIR" >&2
  stop_started_instances
  trap - ERR
  exit 1
fi

trap - ERR

# Keep PX4's GCS link alive during read-only warmup and guarded SITL runs.
# Start the project helper only when UDP 14550 has no existing listener.
if [[ "$SENSOR_DIAGNOSTIC" == "1" ]]; then
  nohup /usr/bin/python3 "$SCRIPT_DIR/fleet_mavlink_heartbeat.py" \
    --duration 35 --fireman-sensor-diagnostic \
    >"$RUN_DIR/fireman_sensor_diagnostic.log" 2>&1 </dev/null &
  echo "$!" >"$GCS_HEARTBEAT_PID_FILE"
  echo "Started Fireman PX4 shell sensor diagnostic. Output: $RUN_DIR/fireman_sensor_diagnostic.log"
elif command -v ss >/dev/null 2>&1 && ! ss -H -lun 'sport = :14550' 2>/dev/null | grep -q .; then
  if /usr/bin/python3 -c 'import pymavlink' >/dev/null 2>&1; then
    nohup /usr/bin/python3 "$SCRIPT_DIR/fleet_mavlink_heartbeat.py" --duration 0 \
      >"$RUN_DIR/fleet_mavlink_heartbeat.log" 2>&1 </dev/null &
    echo "$!" >"$GCS_HEARTBEAT_PID_FILE"
    echo "Started continuous GCS heartbeat on UDP 14550."
  else
    echo "GCS heartbeat helper not started: /usr/bin/python3 cannot import pymavlink." >&2
  fi
else
  echo "Using existing UDP 14550 listener for GCS heartbeats."
fi

# Reopen only this launch's initial GUI after all model insertions. A fresh GUI
# scene avoids the stale/empty view seen when PX4 creates its GUI before the
# standalone vehicles have joined. Focus the camera on the first drone so the
# user can immediately see the swarm's launch area.
if command -v gz >/dev/null 2>&1 && [[ "$HEADLESS" != "1" ]]; then
  while read -r gui_pid; do
    [[ -n "$gui_pid" ]] || continue
    if [[ "$GUI_PIDS_BEFORE" != *" $gui_pid "* ]]; then
      kill -TERM "$gui_pid" 2>/dev/null || true
    fi
  done < <(pgrep -f '[g]z sim -g' 2>/dev/null || true)
  sleep 1
  nohup gz sim -g >"$RUN_DIR/gz_gui.log" 2>&1 </dev/null &
  gui_pid=$!
  for _ in $(seq 1 30); do
    if gz service -l 2>/dev/null | grep -qx '/gui/move_to'; then
      focus_model="${PX4_SCOUT_MODEL_NAME}_0"
      if [[ "$FIREMAN_ONLY" == "1" ]]; then
        focus_model="${PX4_FIREMAN_MODEL_NAME}_${FIREMAN_INSTANCE}"
      fi
      gz service -s /gui/move_to --reqtype gz.msgs.StringMsg \
        --reptype gz.msgs.Boolean -r "data: \"$focus_model\"" --timeout 3000 \
        >/dev/null 2>&1 || true
      if [[ "$FIREMAN_ONLY" == "1" ]]; then
        echo "Gazebo GUI refreshed after Fireman spawned (camera focused on $focus_model)."
      elif [[ "$SCOUT_ONLY" == "1" ]]; then
        echo "Gazebo GUI refreshed after Scout spawned (camera focused on ${PX4_SCOUT_MODEL_NAME}_0)."
      else
        echo "Gazebo GUI refreshed after all three drones spawned (camera focused on ${PX4_SCOUT_MODEL_NAME}_0)."
      fi
      break
    fi
    if ! kill -0 "$gui_pid" 2>/dev/null; then
      echo "Gazebo GUI did not stay open; drones remain spawned in the Gazebo server." >&2
      break
    fi
    sleep 1
  done
elif [[ "$HEADLESS" == "1" ]]; then
  # PX4 may create an initial GUI while starting the world. Remove only GUI
  # processes created by this launch; preserve any pre-existing Gazebo GUI.
  while read -r gui_pid; do
    [[ -n "$gui_pid" ]] || continue
    if [[ "$GUI_PIDS_BEFORE" != *" $gui_pid "* ]]; then
      kill -TERM "$gui_pid" 2>/dev/null || true
    fi
  done < <(pgrep -f '[g]z sim -g' 2>/dev/null || true)
  echo "Headless mode enabled; keeping the Gazebo server and skipping the GUI."
fi

if [[ "$SCOUT_ONLY" == "1" ]]; then
  echo "Scout-only PX4/Gazebo run is active. Logs: $RUN_DIR"
elif [[ "$FIREMAN_ONLY" == "1" ]]; then
  echo "Fireman-only PX4/Gazebo run is active. Logs: $RUN_DIR"
else
  echo "Three PX4/Gazebo vehicles are running. Logs: $RUN_DIR"
fi
echo "Gazebo world: $WORLD_FILE"
if [[ "$FIREMAN_ONLY" == "1" ]]; then
  echo "Spawn pose: fireman_0=$fireman_spawn"
  echo "MAVSDK endpoint: fireman_0=$((14540 + FIREMAN_INSTANCE)) (PX4 instance $FIREMAN_INSTANCE, system ID $((FIREMAN_INSTANCE + 1)))"
elif [[ "$SCOUT_ONLY" == "1" ]]; then
  echo "Spawn pose: scout_0=$scout_spawn"
  echo "MAVSDK endpoint: scout_0=14540"
else
  echo "Spawn poses: scout_0=$scout_spawn, ambulance_0=$ambulance_spawn, fireman_0=$fireman_spawn"
  echo "MAVSDK endpoints: scout_0=14540, ambulance_0=14541, fireman_0=14542"
fi
