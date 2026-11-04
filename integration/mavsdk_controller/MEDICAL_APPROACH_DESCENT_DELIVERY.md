# Medical approach, descent, and first-aid lifecycle

## What is implemented

`medical_response.py` is a pure Python, fail-closed lifecycle evaluator. It
contains the incident-to-delivery state sequence and descent admission checks.
It has no MAVSDK dependency and cannot arm, take off, move, land, set a mode,
or publish a flight setpoint. Its bounded downward velocity is an advisory
value only; `flight_commands_sent` remains false.

Descent evaluation requires fresh finite telemetry, a connected/armed/airborne
vehicle, healthy vehicle status, stable horizontal speed and position error,
vertical speed within bounds, and a separately verified descent-corridor
clearance. Missing or stale inputs fail closed to `SAFETY_HOLD`. Position
stability must be observed for a configured duration before `DELIVERY_READY`.
The current coordinator does not provide verified below-vehicle clearance or
a time-window stability estimate, so its terminal monitor reports both as
unknown/unverified. The state machine has only been exercised with mocked
telemetry; no live descent was attempted.

The coordinator and vehicle-agent status vocabularies now include
`VICTIM_APPROACH`, `DESCENDING`, `STABILIZING`, `DELIVERY_READY`,
`FIRST_AID_DELIVERED`, `MISSION_COMPLETED`, and `SAFETY_HOLD`. The coordinator
accepts a delivery transition only when its message explicitly identifies a
`GROUND_TRUTH_TEST`, a `virtual_simulation_event`, no physical release, and no
treatment claim. The coordinator terminal state includes a First-Aid Response section that shows
the incident, assigned vehicle, coordinate provenance, target distance when
fresh local-NED telemetry permits it, available altitude/vertical speed,
workflow history, and the unknown safety metrics. Reported mission status is
labelled as coordinator-received status, not proof of physical execution.

Inspection of `x500_ambulance_lidar/model.sdf` found only the X500 include and
LiDAR add-on; this model declares no payload joint, release actuator, or first
aid mechanism. The supported delivery representation is therefore a clearly
labelled virtual kit event only. It does not indicate physical delivery or
medical treatment.

## Validation

Run the command-free tests:

```bash
python3 -m unittest discover -s tests -p 'test_medical_response.py' -v
```

The state machine must be connected to fresh, validated clearance and a
windowed stability monitor before any descent demo. Any actual SITL descent
requires explicit user approval. This code does not enable one.
