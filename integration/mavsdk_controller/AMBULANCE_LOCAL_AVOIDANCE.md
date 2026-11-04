# Ambulance local LiDAR avoidance

`ambulance_local_avoidance.py` is a pure decision layer for the configured
Ambulance Gazebo `LaserScan`. It has no MAVSDK import and cannot arm or move a
vehicle. A future Ambulance mission executor can call it on each scan while
continuing to pass the unchanged incident goal in local NED coordinates.

## Verified sensor geometry

`worlds/disaster_assets/models/x500_ambulance_lidar/model.sdf` configures
`/ambulance_0/lidar/points` as a `gpu_lidar`, 180 horizontal by 16 vertical
samples, 5 Hz, 0.20 to 25 m, full 360 degree horizontal sweep, and ±15 degree
vertical field. The sensor is level, has zero relative rotation, and is mounted
0.06 m above `base_link`. The conversion uses Gazebo's vehicle axes: +X forward,
+Y left, +Z up. Scan rows are vertical-major and each row contains the
horizontal samples. The planner maps the positive counter-clockwise scan angle
to clockwise PX4 NED yaw by subtracting it from vehicle yaw.

## Decision API

```python
import time
from integration.mavsdk_controller.ambulance_local_avoidance import (
    AmbulanceLocalAvoidance,
)

avoidance = AmbulanceLocalAvoidance()
# At each mission-control update:
directive = avoidance.update(
    scan_feed.latest,
    scan_age_s=(None if scan_feed.received_at is None
                else time.monotonic() - scan_feed.received_at),
    current_north_m=north,
    current_east_m=east,
    current_down_m=down,
    vehicle_yaw_deg=yaw_deg,
    mission_goal_ned=assigned_incident_goal_ned,
    requested_speed_m_s=0.25,
)
```

The returned directive includes state/reason, bounded horizontal NED velocity,
zero vertical velocity, clearance, stopping distance, six sector summaries,
the original assigned goal, and whether a prior detour has ended. It is a
suggestion only; a future executor must own Offboard setpoint streaming and
PX4 state checks, and must keep sending a zero-velocity hold when the state is
`HOLD`. A stale scan (over 0.5 s), malformed scan, absent valid returns, or an
unobserved candidate corridor yields `HOLD`. No vertical avoidance command is
generated. Above/below summaries only cover obstacles visible inside this
LiDAR's ±15 degree vertical field.

The path check uses a 0.55 m half-width and 0.45 m half-height around the
vehicle body, centered 0.06 m below the sensor. It selects a short (2.5 m)
horizontal corridor, rejects candidates inside the computed stop distance
(braking distance + reaction travel + 0.33 m body radius + 0.20 m margin),
slows in the caution band, and considers headings within 75 degrees of the
global goal bearing. A selected detour is rechecked each scan and the direct
mission corridor must be clear for three scans before the original goal is
resumed. These are conservative SITL prototype values, not validated aircraft
performance limits or a collision-free guarantee.

## Current integration boundary

The incident assignment planner emits dry-run proposals and the queue ledger
stores operator scheduling records; the repository does not currently contain
a live Ambulance assignment receiver or mission executor. This module is the
tested local-avoidance interface for that future executor. It does not alter
global assignment or victim coordinates, and no flight-command code has been
connected or run as part of its implementation.

Run the synthetic tests with:

```bash
python3 -m unittest tests.test_ambulance_local_avoidance -v
```

There has been no live Gazebo scan test or autonomous flight validation for
this module.
