# Three-Drone Urban Search-and-Rescue: Discrete 3D Extension

This project extends the validated RELAY learning problem with a discrete
three-dimensional drone flight layer. It is not a continuous physics simulator:
the policy selects actions on a finite `(x, y, z)` airspace grid.

## Roles

| Existing internal role | Drone presentation role | Mission responsibility |
| --- | --- | --- |
| Scout | Reconnaissance Drone | Searches with the wider local observation radius and selectively reports incidents. |
| Ambulance | Medical-Support Drone | Resolves stranded-person incidents with the existing service action. |
| Fireman | Fire-Response Drone | Resolves fire-hotspot incidents with the existing service action. |

## Discrete 3D flight model

- Drones hold a true discrete `(x, y, z)` state with four flight levels.
- The physical action space includes `ASCEND` and `DESCEND`.
- Buildings occupy the bottom two flight levels; drones can only cross them
  above their modelled height.
- Specialists must descend to ground level to start the existing three-tick
  `INTERACT` service action, then automatically return to cruise altitude.
- The policy learns horizontal motion, altitude, message timing, and recipient
  selection; only continuous flight dynamics remain outside scope.

## Drone experiment configurations

- `configs/experiments/d1_drone_nocomm.yaml`
- `configs/experiments/d2_drone_when.yaml`
- `configs/experiments/d3_drone_when_who.yaml`

All three use `configs/presets/drone_urban_3d.yaml`. Drone replay files
automatically use the isometric 2.5D renderer. The initial visualisation uses
deterministic generated seeds; generate a dedicated frozen manifest before the
final experiment comparison.
