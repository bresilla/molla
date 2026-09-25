# Generic sensors (Webots set)

Molla covers every sensor type in the [Webots sensor guide](https://cyberbotics.com/doc/guide/sensors).
The models are engine-agnostic. They read Molla's model, state, sensor scene
and contacts. Anything a physics engine does not know (which way is north,
where the world sits on the planet, where the lights are, which packets were
sent) is a plain input from the host application.

| Webots device | Molla | Inputs from the host |
|---|---|---|
| Accelerometer, Gyro | `ImuSensor` (`accelerometer`, `gyroscope` channels) | mounts |
| InertialUnit | `inertial_unit(&WorldFrame, rotation)` | world up/north |
| Compass | `compass(&WorldFrame, rotation)` | world up/north |
| GPS | `Gps::sample(position, time)` | `WorldFrame`, `GeoReference` (world origin on WGS84) |
| DistanceSensor | `DistanceSensors` (laser, infra-red, sonar) | sensor world poses |
| LightSensor | `LightSensors` | `LightEnvironment` (lights, sky irradiance, up) |
| PositionSensor | `joint_reading(model, state, joint)` | joint index |
| Radar | `Radars` | target bodies and cross-sections |
| RangeFinder | `range_finder(...)` → `TiledCameraConfig` with the depth channel only | — |
| Receiver (and Emitter) | `RadioMedium` | queued packets, endpoint poses, line of sight |
| TouchSensor | `TouchSensor` (bumper, force, force-3d) | contact forces this step |
| Camera | `TiledCamera`, `CameraBatch` | — |
| Recognition | `recognize(...)` over a camera's shape-index and depth channels | shape → object mapping |
| Lidar | `LidarBatch` | — |

## Frames and conventions

- **`WorldFrame`** names the world's up and north axes. East completes the
  right-handed east-north-up basis. Constants cover Z-up/+Y-north
  (REP-103) and Y-up/−Z-north.
- **`GeoReference`** places the world origin on the WGS84 ellipsoid. World
  positions are treated as east-north-up offsets in the local tangent plane
  there. `enu_to_geodetic` and `geodetic_to_enu` convert exactly through ECEF,
  round-tripping to below a micrometre over kilometres.
- **GPS** reports the geodetic fix, the east-north-up position, and velocity by
  finite difference between fixes (a real receiver's speed); `reset` forgets
  the previous fix.
- **Compass** returns north in sensor axes and the clockwise heading of the
  sensor's +X.
- **InertialUnit** returns the sensor's rotation in east-north-up and its
  roll/pitch/yaw (yaw zero facing east, counter-clockwise positive).
- **Distance sensors** look along +X:
  - Rays are the axis plus a ring at half the aperture (`cone_rays`).
  - A laser reports the axis ray, a sonar the nearest echo, and infra-red the
    Gaussian-weighted mean, where misses count as the maximum range.
  - A lookup table maps distance to the output value.
- **Light sensors** face +X:
  - Each source adds its irradiance times the cosine to the sensor normal,
    when a shadow ray finds it unblocked. Directional sources are given as
    W/m² facing the light; point sources as W/sr.
  - The sky adds `sky · (1 + n·up) / 2`.
- **Radar** looks along +X:
  - Each named target body inside range and field of view is reported with
    distance, azimuth (left of +X), elevation, and radial speed (positive
    receding, relative to the radar's own mount).
  - Received power follows the radar equation with unit antenna gains:
    `P_t + 10 log10(λ²σ / ((4π)³ r⁴))` in dBm.
  - Occluded targets are dropped unless occlusion is off.
- **Position sensor** reads the first coordinate and degree of freedom of a
  joint from host-resident state.
- **Touch sensor** sums contact forces on its body:
  - bumper: touching;
  - force: magnitude;
  - force-3d: the force in sensor axes.
- **Radio:**
  - Radio reaches receivers on a matching channel (`ALL_CHANNELS` matches
    any) within both ranges.
  - Infra-red also needs each end inside the other's aperture and a clear line
    of sight, which the host answers.
  - Serial ignores distance.
  - Deliveries carry inverse-square signal strength and the direction to the
    emitter in receiver axes.
- **Recognition** groups pixels by host-given object id and reports each
  object's pixel count, image box, mean visible point and visible extent, in
  camera axes (−Z forward, +Y up).

## Ray queries

Distance, light and radar sensors cast through a `RayQuery`: a batch of
rays in, the nearest hit distance and body of each out.

- `new(engine, …)` and `update(engine, scene, state, …)` cast on the GPU
  sensor scene through `SceneRays`. Each update waits for the device, so on a
  device shared with a renderer it also waits for queued frames.
- `host(…)` and `update_with(query, …)` take any `RayQuery`:
  - `RigidRays` casts on a live `RigidScene` on the CPU. Keep its
    `CpuRaycaster` between updates so prepared collider geometry is reused.
  - A host with its own physics engine can answer the rays itself.

These sensors cast a few rays each, so the CPU query is the cheaper choice
beside a renderer. Both paths share one reduction.

Noise models remain outside this layer (see `SENSORS.md` S10).

## Tests

- `environment`, `touch`, `radio` and `recognition` unit tests run without a
  GPU:

  ```sh
  make test APP_TARGET='-p molla-sensors --no-default-features --lib'
  ```

- `ray_sensors::tests::host_ray_queries_drive_the_sensor_models` runs the
  distance, light and radar models on a CPU `RigidRays` query (no adapter
  needed):

  ```sh
  make test APP_TARGET='-p molla-sensors --features gpu --lib ray_sensors'
  ```

- `sensor_generic_gpu` checks distance, light, radar and joint readings on a
  Z-up scene whose answers follow from its geometry. It includes a sonar that
  hears the ground before the wall, an occluded radar target and a sensor
  under an overhang:

  ```sh
  make test-gpu P=molla-sensors T=sensor_generic_gpu A='--test-threads=1'
  ```
