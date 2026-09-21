# Continuous-belt drivetrain

`molla-vehicle::track` owns the motor limits and anisotropic friction law.
`molla-solvers::track_forces` binds those laws to stable body, revolute joint and
collider handles in `RigidWorld`. Applications register track descriptors and
command signed belt speeds; they do not set chassis velocities.

Each track has a powered rigid sprocket, its carrier and one or more carrier
collision patches. At each internal Featherstone substep, collision geometry and
compliant normal loads are refreshed. Tangential friction on registered patches
is replaced with belt-relative friction, limited by the normal load and the
lesser of belt and contacted surface friction. Both tangential directions share
one friction ellipse. Motor torque acts equally and oppositely on sprocket and
carrier. Belt traction reacts against sprocket spin and the contacted body;
carrier compensation preserves the external contact moment. Contact points
distributed along each belt produce skid-steering resistance.

Travel is integrated from measured relative sprocket angular velocity, not the
requested speed. Geometry and passive wheel rendering can consume that travel.
Samples report the final substep's loads and torque and accumulated travel.
Failed steps do not publish staged samples. Removed or disabled handles prune
their track registration. Re-register to reset a track; clear registrations
before replacing a scene. Worlds containing registered tracks bypass automatic
sleep until track-aware sleep dependency tracking is implemented.

## Limits

- Reduced rigid continuous-belt model, not individual deformable rubber shoes,
  tension dynamics, derailment, or granular soil excavation.
- Carrier contact shapes must match the support footprint. Roller revolute
  joints and belt rendering are separate from this physical contact model.
- Current integration is the **CPU correctness path**. `RigidWorld`'s
  collision-refreshed Featherstone lane already rejects GPU backends. This does
  not claim GPU parity or a GPU production track solver. A device implementation
  must port contact ownership, coupled motor/belt wrenches, friction bounds and
  state publication together, then pass the same airborne, traction, slip,
  pivot, momentum and timestep tests. No CPU readback fallback is hidden behind
  a GPU API.
- Force integration uses conservative articulated-response bounds to limit
  dissipation. Motor gains and material coefficients are engineering parameters,
  not manufacturer-calibrated CEOL measurements.

## Validation

Use `make test APP_TARGET='-p molla-vehicle --lib track::tests'` and
`make test APP_TARGET='-p molla-solvers --test track_forces'` in the Nix environment.
The application must additionally validate its authored TF tree, controller
bindings, belt animation and scene lifecycle.
