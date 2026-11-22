# Stop-and-spray orchard

```sh
make run-ag-orchard
make test-ag-spray
make capture-ag-orchard ORCHARD_SECONDS=140 ORCHARD_VIDEO=artifacts/ag-spray/full-1080p.mp4 ORCHARD_ARGS='--state artifacts/ag-spray/full-state.json --keyframes artifacts/ag-spray/frames'
make verify-ag-spray ORCHARD_SECONDS=140 ORCHARD_VIDEO=artifacts/ag-spray/full-1080p.mp4
make verify-ag-retention AG_BASELINE_REPORT=artifacts/ag-spray/settling/ag-orchard-full-1080p.json AG_RETENTION_REDUCTION=0.30 ORCHARD_VIDEO=artifacts/ag-spray/full-1080p.mp4
```

The original compact AG quadrotor has a tank, four articulated-looking arms,
motor-driven propellers, landing skids, plumbing and four downward nozzles. Its
rotors follow the integrated motor phases. This is a separate `--ag-spray` mode;
the old orchard, flight lab and Mavic asset remain available.

The mission travels in a serpentine route over all 16 trees, returns to the first
tree and holds. Every travel leg uses the bounded-jerk reference and existing
rigid-body controller. Spraying requires position error below 10 cm, speed below
0.1 m/s and a near-level shaft after at least 0.5 s of stable arrival. Each tree
gets two seconds of nozzle-on time. Losing the hover condition closes the valve.
The native `Fly row` and motor switches also inhibit spraying. Reset resets the
mission and liquid accounting. Pause pauses the whole simulation.

The default leaf attachment threshold scale is 0.85, up from 0.74. Only the
release threshold changes; spring stiffness, damping, mass, wind and detached
leaf dynamics are unchanged. This targets another 30% fewer falling leaves
relative to the previous 1,482-leaf run, not 30% fewer visible leaves or 30%
relative to the original baseline. `verify-ag-retention` accepts an explicit
baseline and requested reduction, with a three-percentage-point tolerance.
The command above requires a measured 27–33% reduction with the same scene,
wind and mission duration. The report gives the actual percentage. Defaults
remain the original baseline and 20% target for reproducing older captures.

Four nozzles emit weighted water parcels from an 84-degree base cone, sampled
uniformly over solid angle before a 2x nozzle-axis velocity gain. Sideways launch
velocity is unchanged, so the resulting full angle is approximately 48.5 degrees.
The downward component is 27.0–36.4 m/s instead of 13.5–18.2 m/s. They use
160–240 micrometre droplet diameters, with 7,200 parcels per second. The current
base pressure multiplier is 27 relative to the initial demo: base nozzle speed is
18.19 m/s before the directional gain. The relative nozzle flow
coefficient is 1/3, preserving the previous flow with the faster nozzle.
Initial velocity includes aircraft linear
and angular motion. Gravity and Schiller–Naumann drag advance droplets through
the actual shared GPU air field; sampling is batched with the existing canopy
queries. A spatial hash and swept intersections deposit water on moving elliptical
leaf patches or the ground. Parcels are not clipped at the airflow grid boundary
or deleted at a fixed age. A bounded population closes emission if capacity is
exhausted; the verification requires that this never happens in the full route.

The 0.75 L reservoir and approximately 0.020785 L/s illustrative flow are accounted in the report:
remaining tank + emitted = initial tank; canopy + ground + airborne = emitted.
The unchanged sixteen two-second stops consume about 0.665 L, leaving 0.085 L.
The prescribed diameter range is artistic tuning, not an identified atomizer
breakup law or an absolute pressure specification. Pressure scaling is credited
to TeeJet's technical information in `ACKNOWLEDGEMENT.md`, section 39.
This is one-way water-to-air sampling: aircraft mass/inertia stay fixed and water
does not add leaf mass, force, liquid films, chemical effects, evaporation, breakup
or nozzle recoil. Leaf interception is a planar approximation to curved blades.
The mist is drawn in neutral white using Gaussian density kernels and subpixel grains at
the simulated parcel positions. Kernel width smooths the sampled density; it is
not a physical droplet diameter. Kernel opacity is 0.012, up from 0.007; kernel
radii and core grains are unchanged. Whiter, more visible mist does not increase
water flow or change motion. There are no animated spray paths or prescribed
sideways movement. The earlier 30–60 micrometre aerosol entrained rapidly into
swirling updrafts. The current larger droplets have more inertia and a larger
gravitational settling slip, while the renderer keeps its fine mist appearance.
Water density and gravity remain 998 kg/m3 and 9.81 m/s2: the change is physical
droplet size, not an artificial gravity multiplier or altered air velocities.
Each parcel still represents a fixed liquid volume, so it represents fewer
physical drops at larger diameter; total flow and reservoir use are unchanged.
The jet is
boosted downward at emission, not forcibly sped up throughout its trajectory.
The 2x directional velocity gain is an authored injection condition, not a claim
that hardware pressure doubles. Droplets still respond to local wind;
no downward force is fabricated after emission and updrafts are not suppressed.
The shared air includes rotor loads, gusts and canopy response. This is a coarse
15 cm airflow grid, not blade-resolved or experimentally validated spray CFD.
An isotropic continuous-random-walk closure adds correlated unresolved air
fluctuations to the local velocity used by droplet drag. Its per-component RMS
is 12.5% of resolved air speed, with a 0.12 s correlation time. This intermediate
preset halves the former 25% fluctuations in all
directions, retaining a more downward-focused plume. No velocity is clamped to
point downward; genuine local updrafts remain possible. These are prescribed
demo parameters, not a solved k-epsilon/LES model. Independent seeded parcel
histories produce progressive dispersion rather than widening the render sprites
or prescribing positions. Zero intensity recovers the resolved-wind-only model.
References are acknowledged in `ACKNOWLEDGEMENT.md`, section 40.
Reports record first crossings at 0.25/0.75/1.5/2.5 m below injection and their
horizontal RMS spread about the mean relative displacement. Deposition truncates
the crossing segment; removed water cannot contribute deeper measurements.
The visible footprints are not a calibrated mist opacity model. The 262,144
parcel ceiling accommodates the entire route even if none deposits; parcels are
neither deleted nor frozen to maintain that limit. Reports include sampled air
speed, wind-sampling count and maximum horizontal drift from emission.
The `settling` report also records liquid-volume-weighted airborne/upward
residence times and maximum rise above each parcel's injection height. These
describe simulated transport, not measurements or a guarantee against all lift.
The historical `verify-ag-settling` compares the settling-preset 140-second run to the previous retention
video: flight, leaf shedding, wind, dose and turbulence must match, while residual
airborne water and maximum drift must each fall by at least half. Upward liquid
residence must stay below 25% and rise above injection below 0.5 m. These are
demonstration acceptance bounds, not universal physical limits or motion clamps.
That historical comparison requires unchanged leaf shedding, so it does not
apply to the later 30%-less-shedding preset. Use the explicitly configured
retention comparison and the full spray verification for the current video.
No efficacy, dosing, drift-safety or real-aircraft calibration is claimed.

The old air-vapor effect stays off; the visible cloud is emitted liquid. The
existing leaf dynamics, wind and branch response are unchanged. Manufacturer
layout inspiration and the drag reference are acknowledged in
[`ACKNOWLEDGEMENT.md`, section 38](../../ACKNOWLEDGEMENT.md). No DJI mesh or
OpenFOAM source was copied into this mode.
