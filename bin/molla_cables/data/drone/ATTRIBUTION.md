# Quadcopter visualization asset

This is a **superseded** visualization. The current orchard renders the Mavic 3
asset documented in `MAVIC_ATTRIBUTION.md`; the retained Innerscene GLB can be
read by `src/drone_asset_legacy.rs`. No mesh from this asset is used currently.

**Innerscene Corp. _Quadcopter with flight electronics_ (2026).**

- Source: https://www.innerscene.com/tools/library/3d-parts/quadcopter-with-flight-electronics-a25968c1
- License: CC0 1.0 Universal, https://creativecommons.org/publicdomain/zero/1.0/
- File: `quadcopter-original.glb`, 274,444 bytes, 5,036 source triangles.
- SHA256: `24d1ac5804afea4076df2b4bc65ce780764d01619eb5fc08900480471cc43a0d`.
- Retrieved 2026-11-18; the GLB's embedded copyright also declares Innerscene
  authorship and CC0 dedication. No external textures or runtime downloads.

`make import-drone-asset` retrieves the pinned file and checks its hash.
The original GLB is unmodified. Molla converts its Y-up coordinates to the
aircraft's Z-up body frame, scales motor spacing to the simulated arm length,
matches rendered blade radius to the actuator disks, and animates the four
separate rotors using simulated motor speeds. Material base colors are
retained; the studio renderer approximates the original PBR shading.

This is an original nominal visualization model, not manufacturer CAD, a
measured aircraft, or the source of the flight/aerodynamic coefficients.
