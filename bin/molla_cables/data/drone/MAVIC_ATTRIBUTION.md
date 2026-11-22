# DJI Mavic 3 visual asset

**"DJI Mavic 3" by llirikslon**, published under
[Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).
[Original model](https://sketchfab.com/3d-models/dji-mavic-3-c5a5abae1dea468ab73b1bdc7d616fa6)
([short link](https://skfb.ly/o8rsR)); [creator](https://sketchfab.com/llirikslon).

The bundled `mavic-3.glb` is the reduced/conversion derivative distributed by
**Lexik and Krutnet / Skyfront**, not code from that project. Their
[attribution](https://github.com/Lexik-mqslll/Skyfront/blob/31f4791441329e969968d3793fd3f41b5e8c516b/README.md#third-party-assets)
explicitly separates this CC BY asset from their GPL software.
Only the model data is imported; no game logic, scripts or other assets are used.

- Repository commit: `31f4791441329e969968d3793fd3f41b5e8c516b`
- Source path: `src/main/resources/assets/sfw/models/entity/drone.glb`
- [Pinned download](https://raw.githubusercontent.com/Lexik-mqslll/Skyfront/31f4791441329e969968d3793fd3f41b5e8c516b/src/main/resources/assets/sfw/models/entity/drone.glb)
- SHA256: `48afb1bb7fe461e1e6ef1df824a768098a1b170f920fb98ed38ed24eeb33a7a5`
- Imported 2026-11-18. 12,701 source triangles, two PNG albedo textures.
- License text: `CC-BY-4.0.txt` beside this file.

Molla modifications are runtime coordinate conversion, bilinear fitting of the
body's four hub locations to the simulation's rotor centers, flattening of the
propeller planes, radius fitting, and animation using integrated motor phases.
Source UVs and normals are retained/transformed; albedo uses filtered sRGB mipmaps.
The GLB file itself is unchanged. Back faces are generated for two-sided rendering.

This is a visual skin scaled to the existing illustrative aircraft. Flight mass,
inertia, motor/controller parameters, airflow, leaves and trees are unchanged;
they do not represent calibrated DJI Mavic 3 specifications. No DJI endorsement
is implied. DJI/Mavic marks and depicted product design belong to their owners.

Suggested credit for images/video: **DJI Mavic 3 model by llirikslon (CC BY 4.0),
reduced/converted by Lexik and Krutnet; adapted for Molla.** Keep the source and
license links with public exports.
