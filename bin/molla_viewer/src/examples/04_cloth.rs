// --- Cloth scene ----------------------------------------------------
// 36² = 1296 particles, 4·35² = 4900 distance constraints.
// CPU XPBD at 12 iters runs in ~0.5 ms per substep on a modern host.
const CLOTH_GRID: u32 = 36;
const CLOTH_SIZE: f32 = 1.8;
const CLOTH_TOP_Y: f32 = 5.0;
const CLOTH_PARTICLE_MASS: f64 = 0.05;
const CLOTH_PARTICLE_RADIUS: f64 = 0.02;
// XPBD compliance — `α = 1/k`. Lower = stiffer XPBD pull toward rest.
// Combined with `CLOTH_MAX_STRETCH` (which is a *hard* cap) this gives
// you a soft elastic feel within the allowed range and a brick wall
// at the limit.
const CLOTH_COMPLIANCE: f64 = 1.5e-5;
// Per-spring hard upper-bound on stretch, expressed as a ratio of
// rest length. The XPBD solver's strain-limit pass enforces
// `length ≤ rest · (1 + max_stretch)` regardless of compliance.
// 0.0 → perfectly inextensible (e.g. metal mesh)
// 0.05 → 5 % linen (default for the demo)
// 0.20 → 20 % jersey
// 1.0  → effectively no cap
const CLOTH_MAX_STRETCH: f64 = 0.05;

const CLOTH_SPHERE_RADIUS: f32 = 0.7;
const CLOTH_SPHERE_Y: f32 = 1.2;

fn build_cloth_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    b.set_angular_damping(0.05);
    b.set_linear_damping(0.4);

    let nx = CLOTH_GRID;
    let ny = CLOTH_GRID;
    let size = CLOTH_SIZE as f64;
    // Cloth starts horizontal in the XZ plane, well above the sphere.
    let origin = MVec3::new(-size * 0.5, CLOTH_TOP_Y as f64, -size * 0.5);
    let u_axis = MVec3::new(size, 0.0, 0.0);
    let v_axis = MVec3::new(0.0, 0.0, size);

    let _ = b.add_cloth_grid(
        origin,
        u_axis,
        v_axis,
        nx,
        ny,
        CLOTH_PARTICLE_MASS,
        CLOTH_PARTICLE_RADIUS,
        CLOTH_COMPLIANCE,
        CLOTH_MAX_STRETCH,
        mp::WorldId(0),
    );

    // Static sphere obstacle. `BodyId::NONE` makes it world-anchored
    // — the GPU `particle_vs_shapes` pass treats `shape_body == -1`
    // as static and uses the shape's own transform as world pose.
    b.add_shape_sphere(
        mp::BodyId::NONE,
        CLOTH_SPHERE_RADIUS as f64,
        mp::Transform::from_translation(MVec3::new(0.0, CLOTH_SPHERE_Y as f64, 0.0)),
    )
    .expect("sphere shape");
    // Floor plane so any escaped particle doesn't free-fall forever.
    b.add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
        .expect("plane shape");

    let model = b.build().expect("model");
    // No `BodyVisual` here — cloth renders via the dedicated triangle
    // mesh + the sphere obstacle is spawned as a `ClothObstacle`
    // entity by `apply_scene_switch` / `setup`.
    (model, Vec::new())
}

// Stage-20 high-resolution VBD cloth — 32×32 sheet falling onto a
// kinematic sphere, solved with `VbdSolver` instead of `XpbdSolver`.
// The model topology is identical to `build_cloth_model` (the same
// `add_cloth_grid` builder); only the solver differs at runtime.
// That's by design — the user can flip between VBD and XPBD on the
// same scene via the right-rail toggle and get a side-by-side feel

