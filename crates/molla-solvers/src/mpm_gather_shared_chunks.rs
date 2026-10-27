use super::*;

pub(super) const WORKGROUP_BYTES: u32 = 16 * 1024;

fn replace(source: &mut String, old: &str, new: &str) -> Result<()> {
    if source.matches(old).count() != 1 {
        return Err(Error::Solver(
            "Shared gather shader source contract changed".into(),
        ));
    }
    *source = source.replace(old, new);
    Ok(())
}

pub(super) fn source(compensated: bool) -> Result<String> {
    let mut source = if compensated {
        compensated_gather::source()?
    } else {
        include_str!("../shaders/mpm_gather_masks.wgsl").to_owned()
    };
    replace(
        &mut source,
        "                        let pid = p_material[u.grid_layout.w + first + word * 32u + bit];
                        let pos = p_pos[pid].xyz;
                        let xp = (pos - u.grid_origin) * u.inv_cell_size;
                        let base_xyz = floor(xp - vec3<f32>(0.5));
                        let offset = cell - (vec3<i32>(base_xyz) - u.grid_window_cells.xyz);
                        let fx = xp - base_xyz;
                        let wx = quad_weights(fx.x);
                        let wy = quad_weights(fx.y);
                        let wz = quad_weights(fx.z);",
        "                        let cached = gather_chunk[word * 32u + bit];
                        let offset = cell - cached.base;
                        let fx = cached.fx;
                        let wx = cached.wx;
                        let wy = cached.wy;
                        let wz = cached.wz;",
    )?;
    for (old, new) in [
        ("p_mass[pid]", "cached.velocity_mass.w"),
        ("p_vel[pid].xyz", "cached.velocity_mass.xyz"),
        ("p_C[pid]", "cached.affine"),
        ("gather_particles[pid].stress", "cached.stress"),
        ("gather_particles[pid].stiffness", "cached.stiffness"),
    ] {
        replace(&mut source, old, new)?;
    }
    let marker = "                let base = vec3<i32>(floor(xp - vec3<f32>(0.5))) - u.grid_window_cells.xyz;";
    replace(
        &mut source,
        marker,
        &format!(
            "{marker}
                let fx = xp - floor(xp - vec3<f32>(0.5));
                gather_chunk[lane] = GatherChunk(base, fx,
                    quad_weights(fx.x), quad_weights(fx.y), quad_weights(fx.z),
                    vec4<f32>(p_vel[pid].xyz, p_mass[pid]), p_C[pid],
                    gather_particles[pid].stress, gather_particles[pid].stiffness);"
        ),
    )?;
    source.push_str(
        "
struct GatherChunk {
    base: vec3<i32>,
    fx: vec3<f32>,
    wx: vec3<f32>,
    wy: vec3<f32>,
    wz: vec3<f32>,
    velocity_mass: vec4<f32>,
    affine: mat3x3<f32>,
    stress: mat3x3<f32>,
    stiffness: f32,
};
var<workgroup> gather_chunk: array<GatherChunk, 64>;
",
    );
    Ok(source)
}
