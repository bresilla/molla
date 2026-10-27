use super::*;

pub(super) fn source() -> Result<String> {
    let mut source = include_str!("../shaders/mpm_gather_masks.wgsl").to_owned();
    for (old, new) in [
        ("var stiffness_sum = 0.0;", "var stiffness_sum = 0.0;\n    var stiffness_tail = 0.0;"),
        ("stiffness_sum += w * w * gather_particles[pid].stiffness;", "let term = w * w * gather_particles[pid].stiffness;
                        let next = stiffness_round(lane, stiffness_sum + term);
                        let large = max(stiffness_sum, term);
                        let small = min(stiffness_sum, term);
                        let error = stiffness_round(lane, stiffness_round(lane, large - next) + small);
                        stiffness_tail = stiffness_round(lane, stiffness_tail + error);
                        stiffness_sum = next;"),
        ("select(stiffness_sum, 0.0,", "select(stiffness_sum + stiffness_tail, 0.0,"),
    ] {
        if source.matches(old).count() != 1 { return Err(Error::Solver("Compensated gather shader source contract changed".into())); }
        source = source.replace(old, new);
    }
    source.push_str("\nvar<workgroup> stiffness_rounding: array<atomic<u32>, 64>;\nfn stiffness_round(lane: u32, value: f32) -> f32 {\n    atomicStore(&stiffness_rounding[lane], bitcast<u32>(value));\n    return bitcast<f32>(atomicAdd(&stiffness_rounding[lane], 0u));\n}\n");
    Ok(source)
}
