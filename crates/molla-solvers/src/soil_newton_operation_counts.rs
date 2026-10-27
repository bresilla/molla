pub(crate) const SHARED_BYTES: u32 = 128;
pub(crate) const FUNCTIONS: [(&str, bool); 27] = [
    ("evaluate", true),
    ("prepare_linear", true),
    ("krylov_cycle", true),
    ("inner_product", true),
    ("scratch_norm", true),
    ("apply_operator", true),
    ("apply_transpose", true),
    ("apply_direction", true),
    ("project_rates", true),
    ("build_inverse", false),
    ("dense_build_inverse", false),
    ("precondition", false),
    ("direct_contact", false),
    ("condensed_inverse", false),
    ("direct_core", true),
    ("condensed_core", true),
    ("condensed_factor", true),
    ("conditioned_cache", true),
    ("conditioned_prepare", true),
    ("conditioned_factor", true),
    ("release_local_candidate", false),
    ("release_merit", false),
    ("release_candidates", false),
    ("condensed_direction", true),
    ("least_squares", true),
    ("conditioned_release", true),
    ("certify_candidate", true),
];

pub(crate) fn source(mut source: String) -> String {
    assert!(!source.contains("operation_counts"));
    for (index, (name, collective)) in FUNCTIONS.iter().enumerate() {
        let marker = format!("fn {name}(");
        assert_eq!(source.matches(&marker).count(), 1, "counter target {name}");
        let start = source.find(&marker).unwrap();
        let at = start + source[start..].find('{').unwrap() + 1;
        let count = format!("operation_count({index}u);");
        source.insert_str(
            at,
            &if *collective {
                format!("\n    if lane==0u {{{count}}}\n")
            } else {
                format!("\n    {count}\n")
            },
        );
    }
    super::replace(
        &mut source,
        "@compute @workgroup_size(64)\nfn newton(@builtin(local_invocation_index) id:u32) {",
        "fn counted_newton(id:u32) {",
    );
    source.push_str(
        r#"
var<workgroup> operation_counts:array<atomic<u32>,32>;
fn operation_count(index:u32) {
    if atomicAdd(&operation_counts[index],1u)==0xffffffffu {atomicOr(&operation_counts[31],1u);}
}
@compute @workgroup_size(64)
fn newton(@builtin(local_invocation_index) id:u32) {
    lane=id;rate_lane=id;
    if lane<32u {atomicStore(&operation_counts[lane],0u);}
    workgroupBarrier();counted_newton(id);workgroupBarrier();storageBarrier();
    if lane<32u {
        var count=atomicLoad(&operation_counts[lane]);
        if lane==30u {count=status.active_count;}
        condensed[lane]=vec2<f32>(bitcast<f32>(count),0.0);
    }
}
"#,
    );
    source
}

pub(crate) fn decode(bytes: &[u8]) -> [u32; 32] {
    assert!(bytes.len() >= 256);
    let counts =
        std::array::from_fn(|i| u32::from_le_bytes(bytes[i * 8..i * 8 + 4].try_into().unwrap()));
    assert_eq!(counts[31], 0, "operation counter overflow");
    assert_eq!(&counts[27..30], &[0, 0, 0], "reserved counters");
    counts
}

#[test]
fn newton_operation_counts_compiler_contract() {
    use crate::soil_newton_gpu::{
        ExperimentalSoilNewtonGpu as Solver, NewtonVariant, shader_source_for,
    };
    let control = super::schur_inverse::source(super::core_source(super::source(
        Solver::cached_contact_source(Solver::cached_delta_source(Solver::stalled_release_source(
            shader_source_for(NewtonVariant::Fischer),
        ))),
    )));
    let candidate = source(control.clone());
    assert_eq!(
        candidate.matches("var<workgroup>").count(),
        control.matches("var<workgroup>").count() + 1
    );
    assert_eq!(
        candidate.matches("var<storage").count(),
        control.matches("var<storage").count()
    );
    assert_eq!(
        candidate.matches("operation_count(").count(),
        FUNCTIONS.len() + 1
    );
    let binary = crate::soil_newton_protected::compile(
        &format!(
            "{}{}256\n{candidate}",
            crate::soil_newton_protected::VOLATILE_MARKER,
            crate::soil_newton_width::MARKER
        ),
        "newton",
    )
    .unwrap();
    if let Some(root) = std::env::var_os("MOLLA_NEWTON_COUNT_ARTIFACTS") {
        let root = std::path::PathBuf::from(root);
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("candidate.wgsl"), candidate).unwrap();
        std::fs::write(
            root.join("candidate.spv"),
            binary
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    }
}

#[test]
fn newton_operation_counts_decode_exact_bits() {
    let mut bytes = [0u8; 256];
    bytes[..4].copy_from_slice(&0x7fc00001u32.to_le_bytes());
    bytes[240..244].copy_from_slice(&601u32.to_le_bytes());
    let result = decode(&bytes);
    assert_eq!(result[0], 0x7fc00001);
    assert_eq!(result[30], 601);
}
