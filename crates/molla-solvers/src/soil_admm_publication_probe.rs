#[cfg(test)]
use super::*;

const INPUT_CHECK: &str = r#"
    if params.pad0!=0u {
        let count=admm_indices.count;
        if lane==0u && (count>params.contacts || count!=atomicLoad(&status.active_count)) {fault(1u,0u);}
        for(var slot=lane;slot<min(count,params.contacts);slot+=64u) {
            let c=admm_indices.ids[slot];
            if c>=params.contacts {fault(1u,0u);continue;}
            if !(contacts[c].soil_mass.w>0.0) {fault(1u,c+1u);}
            if slot>0u && admm_indices.ids[slot-1u]>=c {fault(1u,c+1u);}
        }
    }
    storageBarrier();
"#;
const HELPERS: &str = r#"
struct AdmmIndices {count:u32,ids:array<u32>}
@group(0) @binding(12) var<storage,read> admm_indices:AdmmIndices;
fn admm_publish_count()->u32 {if params.pad0==0u {return params.contacts;}return admm_indices.count;}
fn admm_publish_contact(slot:u32)->u32 {if params.pad0==0u {return slot;}return admm_indices.ids[slot];}
"#;
const ORIGINAL: &str = "for (var c=0u;c<params.contacts;c++) {if contacts[c].soil_mass.w>0.0 {";
const REPLACEMENT: &str = "for (var slot=0u;slot<admm_publish_count();slot++) {let c=admm_publish_contact(slot);if contacts[c].soil_mass.w>0.0 {";
const MARKER: &str = "    if lane==0u {stop=atomicLoad(&status.fault);}\n    let invalid_inputs=";

fn numerical_guard() -> String {
    include_str!("../shaders/soil_publication_numerical_guard.wgsl")
        .replace("params.pad2", "params.pad1")
        .replace("publication_fault(", "fault(")
}

pub(super) fn source(mut source: String) -> String {
    assert!(!source.contains("params.pad0"));
    assert!(!source.contains("params.pad1"));
    assert_eq!(source.matches(ORIGINAL).count(), 2);
    assert_eq!(source.matches(MARKER).count(), 1);
    source = source
        .replace(ORIGINAL, REPLACEMENT)
        .replace(MARKER, &format!("{INPUT_CHECK}{MARKER}"));
    source.push_str(HELPERS);
    let marker = "        var wall_bits=0u;";
    assert_eq!(source.matches(marker).count(), 1);
    source = source.replace(marker, &format!("{}{marker}", numerical_guard()));
    source
}

fn sorted_arithmetic() -> String {
    let sorted = crate::soil_rate_encoding::sorted_arithmetic(64);
    let tail = "return vec2<f32>(s,rate_round(small-z));";
    assert_eq!(sorted.matches(tail).count(), 1);
    sorted.replace(
        tail,
        "let low=rate_round(small-z);return vec2<f32>(s,select(low,0.0,low==0.0));",
    )
}

pub(super) fn sorted_source(source: String) -> String {
    let original = crate::soil_rate_encoding::arithmetic(64);
    assert_eq!(source.matches(&original).count(), 1);
    self::source(source.replace(&original, &sorted_arithmetic()))
}

#[cfg(test)]
pub(super) fn check_sorted_sum_gpu(
    engine: &Engine,
    root: &std::path::Path,
    read: impl Fn(&wgpu::Buffer) -> Vec<u8>,
) -> serde_json::Value {
    let values = [
        0.,
        -0.,
        1.,
        -1.,
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        f32::from_bits(1),
        -f32::from_bits(1),
        f32::MAX,
        -f32::MAX,
        1e-20,
        -1e-20,
        1e20,
        -1e20,
        f32::from_bits(0x3f800001),
    ];
    let mut cases = Vec::new();
    let mut push = |a: f32, b: f32| {
        if a.is_finite() && b.is_finite() && (a + b).is_finite() {
            cases.push([a, b]);
        }
    };
    for a in values {
        for b in values {
            push(a, b);
        }
    }
    let mut seed = 0x12345678u32;
    for _ in 0..100000 {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        let a = f32::from_bits(seed);
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        push(a, f32::from_bits(seed));
    }
    assert!(cases.len() > 99000);
    let device = engine.device().unwrap();
    let bytes = bytemuck::cast_slice(&cases);
    let buffer = || {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ADMM sum parity fixture"),
            size: bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    };
    let input = buffer();
    engine.queue().unwrap().write_buffer(&input, 0, bytes);
    std::fs::write(root.join("sum-inputs.bin"), bytes).unwrap();
    let entry = "\n@group(0) @binding(0) var<storage,read> inputs:array<vec2<f32>>;\n@group(0) @binding(1) var<storage,read_write> outputs:array<vec2<f32>>;\n@compute @workgroup_size(64) fn sums(@builtin(global_invocation_id) id:vec3<u32>,@builtin(local_invocation_index) lane:u32) {rate_lane=lane;if id.x<arrayLength(&inputs) {outputs[id.x]=rate_sum(inputs[id.x].x,inputs[id.x].y);}}";
    let mut results = Vec::new();
    for (mode, prefix) in [
        crate::soil_rate_encoding::arithmetic(64),
        sorted_arithmetic(),
    ]
    .iter()
    .enumerate()
    {
        let source = format!("{prefix}{entry}");
        std::fs::write(root.join(format!("sum-{mode}.wgsl")), &source).unwrap();
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ADMM sum parity"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &module,
            entry_point: Some("sums"),
            compilation_options: Default::default(),
            cache: engine.pipeline_cache(),
        });
        let output = buffer();
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output.as_entire_binding(),
                },
            ],
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups((cases.len() as u32).div_ceil(64), 1, 1);
        }
        engine.queue().unwrap().submit([encoder.finish()]);
        let result = read(&output);
        std::fs::write(root.join(format!("sum-{mode}-arithmetic.bin")), &result).unwrap();
        results.push(result);
    }
    let mismatch = results[0]
        .chunks_exact(4)
        .zip(results[1].chunks_exact(4))
        .position(|(a, b)| a != b);
    let result = serde_json::json!({"cases":cases.len(),"exact":mismatch.is_none(),"first_mismatch_word":mismatch,"acceptance":false});
    std::fs::write(
        root.join("sum-arithmetic.json"),
        serde_json::to_vec_pretty(&result).unwrap(),
    )
    .unwrap();
    assert!(
        mismatch.is_none(),
        "GPU sum parity mismatch at word {mismatch:?}"
    );
    result
}

#[test]
fn admm_publication_sorted_sum_contract() {
    let round = std::hint::black_box::<f32>;
    let original = |a: f32, b: f32| {
        let s = round(a + b);
        let vb = round(s - a);
        let va = round(s - vb);
        [s, round(round(a - va) + round(b - vb))]
    };
    let sorted = |a: f32, b: f32| {
        let (large, small) = if b.abs() > a.abs() { (b, a) } else { (a, b) };
        let s = round(large + small);
        let z = round(s - large);
        let low = round(small - z);
        [s, if low == 0. { 0. } else { low }]
    };
    let values = [
        0.,
        -0.,
        1.,
        -1.,
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        f32::from_bits(1),
        -f32::from_bits(1),
        f32::MAX,
        -f32::MAX,
        1e-20,
        -1e-20,
        1e20,
        -1e20,
        f32::from_bits(0x3f800001),
    ];
    let mut count = 0;
    let mut compare = |a: f32, b: f32| {
        if a.is_finite() && b.is_finite() && (a + b).is_finite() {
            assert_eq!(
                original(a, b).map(f32::to_bits),
                sorted(a, b).map(f32::to_bits),
                "a={a:?} b={b:?}"
            );
            count += 1;
        }
    };
    for a in values {
        for b in values {
            compare(a, b);
        }
    }
    let mut seed = 0x12345678u32;
    for _ in 0..100000 {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        let a = f32::from_bits(seed);
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        compare(a, f32::from_bits(seed));
    }
    assert!(count > 99000);
    let control = source(shader_source());
    let candidate = sorted_source(shader_source());
    let start = candidate.find("fn rate_sum(").unwrap();
    let end = candidate.find("fn rate_add(").unwrap();
    let old_start = control.find("fn rate_sum(").unwrap();
    let old_end = control.find("fn rate_add(").unwrap();
    assert_eq!(candidate[start..end].matches("rate_round(").count(), 3);
    assert_eq!(
        control[old_start..old_end].matches("rate_round(").count(),
        6
    );
    assert_eq!(
        candidate.replacen(&candidate[start..end], &control[old_start..old_end], 1),
        control
    );
    let module = naga::front::wgsl::parse_str(&candidate).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
    if let Some(path) = std::env::var_os("MOLLA_ADMM_SORTED_PUBLICATION_ARTIFACT") {
        std::fs::write(path, candidate).unwrap();
    }
}

#[test]
fn admm_publication_shader_preserves_certification() {
    let control = shader_source();
    let candidate = source(control.clone());
    assert_eq!(
        candidate
            .replace(INPUT_CHECK, "")
            .replace(REPLACEMENT, ORIGINAL)
            .replace(HELPERS, ""),
        control.replace(
            "        var wall_bits=0u;",
            &format!("{}        var wall_bits=0u;", numerical_guard())
        )
    );
    for marker in [
        "for (var c=lane;c<params.contacts;c+=64u)",
        "for (var c=lane;c<max(params.contacts,1u);c+=64u)",
        "energy_parts[lane]=energy",
        "if rejected {outputs[c]=Output();}",
    ] {
        assert!(control.contains(marker));
        assert_eq!(
            candidate.matches(marker).count(),
            control.matches(marker).count()
        );
    }
    let module = naga::front::wgsl::parse_str(&candidate).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
    if let Some(path) = std::env::var_os("MOLLA_ADMM_PUBLICATION_ARTIFACT") {
        std::fs::write(path, candidate).unwrap();
    }
}

#[test]
fn admm_publication_numerical_guard_contract() {
    let common = include_str!("../shaders/soil_publication_numerical_guard.wgsl");
    let guard = numerical_guard();
    assert_eq!(
        guard
            .replace("params.pad1", "params.pad2")
            .replace("fault(", "publication_fault("),
        common
    );
    assert!(guard.contains("normal*min(lambda,rho*strict.x)+tangent-projected"));
    assert!(guard.contains("abs(residual)>1e-6"));
    assert!(guard.contains("!all(abs(residual)<=vec3<f32>(1e-6))"));
    let candidate = source(shader_source());
    assert_eq!(candidate.matches(&guard).count(), 1);
    assert!(candidate.find(&guard).unwrap() < candidate.find("let rejected=").unwrap());
    assert!(
        include_str!("soil_newton_numerical_guard.rs")
            .contains("include_str!(\"../shaders/soil_publication_numerical_guard.wgsl\")")
    );
}

#[test]
fn admm_publication_index_completeness_contract() {
    let valid = |mass: &[f32], ids: &[u32], declared: u32| {
        let positive = mass.iter().filter(|&&v| v > 0.).count();
        declared as usize <= mass.len()
            && declared as usize == positive
            && ids.len() >= positive
            && ids[..positive]
                .iter()
                .all(|&i| (i as usize) < mass.len() && mass[i as usize] > 0.)
            && ids[..positive].windows(2).all(|v| v[0] < v[1])
    };
    for count in [0, 1, 63, 64, 65, 251, 256, 4096, 16384] {
        let masses: Vec<_> = (0..count)
            .map(|i| if i % 7 == 0 { 1. } else { 0. })
            .collect();
        let ids: Vec<_> = (0..count)
            .filter(|i| i % 7 == 0)
            .map(|i| i as u32)
            .collect();
        assert!(valid(&masses, &ids, ids.len() as u32));
        assert!(!valid(&masses, &ids, u32::MAX));
        if !ids.is_empty() {
            assert!(!valid(&masses, &ids, (ids.len() - 1) as u32));
            let mut bad = ids.clone();
            bad[0] = count as u32;
            assert!(!valid(&masses, &bad, ids.len() as u32));
        }
        if ids.len() > 1 {
            let mut bad = ids.clone();
            bad[1] = bad[0];
            assert!(!valid(&masses, &bad, ids.len() as u32));
            bad = ids.clone();
            bad.reverse();
            assert!(!valid(&masses, &bad, ids.len() as u32));
            bad = ids.clone();
            bad[1] = 1;
            assert!(!valid(&masses, &bad, ids.len() as u32));
        }
    }
}
