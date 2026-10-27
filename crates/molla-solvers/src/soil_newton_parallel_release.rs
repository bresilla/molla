#[cfg(any(test, feature = "active-publication-newton"))]
#[path = "soil_newton_active_publication.rs"]
pub(crate) mod active_publication;
#[cfg(test)]
#[path = "soil_newton_candidate_cache.rs"]
pub(crate) mod candidate_cache;
#[cfg(test)]
#[path = "soil_newton_cgs2.rs"]
pub(crate) mod cgs2;
#[cfg(test)]
#[path = "soil_newton_inverse_skip.rs"]
pub(crate) mod inverse_skip;
#[cfg(test)]
#[path = "soil_newton_nonlinear_cache.rs"]
pub(crate) mod nonlinear_cache;
#[cfg(test)]
#[path = "soil_newton_norm_quaternary.rs"]
pub(crate) mod norm_quaternary;
#[cfg(any(test, feature = "norm-tree-newton"))]
#[path = "soil_newton_norm_tree.rs"]
pub(crate) mod norm_tree;
#[cfg(test)]
#[path = "soil_newton_operation_cost.rs"]
pub(crate) mod operation_cost;
#[cfg(test)]
#[path = "soil_newton_operation_counts.rs"]
pub(crate) mod operation_counts;
#[cfg(test)]
#[path = "soil_newton_phase_probe.rs"]
pub(crate) mod phase_probe;
#[cfg(test)]
#[path = "soil_newton_private_mgs.rs"]
pub(crate) mod private_mgs;
#[cfg(any(test, feature = "publication-products-newton"))]
#[path = "soil_newton_publication_products.rs"]
pub(crate) mod publication_products;
#[cfg(test)]
#[path = "soil_newton_row_products.rs"]
pub(crate) mod row_products;
#[cfg(test)]
#[path = "soil_newton_schedule.rs"]
pub(crate) mod schedule;
#[cfg(test)]
#[path = "soil_newton_schur_contact.rs"]
pub(crate) mod schur_contact;
#[cfg(any(test, feature = "schur-newton"))]
#[path = "soil_newton_schur_inverse.rs"]
pub(crate) mod schur_inverse;
#[cfg(test)]
#[path = "soil_newton_shared_inverse.rs"]
pub(crate) mod shared_inverse;
#[cfg(test)]
#[path = "soil_newton_sliced.rs"]
pub(crate) mod sliced;
#[cfg(test)]
#[path = "soil_newton_sparse_products.rs"]
pub(crate) mod sparse_products;
#[cfg(test)]
#[path = "soil_newton_zero_publication.rs"]
pub(crate) mod zero_publication;

pub(crate) fn source(mut source: String) -> String {
    assert_eq!(
        source
            .matches("struct ReleaseCandidates { values:array<vec3<f32>,20>, count:u32 }")
            .count(),
        1,
        "parallel candidate capacity contract"
    );
    let start = source.find("fn release_local(").unwrap();
    let end = source[start..].find("fn release_candidates(").unwrap() + start;
    let mut local = source[start..end].to_owned();
    replace(
        &mut local,
        "fn release_local(c:u32,mask:u32)->ReleaseChoice {",
        "fn release_local_candidate(c:u32,mask:u32,candidate_index:u32)->ReleaseChoice {",
    );
    replace(
        &mut local,
        "for(var i=0u;i<candidates.count;i++) {release_consider(c,mask,candidates.values[i],other,machine,&best);}",
        "if candidate_index<candidates.count {release_consider(c,mask,candidates.values[candidate_index],other,machine,&best);}",
    );
    replace(
        &mut source,
        OLD_CHOICE,
        "        conditioned_parallel_choice(c);",
    );
    source.push_str(&local);
    source.push_str(include_str!("../shaders/soil_newton_parallel_release.wgsl"));
    source
}

pub(crate) fn core_source(mut source: String) -> String {
    let start = source.find("fn direct_core() {").unwrap();
    let end = source[start..].find("fn direct_recover()").unwrap() + start;
    source.replace_range(
        start..end,
        include_str!("../shaders/soil_newton_parallel_core.wgsl"),
    );
    let prepare = source.find("fn conditioned_prepare(").unwrap();
    let start = source[prepare..]
        .find("    if lane<n {\n        var sums:array<vec2<f32>,36>;")
        .unwrap()
        + prepare;
    let end = source[start..]
        .find("    if lane<3u {condensed[4090u+lane]")
        .unwrap()
        + start;
    source.replace_range(
        start..end,
        include_str!("../shaders/soil_newton_parallel_conditioned_core.wgsl"),
    );
    source
}

#[cfg(test)]
pub(crate) fn tiled_source(mut source: String) -> String {
    replace(
        &mut source,
        include_str!("../shaders/soil_newton_parallel_core.wgsl"),
        "fn direct_core() {tiled_core(0u,false);}\n",
    );
    let body = include_str!("../shaders/soil_newton_parallel_conditioned_core.wgsl");
    assert!(matches!(source.matches(body).count(), 1 | 2));
    source = source.replace(body, "    tiled_core(chosen,true);\n");
    let mut tiled = include_str!("../shaders/soil_newton_tiled_core.wgsl").to_owned();
    if std::env::var_os("MOLLA_NEWTON_TILED_SHARED").is_some() {
        assert_eq!(
            source.matches("var<workgroup> h:array<f32,4160>;").count(),
            1
        );
        assert_eq!(
            source
                .matches("for (var i=lane;i<4160u;i+=64u) {h[i]=0.0;}")
                .count(),
            1
        );
        replace(
            &mut tiled,
            "condensed[2304u+item]=value;",
            "h[2u*item]=value.x;h[2u*item+1u]=value.y;",
        );
        replace(
            &mut tiled,
            "let base=2304u+(local*n+row_index)*3u;",
            "let base=(local*n+row_index)*6u;",
        );
        replace(
            &mut tiled,
            "condensed[base+a]",
            "vec2<f32>(h[base+2u*a],h[base+2u*a+1u])",
        );
        assert_eq!(tiled.matches("        storageBarrier();").count(), 2);
        tiled = tiled.replace("        storageBarrier();", "        workgroupBarrier();");
    }
    source.push_str(&tiled);
    source
}

#[test]
fn newton_tiled_core_source_contract() {
    use crate::soil_newton_gpu::{
        ExperimentalSoilNewtonGpu as Solver, NewtonVariant, shader_source_for,
    };
    let original = Solver::cached_contact_source(Solver::cached_delta_source(
        Solver::stalled_release_source(shader_source_for(NewtonVariant::Fischer)),
    ));
    let control = core_source(source(original));
    let candidate = tiled_source(control.clone());
    assert_eq!(
        candidate.matches("var<workgroup>").count(),
        control.matches("var<workgroup>").count()
    );
    assert!(2304 + 16 * 32 * 3 <= 4090);
    assert!(32 * (32 + 4) <= 5 * 256);
    let binary = crate::soil_newton_protected::compile(
        &format!(
            "{}{}256\n{candidate}",
            crate::soil_newton_protected::VOLATILE_MARKER,
            crate::soil_newton_width::MARKER
        ),
        "newton",
    )
    .unwrap();
    if let Some(root) = std::env::var_os("MOLLA_NEWTON_TILED_ARTIFACTS") {
        let root = std::path::PathBuf::from(root);
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("candidate.wgsl"), &candidate).unwrap();
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

#[cfg(test)]
pub(crate) fn probe_source(mut source: String, parallel: bool, mode: u32) -> String {
    if parallel {
        source = core_source(source);
    }
    let prepare = source.find("fn conditioned_prepare(").unwrap();
    let marker = if parallel {
        "    for(var cell=lane;cell<n*stride;cell+=256u)"
    } else {
        "    if lane<n {\n        var sums:array<vec2<f32>,36>;"
    };
    let start = source[prepare..].find(marker).unwrap() + prepare;
    let end = source[start..]
        .find("    if lane<3u {condensed[4090u+lane]")
        .unwrap()
        + start;
    let assembly = source[start..end].to_owned();
    source.push_str(&format!("\nfn isolated_conditioned_core(chosen:u32) {{let n=params.width;let stride=n+4u;{assembly}storageBarrier();}}\n"));
    source.push_str(&format!(r#"
@compute @workgroup_size(256) fn core_probe(@builtin(local_invocation_index) id:u32) {{
    lane=id;rate_lane=id;
    if lane==0u {{status.active_count=0u;atomicStore(&condensed_failed,0u);atomicStore(&report.fault,0u);
        for(var c=0u;c<params.contacts;c++) {{if contacts[c].soil_mass.w>0.0 {{active_indices[status.active_count]=c;status.active_count++;}}}}
    }}
    for(var i=lane;i<4160u;i+=256u) {{condensed[i]=vec2<f32>(0.0);}}
    for(var c=lane;c<params.contacts;c+=256u) {{let offset=4160u+54u*c;
        for(var j=0u;j<54u;j++) {{
            let high=f32(i32((c*7u+j)%31u)-15)/64.0;
            let low=f32(i32((c+j)%5u)-2)*0.0000000000582076609134674072265625;
            condensed[offset+j]=select(vec2<f32>(bitcast<f32>(0x7fc00000u),0.0),vec2<f32>(high,low),contacts[c].soil_mass.w>0.0);
        }}
        condensed[offset+53u]=vec2<f32>(select(0.0,1.0,contacts[c].soil_mass.w>0.0),0.0);
        if ({mode}u==3u && c==2u) || ({mode}u==4u && c==1u) {{condensed[offset+53u]=vec2<f32>(0.0);}}
    }}
    storageBarrier();
    if {mode}u==0u {{direct_core();}}
    else if {mode}u<3u {{isolated_conditioned_core(select(1u,params.contacts-1u,{mode}u==2u));}}
    else {{let accepted=conditioned_prepare(1u);if lane==0u {{atomicStore(&report.fault,select(8u,0u,accepted));}}}}
}}
"#));
    source
}

#[cfg(test)]
#[path = "soil_newton_parallel_core_tests.rs"]
mod core_tests;

#[test]
fn newton_parallel_release_core_source_contract() {
    use crate::soil_newton_gpu::{
        ExperimentalSoilNewtonGpu as Solver, NewtonVariant, shader_source_for,
    };
    let original = Solver::cached_contact_source(Solver::cached_delta_source(
        Solver::stalled_release_source(shader_source_for(NewtonVariant::Fischer)),
    ));
    let source = core_source(source(original.clone()));
    for body in [
        include_str!("../shaders/soil_newton_parallel_core.wgsl"),
        include_str!("../shaders/soil_newton_parallel_conditioned_core.wgsl"),
    ] {
        assert!(body.contains("var value=vec2<f32>(0.0);"));
        assert!(body.contains("value=rate_add(value,rate_mul(-r[b],condensed[offset+b*3u+a]));"));
        assert!(body.contains("projected[a]=value;"));
        assert!(!body.contains("projected[a]=rate_add("));
    }
    assert_eq!(
        source
            .matches("for(var cell=lane;cell<n*stride;cell+=256u)")
            .count(),
        2
    );
    assert_eq!(
        source.matches("var<workgroup>").count(),
        original.matches("var<workgroup>").count()
    );
    assert!(source.contains("condensed[cell]=sum;condensed[1152u+cell]=sum;"));
    let factor = |s: &str| {
        s.split_once("fn conditioned_factor()")
            .unwrap()
            .1
            .split_once("fn conditioned_prepare(")
            .unwrap()
            .0
            .to_owned()
    };
    assert_eq!(factor(&source), factor(&original));
    crate::soil_newton_protected::compile(
        &format!("{}256\n{source}", crate::soil_newton_width::MARKER),
        "newton",
    )
    .unwrap();
}

fn replace(source: &mut String, old: &str, new: &str) {
    assert_eq!(
        source.matches(old).count(),
        1,
        "parallel release source contract: {old}"
    );
    *source = source.replacen(old, new, 1);
}

const OLD_CHOICE: &str = "        if lane<8u {release_choices[lane]=release_local(c,lane);}
        workgroupBarrier();
        if lane==0u {for(var i=0u;i<8u;i++) {
            let candidate=release_choices[i];
            if candidate.valid!=0u && candidate.score<release_best.score {
                release_best=candidate;release_contact=c;conditioned_best=select(0u,1u,conditioned_model);
            }
        }}
        workgroupBarrier();";

#[test]
fn newton_parallel_release_source_contract() {
    use crate::soil_newton_gpu::{
        ExperimentalSoilNewtonGpu as Solver, NewtonVariant, shader_source_for,
    };
    let original = Solver::cached_contact_source(Solver::cached_delta_source(
        Solver::stalled_release_source(shader_source_for(NewtonVariant::Fischer)),
    ));
    let source = source(original.clone());
    assert_eq!(source.matches("conditioned_parallel_choice(c);").count(), 1);
    assert!(source.contains("lane/20u,lane%20u"));
    assert!(source.contains("b.x==a.x && b.y<a.y"));
    assert!(source.contains("candidate.score<threshold"));
    assert!(!source.contains(OLD_CHOICE));
    let candidates = |s: &str| {
        s.split_once("fn release_candidates(")
            .unwrap()
            .1
            .split_once("fn release_sliding_wall(")
            .unwrap()
            .0
            .to_owned()
    };
    assert_eq!(candidates(&source), candidates(&original));
    assert_eq!(
        source.matches("var<workgroup>").count(),
        original.matches("var<workgroup>").count()
    );
    crate::soil_newton_protected::compile(
        &format!("{}256\n{source}", crate::soil_newton_width::MARKER),
        "newton",
    )
    .unwrap();
}

#[test]
#[ignore = "requires GPU; parallel candidate ties, invalid entries and rank order"]
fn newton_parallel_release_gpu_ranking() {
    let engine = molla_compute::Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut source = String::from(
        r#"
struct Six {k:vec4<f32>,w:vec4<f32>}
struct ReleaseChoice {state:Six,score:f32,valid:u32}
var<private> lane:u32;
var<private> conditioned_model:bool=true;
var<workgroup> parts:array<vec2<f32>,256>;
var<workgroup> release_best:ReleaseChoice;
var<workgroup> release_contact:u32;
var<workgroup> conditioned_best:u32;
@group(0) @binding(0) var<storage,read_write> result_words:array<u32>;
fn release_local_candidate(c:u32,mask:u32,index:u32)->ReleaseChoice {
    let task=mask*20u+index;
    var score=-100.0;var valid=0u;
    if c==1u && task==79u {score=4.0;valid=1u;}
    if c==2u && task==0u {score=4.0;valid=1u;}
    if c==3u && task==159u {score=3.0;valid=1u;}
    if c==4u && (task==19u || task==20u) {score=2.0;valid=1u;}
    if c==5u && (task==3u || task==19u) {score=1.0;valid=1u;}
    return ReleaseChoice(Six(vec4<f32>(f32(task),0.0,0.0,0.0),vec4<f32>(0.0)),score,valid);
}
@compute @workgroup_size(256) fn probe(@builtin(local_invocation_index) id:u32) {
    lane=id;
    if lane==0u {release_best=ReleaseChoice(Six(),3.402823e38,0u);release_contact=0xffffffffu;conditioned_best=0u;}
    workgroupBarrier();
    for(var c=0u;c<7u;c++) {
        conditioned_parallel_choice(c);
        if lane==0u {
            result_words[4u*c]=u32(release_best.state.k.x);
            result_words[4u*c+1u]=release_contact;
            result_words[4u*c+2u]=conditioned_best;
            result_words[4u*c+3u]=release_best.valid;
        }
        workgroupBarrier();
    }
}
"#,
    );
    source.push_str(include_str!("../shaders/soil_newton_parallel_release.wgsl"));
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("parallel release ranking"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("parallel release ranking"),
        layout: None,
        module: &module,
        entry_point: Some("probe"),
        compilation_options: Default::default(),
        cache: None,
    });
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 112,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: output.as_entire_binding(),
        }],
    });
    for _ in 0..2 {
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        queue.submit([encoder.finish()]);
        let bytes = crate::soil_shared_wall_friction_gpu::capture::read(&engine, &output);
        let words: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        assert_eq!(
            words,
            [
                0,
                u32::MAX,
                0,
                0,
                79,
                1,
                1,
                1,
                79,
                1,
                1,
                1,
                159,
                3,
                1,
                1,
                19,
                4,
                1,
                1,
                3,
                5,
                1,
                1,
                3,
                5,
                1,
                1
            ]
        );
    }
}
