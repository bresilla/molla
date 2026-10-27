//! Experimental resident Newton iteration over paired contact impulses.

use crate::soil_newton_failure::NewtonFailureMethod;
use molla_compute::Engine;
use molla_core::{Error, Result};

#[cfg(all(test, feature = "soa-scratch-newton", not(target_arch = "wasm32")))]
#[path = "soil_newton_first_release.rs"]
pub(crate) mod first_release;

#[cfg(all(test, feature = "soa-scratch-newton", not(target_arch = "wasm32")))]
#[path = "soil_newton_parallel_linear_tests.rs"]
mod parallel_linear;

#[cfg(all(test, feature = "protected-newton", not(target_arch = "wasm32")))]
#[path = "soil_newton_publication_projection.rs"]
mod publication_projection;

#[cfg(all(test, feature = "protected-newton", not(target_arch = "wasm32")))]
#[path = "soil_newton_publication_energy.rs"]
mod publication_energy;

#[cfg(all(test, feature = "protected-newton", not(target_arch = "wasm32")))]
#[path = "soil_newton_precondition_unroll.rs"]
mod precondition_unroll;

#[cfg(all(test, feature = "protected-newton", not(target_arch = "wasm32")))]
#[path = "soil_newton_control_snapshot.rs"]
mod control_snapshot;

#[cfg(all(test, feature = "soa-scratch-newton", not(target_arch = "wasm32")))]
#[path = "soil_newton_bounded_reset.rs"]
mod bounded_reset;

#[cfg(all(test, feature = "soa-scratch-newton", not(target_arch = "wasm32")))]
#[path = "soil_newton_publication_input_cache.rs"]
mod publication_input_cache;

#[cfg(all(test, feature = "soa-scratch-newton", not(target_arch = "wasm32")))]
#[path = "soil_newton_scalar_product.rs"]
pub(crate) mod scalar_product;

#[cfg(all(test, feature = "soa-scratch-newton", not(target_arch = "wasm32")))]
#[path = "soil_newton_operator_rows.rs"]
mod operator_rows;

#[cfg(all(test, feature = "soa-scratch-newton", not(target_arch = "wasm32")))]
#[path = "soil_newton_residual_fusion.rs"]
mod residual_fusion;

#[cfg(all(test, feature = "soa-scratch-newton", not(target_arch = "wasm32")))]
#[path = "soil_newton_numerical_guard.rs"]
mod numerical_guard;

#[cfg(all(test, feature = "stagnant-branch-newton", not(target_arch = "wasm32")))]
#[path = "soil_newton_subgroup_reduce.rs"]
pub(crate) mod subgroup_reduce;

#[cfg(all(
    any(
        all(test, feature = "protected-newton"),
        feature = "parallel-indices-newton"
    ),
    not(target_arch = "wasm32")
))]
#[path = "soil_newton_active_indices.rs"]
pub(crate) mod active_indices;

#[cfg(all(
    any(
        all(test, feature = "protected-newton"),
        feature = "soa-scratch-newton"
    ),
    not(target_arch = "wasm32")
))]
#[path = "soil_newton_soa_scratch.rs"]
pub(crate) mod soa_scratch;

#[cfg(all(any(test, feature = "grouped-sum-newton"), not(target_arch = "wasm32")))]
#[path = "soil_newton_grouped_sum.rs"]
pub(crate) mod grouped_sum;

#[cfg(all(
    any(
        all(test, feature = "protected-newton"),
        all(
            feature = "retained-local-newton",
            feature = "publication-products-newton"
        )
    ),
    not(target_arch = "wasm32")
))]
#[path = "soil_newton_retained_conditioned.rs"]
mod retained_conditioned;

#[cfg(all(
    any(
        test,
        all(
            feature = "conditioned-projection-cache-newton",
            not(feature = "shared-conditioned-core-newton")
        )
    ),
    not(target_arch = "wasm32")
))]
#[path = "soil_newton_conditioned_projection_cache.rs"]
pub(crate) mod conditioned_projection_cache;

#[cfg(all(
    any(test, feature = "shared-conditioned-core-newton"),
    not(target_arch = "wasm32")
))]
#[path = "soil_newton_shared_conditioned_core.rs"]
pub(crate) mod shared_conditioned_core;

#[cfg(all(
    any(test, feature = "stream-block-newton"),
    not(target_arch = "wasm32")
))]
#[path = "soil_newton_stream_block.rs"]
pub(crate) mod stream_block;

#[cfg(all(test, feature = "retained-local-newton", not(target_arch = "wasm32")))]
#[path = "soil_newton_publication_validation.rs"]
mod publication_validation;

#[cfg(all(test, feature = "retained-local-newton", not(target_arch = "wasm32")))]
#[path = "soil_newton_retained_direct.rs"]
pub(crate) mod retained_direct;

#[cfg(all(test, feature = "retained-local-newton", not(target_arch = "wasm32")))]
#[path = "soil_newton_direct_quality.rs"]
pub(crate) mod direct_quality;

#[cfg(all(test, feature = "protected-newton", not(target_arch = "wasm32")))]
#[path = "soil_newton_distributed.rs"]
pub(crate) mod distributed;

#[cfg(all(
    test,
    feature = "active-publication-newton",
    not(target_arch = "wasm32")
))]
#[path = "soil_newton_shadow.rs"]
pub(crate) mod shadow;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum NewtonVariant {
    Paired,
    Cooperative,
    Sorted,
    #[cfg(test)]
    Chunked,
    Stagnation,
    Fischer,
    #[cfg(test)]
    SortedSerial,
    #[cfg(test)]
    NativeF64,
}

pub struct SoilNewtonInputs<'a> {
    pub mass: &'a wgpu::Buffer,
    pub free_rates: &'a wgpu::Buffer,
    pub point_rows: &'a wgpu::Buffer,
    pub point_response: &'a wgpu::Buffer,
    pub contacts: &'a wgpu::Buffer,
    pub upstream_status: &'a wgpu::Buffer,
}

#[derive(Clone, Copy)]
pub struct SoilNewtonCapacity {
    pub dofs: u32,
    pub contacts: u32,
    pub newton_iterations: u32,
    pub krylov_iterations: u32,
    pub max_owned_bytes: u64,
}

struct NewtonProgram {
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
}

fn program_key(source: &str, entry: &str, condensed: bool) -> String {
    format!(
        "entry={entry};condensed={condensed};wgsl={}",
        blake3::hash(source.as_bytes())
    )
}

#[test]
fn newton_program_key_covers_compiler_inputs() {
    let key = program_key("shader", "newton", false);
    assert_eq!(key, program_key("shader", "newton", false));
    assert_ne!(key, program_key("shader2", "newton", false));
    assert_ne!(key, program_key("shader", "probe", false));
    assert_ne!(key, program_key("shader", "newton", true));
    assert_ne!(
        key,
        program_key("// NEWTON_PROTECTED_V1\nshader", "newton", false)
    );
    assert_ne!(
        program_key("// NEWTON_PROTECTED_V1\nshader", "newton", false),
        program_key("// NEWTON_PROTECTED_V2\nshader", "newton", false)
    );
}

#[test]
fn newton_protected_cpu_falls_back() {
    assert!(!protected_enabled(&Engine::new_cpu()));
}

/// Produces paired iterates and independently certified machine/soil outputs.
/// Mass SPD and response preparation must already be validated by the caller.
pub struct ExperimentalSoilNewtonGpu {
    pipeline: wgpu::ComputePipeline,
    group: wgpu::BindGroup,
    state: wgpu::Buffer,
    status: wgpu::Buffer,
    outputs: wgpu::Buffer,
    rates: wgpu::Buffer,
    publication_status: wgpu::Buffer,
    bytes: u64,
    failure_method: NewtonFailureMethod,
    #[cfg(test)]
    scratch: Vec<wgpu::Buffer>,
    #[cfg(test)]
    scratch_soa: bool,
    #[cfg(test)]
    arithmetic: NewtonVariant,
    #[cfg(test)]
    slice_dispatches: Option<u32>,
    #[cfg(test)]
    prefix_pipeline: Option<wgpu::ComputePipeline>,
    #[cfg(test)]
    clock_trace: bool,
}

fn invalid() -> Error {
    Error::Solver("soil Newton: invalid capacity, usage or device limit".into())
}

/// Whether conditioned Newton selects the protected native compiler on this engine.
pub fn protected_enabled(engine: &Engine) -> bool {
    cfg!(all(
        feature = "protected-newton",
        not(target_arch = "wasm32")
    )) && engine.adapter_info().is_some_and(|a| a.backend == "Vulkan")
        && engine
            .device()
            .is_some_and(|d| d.features().contains(wgpu::Features::PASSTHROUGH_SHADERS))
}

fn parse_diagnostic_limit(value: &str) -> Result<u32> {
    value
        .parse::<u32>()
        .ok()
        .filter(|v| (1..=36864).contains(v))
        .ok_or_else(|| Error::Solver("MOLLA_NEWTON_DIAGNOSTIC_INNER_LIMIT must be 1..36864".into()))
}

/// Resident conditioned Newton workgroup width for the selected compiler route.
pub fn execution_lanes(protected: bool) -> u32 {
    if protected && cfg!(feature = "wide-newton") {
        256
    } else {
        64
    }
}

pub(crate) fn execution_limits(engine: &Engine) -> (u32, u32) {
    let lanes = execution_lanes(protected_enabled(engine));
    (lanes, 24000 + 24 * (lanes - 64))
}

#[test]
fn newton_diagnostic_limit_parser() {
    for value in ["1", "512", "36864"] {
        assert_eq!(
            parse_diagnostic_limit(value).unwrap(),
            value.parse::<u32>().unwrap()
        );
    }
    for value in ["", "0", "-1", "36865", "4294967296", " 512", "NaN"] {
        assert!(parse_diagnostic_limit(value).is_err());
    }
}

/// Returns the opt-in diagnostic inner-work ceiling; normal execution has none.
pub fn diagnostic_inner_limit() -> Result<Option<u32>> {
    match std::env::var("MOLLA_NEWTON_DIAGNOSTIC_INNER_LIMIT") {
        Ok(value) => parse_diagnostic_limit(&value).map(Some),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(_) => Err(invalid()),
    }
}

pub(crate) fn diagnostic_source(mut source: String, limit: u32) -> String {
    assert!((1..=36864).contains(&limit));
    replace(
        &mut source,
        "report.diagnostic_limit=0u;",
        &format!("report.diagnostic_limit={limit}u;"),
    );
    replace(
        &mut source,
        "if step==params.newton_iterations",
        "if diagnostic_exhausted() {return;}\n        if step==params.newton_iterations",
    );
    replace(
        &mut source,
        "prepare_linear();least_squares(lane,sqrt(damping));",
        "if diagnostic_exhausted() {return;}\n                prepare_linear();least_squares(lane,sqrt(damping));",
    );
    source.push_str("\nfn diagnostic_exhausted()->bool {\n    if lane==0u {outer.stop=select(0u,1u,report.linear_iterations>=report.diagnostic_limit);if outer.stop!=0u {atomicOr(&report.fault,32u);}}\n    return workgroupUniformLoad(&outer.stop)!=0u;\n}\n");
    source
}

fn replace(source: &mut String, from: &str, to: &str) {
    assert_eq!(
        source.matches(from).count(),
        1,
        "Newton source anchor: {from}"
    );
    *source = source.replace(from, to);
}

pub(crate) fn shader_source_for(arithmetic: NewtonVariant) -> String {
    let mut krylov = include_str!("../shaders/soil_newton_krylov.wgsl").to_owned();
    #[cfg(not(test))]
    let chunked = matches!(
        arithmetic,
        NewtonVariant::Stagnation | NewtonVariant::Fischer
    );
    #[cfg(test)]
    let chunked = matches!(
        arithmetic,
        NewtonVariant::Chunked | NewtonVariant::Stagnation | NewtonVariant::Fischer
    );
    let cached = matches!(
        arithmetic,
        NewtonVariant::Cooperative | NewtonVariant::Sorted
    ) || chunked;
    if cached {
        let start = krylov.find("fn apply_transpose() {").unwrap();
        let end = krylov.find("fn scratch_norm()").unwrap();
        krylov.replace_range(
            start..end,
            include_str!("../shaders/soil_newton_transpose_cached.wgsl"),
        );
        replace(
            &mut krylov,
            "let true_error=recover_direction(current_count);",
            "let estimate=abs(workgroupUniformLoad(&g[j+1u]))/beta;\n        let stopped=workgroupUniformLoad(&control.stop);var true_error=3.402823e38;\n        if estimate<=params.tolerance || j+1u==params.iterations || stopped!=0u {true_error=recover_direction(current_count);}",
        );
    }
    if arithmetic == NewtonVariant::Cooperative || chunked {
        replace(
            &mut krylov,
            "parts[lane]=sum; workgroupBarrier();\n    if lane==0u {\n        var total=vec2<f32>(0.0);\n        for (var i=0u;i<64u;i++) {total=rate_add(total,parts[i]);}\n        control.scalar=value(total);\n    }",
            "parts[lane]=sum;let total=reduce_parts();\n    if lane==0u {control.scalar=value(total);}",
        );
        replace(
            &mut krylov,
            "parts[lane]=sum;workgroupBarrier();\n    if lane==0u {\n        var total=vec2<f32>(0.0);for (var i=0u;i<64u;i++) {total=rate_add(total,parts[i]);}\n        control.scalar=scale*sqrt(max(value(total),0.0));\n    }",
            "parts[lane]=sum;let total=reduce_parts();\n    if lane==0u {control.scalar=scale*sqrt(max(value(total),0.0));}",
        );
        krylov.push_str(include_str!("../shaders/soil_newton_reduce.wgsl"));
    }
    if chunked {
        let start = krylov.find("fn apply_operator() {").unwrap() + "fn apply_operator() {".len();
        let end = start + krylov[start..].find("    for (var slot=lane;").unwrap();
        krylov.replace_range(start..end, "\n    project_rates(0u);\n");
        let transpose = krylov.find("fn apply_transpose() {").unwrap();
        let start = transpose
            + krylov[transpose..]
                .find("    if lane<params.width {")
                .unwrap();
        let end = start + krylov[start..].find("    for (var slot=lane;").unwrap();
        krylov.replace_range(start..end, "    project_rates(2u);storageBarrier();\n");
        krylov.push_str(include_str!("../shaders/soil_newton_project_chunks.wgsl"));
    }
    if matches!(
        arithmetic,
        NewtonVariant::Stagnation | NewtonVariant::Fischer
    ) {
        replace(
            &mut krylov,
            "    for (var j=0u;j<params.iterations;j++) {",
            "    var checkpoint_estimate=1.0;\n    for (var j=0u;j<params.iterations;j++) {",
        );
        replace(
            &mut krylov,
            "if estimate<=params.tolerance || j+1u==params.iterations || stopped!=0u {true_error=recover_direction(current_count);}",
            "var stagnant=false;\n        if (j+1u)%8u==0u {stagnant=j>=15u && estimate>=0.99*checkpoint_estimate;checkpoint_estimate=estimate;}\n        if estimate<=params.tolerance || j+1u==params.iterations || stopped!=0u || stagnant {true_error=recover_direction(current_count);}\n        if stagnant {break;}",
        );
    }
    replace(
        &mut krylov,
        "var<storage,read> blocks:",
        "var<storage,read_write> blocks:",
    );
    replace(
        &mut krylov,
        "struct Params { width:u32, contacts:u32, iterations:u32, tolerance:f32 }",
        "struct Params { width:u32, contacts:u32, iterations:u32, tolerance:f32, newton_iterations:u32, pad0:u32, pad1:u32, pad2:u32 }",
    );
    replace(
        &mut krylov,
        "@compute @workgroup_size(64)\nfn solve(@builtin(local_invocation_index) id:u32)",
        "fn solve(id:u32)",
    );
    replace(
        &mut krylov,
        "@compute @workgroup_size(64)\nfn least_squares(@builtin(local_invocation_index) id:u32)",
        "fn least_squares(id:u32,damping:f32)",
    );
    replace(
        &mut krylov,
        "let scale=max(abs(rhobar),beta);var rho=0.0;",
        "let damp_scale=max(abs(rhobar),damping);\n        if damp_scale>0.0 {let damp_rho=damp_scale*sqrt((rhobar/damp_scale)*(rhobar/damp_scale)+(damping/damp_scale)*(damping/damp_scale));phibar*=rhobar/damp_rho;rhobar=damp_rho;}\n        let scale=max(abs(rhobar),beta);var rho=0.0;",
    );
    replace(
        &mut krylov,
        "var phibar=initial;var rhobar=alpha;",
        "let normal_scale=initial*alpha;\n    var phibar=initial;var rhobar=alpha;",
    );
    replace(
        &mut krylov,
        "storageBarrier();let error=true_direction_error();",
        "storageBarrier();let estimate=alpha*abs(sine*phi)/normal_scale;var error=3.402823e38;\n        if !finite(estimate) {if lane==0u {atomicOr(&status.fault,2u);}}\n        if estimate<=params.tolerance || iteration+1u==params.iterations || alpha==0.0 {error=regularized_direction_error(damping,normal_scale);}",
    );
    replace(
        &mut krylov,
        "status.iterations=iteration+1u;status.pad=abs(phibar)/initial;",
        "status.iterations=iteration+1u;status.pad=estimate;",
    );
    let operator = include_str!("../shaders/soil_newton_operator.wgsl");
    let flags = &operator[operator.find("fn flags(").unwrap()..operator.find("@compute").unwrap()];
    let local = &operator[operator.find("struct Local").unwrap()
        ..operator
            .find("@compute @workgroup_size(64)\nfn vector_rates")
            .unwrap()];
    let mut local = local.to_owned();
    if arithmetic == NewtonVariant::Fischer {
        replace(
            &mut local,
            "struct Local { normal:vec3<f32>,",
            "struct Local { paired_normal:array<vec2<f32>,3>, paired_k:array<vec2<f32>,3>, paired_relative:array<vec2<f32>,3>, paired_w:array<vec2<f32>,3>, paired_soil:array<vec2<f32>,3>, paired_rho:vec2<f32>, paired_separation:vec2<f32>, separation:f32, normal:vec3<f32>,",
        );
        replace(
            &mut local,
            "var trace=vec2<f32>(0.0);",
            "let normal_length=fischer_length(node.normal_gap.xyz);var trace=vec2<f32>(0.0);var separation_pair=vec2<f32>(0.0);",
        );
        replace(
            &mut local,
            "a.k[axis]=value(vec2<f32>(x.high.k[axis],x.low.k[axis]));",
            "a.paired_normal[axis]=fischer_div(vec2<f32>(node.normal_gap[axis],0.0),normal_length);a.normal[axis]=value(a.paired_normal[axis]);a.paired_k[axis]=vec2<f32>(x.high.k[axis],x.low.k[axis]);a.k[axis]=value(a.paired_k[axis]);",
        );
        replace(
            &mut local,
            "let soil_velocity=rate_mul(1.0/a.mass,soil);",
            "let soil_velocity=fischer_div(soil,vec2<f32>(a.mass,0.0));",
        );
        replace(
            &mut local,
            "a.w[axis]=value(vec2<f32>(x.high.w[axis],x.low.w[axis]));",
            "a.paired_w[axis]=vec2<f32>(x.high.w[axis],x.low.w[axis]);a.w[axis]=value(a.paired_w[axis]);",
        );
        replace(
            &mut local,
            "a.soil[axis]=value(soil_velocity); a.relative[axis]=value(rate_add(soil_velocity,-machine));",
            "a.paired_soil[axis]=soil_velocity;a.soil[axis]=value(soil_velocity);let relative_pair=rate_add(soil_velocity,-machine);a.paired_relative[axis]=relative_pair;a.relative[axis]=value(relative_pair);separation_pair=rate_add(separation_pair,rate_mul(node.normal_gap[axis],relative_pair));",
        );
        replace(
            &mut local,
            "a.lambda=dot(a.normal,a.k);",
            "a.paired_separation=rate_add(fischer_div(separation_pair,normal_length),vec2<f32>(a.gap,0.0));a.separation=value(a.paired_separation);a.lambda=value(fischer_dot(a.paired_normal,a.paired_k));",
        );
        replace(
            &mut local,
            "a.rho=1.0/scale;",
            "a.paired_rho=fischer_div(vec2<f32>(1.0,0.0),rate_add(fischer_div(vec2<f32>(1.0,0.0),vec2<f32>(a.mass,0.0)),trace));a.rho=1.0/scale;",
        );
        replace(
            &mut local,
            "block.residual=residual(a);",
            "block.residual=fischer_residual(a);",
        );
        let start = local.find("    for (var j=0u;j<6u;j++) {").unwrap();
        let end = local[start..]
            .find("    for (var i=0u;i<6u;i++) { if !finite(block.residual[i])")
            .unwrap()
            + start;
        local.replace_range(start..end, "    block=fischer_block(a,block.residual);\n");
        replace(
            &mut local,
            "!finite(block.direct[i])",
            "!all(abs(block.direct[i])<=vec2<f32>(3.402823e38))",
        );
        replace(
            &mut local,
            "!finite(block.machine[i])",
            "!all(abs(block.machine[i])<=vec2<f32>(3.402823e38))",
        );
        replace(
            &mut local,
            "let normal=select(a.lambda,separation,a.lambda-separation>0.0);",
            "let normal=fischer(a.lambda,separation).x;",
        );
        replace(
            &mut local,
            "for (var i=0u;i<3u;i++) { if a.wall[i]*(a.w[i]-a.mass*a.soil[i])>0.0 { w[i]=a.mass*a.soil[i]; } }",
            "for (var i=0u;i<3u;i++) {if a.wall[i]!=0.0 {w[i]=a.wall[i]*fischer(a.wall[i]*a.w[i],a.wall[i]*a.mass*a.soil[i]).x;}}",
        );
        replace(
            &mut local,
            "var dn=dl; if a.lambda-a.rho*(dot(a.normal,a.relative)+a.gap)>0.0 { dn=a.rho*dot(a.normal,dr); }",
            "let fb=fischer(a.lambda,a.rho*(dot(a.normal,a.relative)+a.gap));let dn=fb.y*dl+fb.z*a.rho*dot(a.normal,dr);",
        );
        replace(
            &mut local,
            "for (var i=0u;i<3u;i++) { if a.wall[i]*(a.w[i]-a.mass*a.soil[i])>0.0 { w[i]=a.mass*ds[i]; } }",
            "for (var i=0u;i<3u;i++) {if a.wall[i]!=0.0 {let fb=fischer(a.wall[i]*a.w[i],a.wall[i]*a.mass*a.soil[i]);w[i]=fb.y*dw[i]+fb.z*a.mass*ds[i];}}",
        );
        local = local.replace(
            "a.rho*(dot(a.normal,a.relative)+a.gap)",
            "a.rho*a.separation",
        );
        local.push_str(include_str!("../shaders/soil_newton_fischer.wgsl"));
        local.push_str(include_str!("../shaders/soil_newton_fischer_linear.wgsl"));
    }
    replace(
        &mut local,
        "@compute @workgroup_size(64)\nfn linearize(@builtin(global_invocation_id) id:vec3<u32>) {\n    let c=id.x; rate_lane=c%64u;",
        "fn nonlinear_contact(c:u32) {\n    rate_lane=lane;",
    );
    local = local
        .replace("work.q", "rates")
        .replace("work.fault", "status.fault");
    replace(
        &mut local,
        "a.rho=1.0/scale; var block:Block;",
        "a.rho=1.0/scale; vector[c]=Six(vec4<f32>(physical_error(a),0.0,0.0,0.0),vec4<f32>(0.0)); var block:Block;",
    );
    let mut outer = include_str!("../shaders/soil_newton.wgsl").to_owned();
    if arithmetic == NewtonVariant::Stagnation {
        replace(
            &mut outer,
            "var damping=merit/(1.0+state_norm);",
            "restore();if lane==0u {atomicStore(&status.fault,0u);}storageBarrier();evaluate();\n            if release_separating_wall() {merit=evaluate();if lane==0u {report.steps=step+1u;}continue;}\n            var damping=merit/(1.0+state_norm);",
        );
        outer.push_str(include_str!("../shaders/soil_newton_wall_release.wgsl"));
    }
    if chunked {
        let start = outer.find("fn evaluate()->f32 {").unwrap() + "fn evaluate()->f32 {".len();
        let end = outer[start..].find("    var worst=0.0;").unwrap() + start;
        outer.replace_range(start..end, "\n    project_rates(1u);\n");
    }
    if arithmetic == NewtonVariant::Fischer {
        let start = outer.find("    var merit=evaluate();").unwrap();
        outer.truncate(start);
        outer.push_str(include_str!("../shaders/soil_newton_fischer_schedule.wgsl"));
        let start = krylov.find("fn recover_direction(count:u32)").unwrap();
        let end = krylov.find("fn true_direction_error()").unwrap();
        let mut recover = krylov[start..end].to_owned();
        replace(
            &mut recover,
            "var sum=vec2<f32>(0.0);",
            "var sum=base_direction_pair(c,i);",
        );
        replace(
            &mut recover,
            "var result:array<f32,6>;",
            "var result:array<f32,6>;var low:array<f32,6>;",
        );
        replace(
            &mut recover,
            "result[i]=value(sum);",
            "result[i]=sum.x;low[i]=sum.y;",
        );
        replace(
            &mut recover,
            "direction[c]=output;vector[c]=output;",
            "direction[c]=output;vector[c]=output;basis[params.iterations*max(params.contacts,1u)+c].z=packed(low);",
        );
        krylov.replace_range(start..end, &recover);
        replace(
            &mut krylov,
            "for (var slot=lane;slot<status.active_count;slot+=64u) {let c=active_indices[slot];vector[c]=direction[c];}\n    storageBarrier();apply_operator();",
            "storageBarrier();apply_direction();",
        );
        replace(
            &mut krylov,
            "let w=basis[stride+c].v;var output:array<f32,6>;var next:array<f32,6>;",
            "let w=basis[stride+c].v;var output:array<f32,6>;var low:array<f32,6>;var next:array<f32,6>;",
        );
        replace(
            &mut krylov,
            "output[i]=value(rate_add(vec2<f32>(component(direction[c],i),0.0),rate_mul(phi/rho,vec2<f32>(component(w,i),0.0))));",
            "let updated=rate_add(direction_pair(c,i),rate_mul(phi/rho,vec2<f32>(component(w,i),0.0)));output[i]=updated.x;low[i]=updated.y;",
        );
        replace(
            &mut krylov,
            "direction[c]=packed(output);basis[stride+c].v=packed(next);basis[c].z=v;",
            "direction[c]=packed(output);basis[params.iterations*max(params.contacts,1u)+c].z=packed(low);basis[stride+c].v=packed(next);basis[c].z=v;",
        );
        replace(
            &mut krylov,
            "else {point=vec2<f32>(vector[c].k[axis],0.0);}",
            "else if kind==3u {point=direction_pair(c,axis);}else {point=vec2<f32>(vector[c].k[axis],0.0);}",
        );
        replace(
            &mut outer,
            "direction[c]=Six();vector[c]=Six();scratch[c]=Six();inverses[c]=Inverse();",
            "direction[c]=Six();direction[max(params.contacts,1u)+c]=Six();direction[2u*max(params.contacts,1u)+c]=Six();basis[params.iterations*max(params.contacts,1u)+c].z=Six();vector[c]=Six();scratch[c]=Six();inverses[c]=Inverse();",
        );
        replace(&mut outer, "vec2<f32>(d.k[i],0.0)", "direction_pair(c,i)");
        replace(
            &mut outer,
            "vec2<f32>(d.w[i],0.0)",
            "direction_pair(c,3u+i)",
        );
        replace(
            &mut outer,
            "for (var slot=lane;slot<status.active_count;slot+=64u) {let c=active_indices[slot];vector[c]=direction[c];}\n    storageBarrier();apply_operator();",
            "storageBarrier();apply_direction();",
        );
        replace(
            &mut outer,
            "vec2<f32>(component(direction[c],i),0.0)",
            "direction_pair(c,i)",
        );
        krylov.push_str(include_str!("../shaders/soil_newton_paired_direction.wgsl"));
        replace(
            &mut krylov,
            "direct:array<f32,36>, machine:array<f32,18>",
            "direct:array<vec2<f32>,36>, machine:array<vec2<f32>,18>",
        );
        replace(
            &mut krylov,
            "vec2<f32>(blocks[c].direct[i*6u+j],0.0)",
            "blocks[c].direct[i*6u+j]",
        );
        krylov = krylov
            .replace("rate_mul(blocks[c].", "fischer_product(blocks[c].")
            .replace("rate_mul(-blocks[c].", "fischer_product(-blocks[c].");
        replace(
            &mut krylov,
            "!finite(blocks[c].direct[i])",
            "!all(abs(blocks[c].direct[i])<=vec2<f32>(3.402823e38))",
        );
        replace(
            &mut krylov,
            "!finite(blocks[c].machine[i])",
            "!all(abs(blocks[c].machine[i])<=vec2<f32>(3.402823e38))",
        );
        let start = krylov.find("fn solve(id:u32)").unwrap();
        let end = krylov.find("fn least_squares(id:u32,damping:f32)").unwrap();
        let mut cycle = krylov[start..end].to_owned();
        replace(
            &mut cycle,
            "fn solve(id:u32)",
            "fn krylov_cycle(id:u32,cycle_limit:u32)",
        );
        replace(
            &mut cycle,
            "rhs[i]=-blocks[c].residual[i];",
            "rhs[i]=component(vector[c],i);",
        );
        cycle = cycle.replace("params.iterations", "cycle_limit");
        replace(
            &mut cycle,
            "let estimate=abs(workgroupUniformLoad(&g[j+1u]))/beta;",
            "let estimate=abs(workgroupUniformLoad(&g[j+1u]))/refinement_norm;",
        );
        krylov.replace_range(start..end, &cycle);
        replace(
            &mut krylov,
            "status.relative=error/control.beta;",
            "status.relative=error/refinement_norm;",
        );
        krylov.push_str(include_str!("../shaders/soil_newton_refinement.wgsl"));
    }
    format!(
        "{}\n{}\n{krylov}\n{flags}\n{local}\n{}\n{}",
        match arithmetic {
            NewtonVariant::Paired => crate::soil_rate_encoding::arithmetic(64),
            NewtonVariant::Cooperative => crate::soil_rate_encoding::sorted_arithmetic(64),
            #[cfg(test)]
            NewtonVariant::Chunked => crate::soil_rate_encoding::sorted_arithmetic(64),
            NewtonVariant::Stagnation | NewtonVariant::Fischer =>
                crate::soil_rate_encoding::sorted_arithmetic(64),
            #[cfg(test)]
            NewtonVariant::SortedSerial => crate::soil_rate_encoding::sorted_arithmetic(64),
            NewtonVariant::Sorted => crate::soil_rate_encoding::sorted_arithmetic(64),
            #[cfg(test)]
            NewtonVariant::NativeF64 => crate::soil_rate_encoding::native_arithmetic(64),
        },
        include_str!("../shaders/soil_normal_rate_contract.wgsl"),
        outer,
        crate::soil_newton_publish_gpu::resident_source()
    )
}

impl ExperimentalSoilNewtonGpu {
    /// Includes a paired dense core and fixed-size per-contact damping cache.
    pub fn required_condensed_bytes(c: SoilNewtonCapacity) -> Result<u64> {
        Ok(Self::required_fischer_bytes(c)? + 33280 + 432 * u64::from(c.contacts.max(1)))
    }

    /// Includes paired Jacobian coefficients and refinement bases for Fischer.
    pub fn required_fischer_bytes(c: SoilNewtonCapacity) -> Result<u64> {
        Ok(Self::required_bytes(c)? + 280 * u64::from(c.contacts.max(1)))
    }

    pub fn required_bytes(c: SoilNewtonCapacity) -> Result<u64> {
        if !(1..=32).contains(&c.dofs)
            || !(1..=64).contains(&c.newton_iterations)
            || !(1..=64).contains(&c.krylov_iterations)
        {
            return Err(invalid());
        }
        Ok(
            (756 + 64 * u64::from(c.krylov_iterations)) * u64::from(c.contacts.max(1))
                + 8 * u64::from(c.dofs)
                + 128,
        )
    }

    pub fn new(
        engine: &Engine,
        input: SoilNewtonInputs<'_>,
        capacity: SoilNewtonCapacity,
    ) -> Result<Self> {
        Self::new_with_arithmetic(engine, input, capacity, NewtonVariant::Paired)
    }

    /// Uses magnitude-sorted error-free sums with paired storage.
    pub fn new_sorted_pairs(
        engine: &Engine,
        input: SoilNewtonInputs<'_>,
        capacity: SoilNewtonCapacity,
    ) -> Result<Self> {
        Self::new_with_arithmetic(engine, input, capacity, NewtonVariant::Sorted)
    }

    #[cfg(test)]
    pub(crate) fn new_sorted_serial(
        engine: &Engine,
        input: SoilNewtonInputs<'_>,
        capacity: SoilNewtonCapacity,
    ) -> Result<Self> {
        Self::new_with_arithmetic(engine, input, capacity, NewtonVariant::SortedSerial)
    }

    /// Uses sorted paired sums with cooperative Krylov reductions.
    pub fn new_cooperative_pairs(
        engine: &Engine,
        input: SoilNewtonInputs<'_>,
        capacity: SoilNewtonCapacity,
    ) -> Result<Self> {
        Self::new_with_arithmetic(engine, input, capacity, NewtonVariant::Cooperative)
    }

    #[cfg(test)]
    pub(crate) fn new_chunked_pairs(
        engine: &Engine,
        input: SoilNewtonInputs<'_>,
        capacity: SoilNewtonCapacity,
    ) -> Result<Self> {
        Self::new_with_arithmetic(engine, input, capacity, NewtonVariant::Chunked)
    }

    /// Uses chunked paired products and true-residual-checked GMRES stagnation fallback.
    pub fn new_stagnation_pairs(
        engine: &Engine,
        input: SoilNewtonInputs<'_>,
        capacity: SoilNewtonCapacity,
    ) -> Result<Self> {
        Self::new_with_arithmetic(engine, input, capacity, NewtonVariant::Stagnation)
    }

    /// Uses paired Fischer-Burmeister coefficients with `required_fischer_bytes` storage.
    pub fn new_fischer_pairs(
        engine: &Engine,
        input: SoilNewtonInputs<'_>,
        capacity: SoilNewtonCapacity,
    ) -> Result<Self> {
        Self::new_with_arithmetic(engine, input, capacity, NewtonVariant::Fischer)
    }

    /// Uses bounded smoothing continuation with independent physical publication.
    pub fn new_fischer_continuation(
        engine: &Engine,
        input: SoilNewtonInputs<'_>,
        capacity: SoilNewtonCapacity,
    ) -> Result<Self> {
        let mut solver = Self::new_with_program(
            engine,
            input,
            capacity,
            NewtonVariant::Fischer,
            |mut source| {
                replace(
                    &mut source,
                    "var<private> fischer_smoothing:f32=0.0;",
                    "var<private> fischer_smoothing:f32=0.01;",
                );
                source
            },
            "newton",
        )?;
        solver.failure_method = NewtonFailureMethod::FischerContinuation;
        Ok(solver)
    }

    /// Uses continuation and at most eight local sliding-wall releases.
    pub fn new_fischer_continuation_release(
        engine: &Engine,
        input: SoilNewtonInputs<'_>,
        capacity: SoilNewtonCapacity,
    ) -> Result<Self> {
        let mut solver = Self::new_with_program(
            engine,
            input,
            capacity,
            NewtonVariant::Fischer,
            Self::local_release_source,
            "newton",
        )?;
        solver.failure_method = NewtonFailureMethod::FischerContinuationRelease;
        Ok(solver)
    }

    pub(crate) fn local_release_source(mut source: String) -> String {
        replace(
            &mut source,
            "var<private> fischer_smoothing:f32=0.0;",
            "var<private> fischer_smoothing:f32=0.01;",
        );
        replace(
            &mut source,
            "prepare_linear();least_squares(lane,sqrt(damping));",
            "if attempt==0u && release_sliding_wall() {\n\
                step++;if lane==0u {report.steps=step;report.trials++;}\n\
                phase=0u;continue;\n\
            }\n\
            prepare_linear();least_squares(lane,sqrt(damping));",
        );
        source.push_str(include_str!("../shaders/soil_newton_local_release.wgsl"));
        source
    }

    /// Uses a verified condensed damped direction with bounded LSQR fallback.
    pub fn new_fischer_condensed(
        engine: &Engine,
        input: SoilNewtonInputs<'_>,
        capacity: SoilNewtonCapacity,
    ) -> Result<Self> {
        let mut solver = Self::new_with_storage(
            engine,
            input,
            capacity,
            NewtonVariant::Fischer,
            Self::condensed_source,
            "newton",
            true,
        )?;
        solver.failure_method = NewtonFailureMethod::FischerCondensed;
        Ok(solver)
    }

    pub(crate) fn condensed_source(source: String) -> String {
        let mut source = Self::local_release_source(source);
        replace(
            &mut source,
            "prepare_linear();least_squares(lane,sqrt(damping));",
            "prepare_condensed();if !condensed_direction(damping) {prepare_condensed();least_squares(lane,sqrt(damping));}",
        );
        source.push_str(include_str!("../shaders/soil_newton_condensed.wgsl"));
        source
    }

    /// Uses verified direct Newton and damped directions with bounded iterative fallbacks.
    pub fn new_fischer_direct(
        engine: &Engine,
        input: SoilNewtonInputs<'_>,
        capacity: SoilNewtonCapacity,
    ) -> Result<Self> {
        Self::new_direct_variant(engine, input, capacity, true)
    }

    #[cfg(test)]
    pub(crate) fn new_fischer_direct_full(
        engine: &Engine,
        input: SoilNewtonInputs<'_>,
        capacity: SoilNewtonCapacity,
    ) -> Result<Self> {
        Self::new_direct_variant(engine, input, capacity, false)
    }

    fn new_direct_variant(
        engine: &Engine,
        input: SoilNewtonInputs<'_>,
        capacity: SoilNewtonCapacity,
        lazy: bool,
    ) -> Result<Self> {
        let mut solver = Self::new_with_storage(
            engine,
            input,
            capacity,
            NewtonVariant::Fischer,
            |source| {
                let mut source = Self::direct_source(source);
                if !lazy {
                    replace(
                        &mut source,
                        "const DIRECT_LAZY_JACOBIAN:bool=true;",
                        "const DIRECT_LAZY_JACOBIAN:bool=false;",
                    );
                }
                source
            },
            "newton",
            true,
        )?;
        solver.failure_method = NewtonFailureMethod::FischerDirect;
        Ok(solver)
    }

    pub(crate) fn direct_source(source: String) -> String {
        let mut source = Self::condensed_source(source);
        replace(
            &mut source,
            "prepare_linear();solve(lane);",
            "prepare_condensed();if !direct_direction() {prepare_linear();solve(lane);}",
        );
        replace(
            &mut source,
            "block=fischer_block(a,block.residual);",
            "if direct_linearize {block=fischer_block(a,block.residual);}",
        );
        replace(
            &mut source,
            "let candidate=evaluate();",
            "direct_linearize=!DIRECT_LAZY_JACOBIAN || (phase!=1u && phase!=3u);let candidate=evaluate();",
        );
        let accepted = "if reduce_fischer_smoothing(candidate) {continue;}";
        assert_eq!(source.matches(accepted).count(), 2);
        source = source.replace(accepted, "reduce_fischer_smoothing(candidate);continue;");
        source.push_str(
            "\nconst DIRECT_LAZY_JACOBIAN:bool=true;\nvar<private> direct_linearize:bool=true;\n",
        );
        source.push_str(include_str!("../shaders/soil_newton_direct.wgsl"));
        source
    }

    /// Reduces continuation smoothing after two accepted residual-stagnating steps.
    pub fn new_fischer_adaptive_direct(
        engine: &Engine,
        input: SoilNewtonInputs<'_>,
        capacity: SoilNewtonCapacity,
    ) -> Result<Self> {
        let mut solver = Self::new_with_storage(
            engine,
            input,
            capacity,
            NewtonVariant::Fischer,
            Self::adaptive_direct_source,
            "newton",
            true,
        )?;
        solver.failure_method = NewtonFailureMethod::FischerAdaptiveDirect;
        Ok(solver)
    }

    pub(crate) fn adaptive_direct_source(source: String) -> String {
        Self::adaptive_smoothing_source(Self::direct_source(source))
    }

    fn adaptive_smoothing_source(mut source: String) -> String {
        let call = "reduce_fischer_smoothing(candidate);";
        assert_eq!(source.matches(call).count(), 2);
        source = source.replace(call, "reduce_fischer_smoothing(candidate,merit);");
        replace(
            &mut source,
            "fn reduce_fischer_smoothing(residual:f32)->bool {",
            "fn reduce_fischer_smoothing(residual:f32,previous:f32)->bool {\n smoothing_stalled=select(0u,smoothing_stalled+1u,residual>=0.99*previous);",
        );
        replace(
            &mut source,
            "if fischer_smoothing>0.0 && residual<=fischer_smoothing {",
            "if fischer_smoothing>0.0 && (residual<=fischer_smoothing || smoothing_stalled>=2u) {\n smoothing_stalled=0u;",
        );
        source.push_str("\nvar<private> smoothing_stalled:u32=0u;\n");
        source
    }

    /// Combines a bounded iterative prefix with stagnation-controlled smoothing.
    pub fn new_fischer_adaptive_hybrid(
        engine: &Engine,
        input: SoilNewtonInputs<'_>,
        capacity: SoilNewtonCapacity,
    ) -> Result<Self> {
        let mut solver = Self::new_with_storage(
            engine,
            input,
            capacity,
            NewtonVariant::Fischer,
            Self::adaptive_hybrid_source,
            "newton",
            true,
        )?;
        solver.failure_method = NewtonFailureMethod::FischerAdaptiveHybrid;
        Ok(solver)
    }

    pub(crate) fn adaptive_hybrid_source(source: String) -> String {
        Self::adaptive_smoothing_source(Self::hybrid_source(source))
    }

    pub(crate) fn conditioned_program(
        source: String,
        protected: bool,
        lanes: u32,
        _capacity: SoilNewtonCapacity,
    ) -> String {
        let source = Self::stalled_release_source(source);
        #[cfg(test)]
        let source = if std::env::var_os("MOLLA_NEWTON_RELEASE_REVISIT").is_some() {
            Self::revisited_release_source(source)
        } else {
            source
        };
        #[cfg(test)]
        let source = if std::env::var_os("MOLLA_NEWTON_CONVERGENCE_CONTROL").is_some() {
            Self::convergence_control_source(source)
        } else {
            source
        };
        #[cfg(test)]
        let source = if std::env::var_os("MOLLA_NEWTON_NONMONOTONE_STALL").is_some() {
            assert!(std::env::var_os("MOLLA_NEWTON_RELEASE_REVISIT").is_none());
            Self::nonmonotone_stall_source(source)
        } else {
            source
        };
        #[cfg(test)]
        let source = if std::env::var_os("MOLLA_NEWTON_BRANCH_PIVOT").is_some() {
            assert!(std::env::var_os("MOLLA_NEWTON_RELEASE_REVISIT").is_none());
            assert!(std::env::var_os("MOLLA_NEWTON_NONMONOTONE_STALL").is_none());
            Self::branch_pivot_source(source)
        } else {
            source
        };
        #[cfg(test)]
        let source = if std::env::var_os("MOLLA_NEWTON_FALLBACK_BRANCH_PIVOT").is_some() {
            for flag in [
                "MOLLA_NEWTON_BRANCH_PIVOT",
                "MOLLA_NEWTON_NONMONOTONE_STALL",
                "MOLLA_NEWTON_RELEASE_REVISIT",
                "MOLLA_NEWTON_CONDITIONED_REFINE",
            ] {
                assert!(std::env::var_os(flag).is_none());
            }
            Self::fallback_branch_pivot_source(source)
        } else {
            source
        };
        #[cfg(test)]
        let source = if std::env::var_os("MOLLA_NEWTON_POST_RELEASE_SETTLE").is_some()
            || std::env::var_os("MOLLA_NEWTON_POST_RELEASE_PRIORITY").is_some()
            || std::env::var_os("MOLLA_NEWTON_STAGNANT_BRANCH").is_some()
        {
            assert!(!cfg!(feature = "stagnant-branch-newton"));
            for flag in [
                "MOLLA_NEWTON_BRANCH_PIVOT",
                "MOLLA_NEWTON_FALLBACK_BRANCH_PIVOT",
                "MOLLA_NEWTON_NONMONOTONE_STALL",
                "MOLLA_NEWTON_RELEASE_REVISIT",
                "MOLLA_NEWTON_CONDITIONED_REFINE",
            ] {
                assert!(std::env::var_os(flag).is_none());
            }
            let source = Self::post_release_settle_source(source);
            if std::env::var_os("MOLLA_NEWTON_STAGNANT_BRANCH").is_some() {
                assert!(std::env::var_os("MOLLA_NEWTON_POST_RELEASE_SETTLE").is_none());
                assert!(std::env::var_os("MOLLA_NEWTON_POST_RELEASE_PRIORITY").is_none());
                Self::stagnant_branch_source(source)
            } else if std::env::var_os("MOLLA_NEWTON_POST_RELEASE_PRIORITY").is_some() {
                assert!(std::env::var_os("MOLLA_NEWTON_POST_RELEASE_SETTLE").is_none());
                Self::post_release_priority_source(source)
            } else {
                source
            }
        } else {
            source
        };
        #[cfg(test)]
        let source = if std::env::var_os("MOLLA_NEWTON_CONDITIONED_REFINE").is_some() {
            for flag in [
                "MOLLA_NEWTON_BRANCH_PIVOT",
                "MOLLA_NEWTON_NONMONOTONE_STALL",
                "MOLLA_NEWTON_RELEASE_REVISIT",
            ] {
                assert!(std::env::var_os(flag).is_none());
            }
            Self::conditioned_refine_source(source)
        } else {
            source
        };
        #[cfg(all(feature = "protected-newton", not(target_arch = "wasm32")))]
        if protected {
            #[cfg(feature = "stagnant-branch-newton")]
            let source = Self::stagnant_branch_source(Self::post_release_settle_source(source));
            #[cfg(feature = "cached-newton")]
            let source = Self::cached_contact_source(Self::cached_delta_source(source));
            #[cfg(feature = "parallel-newton")]
            let source = crate::soil_newton_parallel_release::core_source(
                crate::soil_newton_parallel_release::source(source),
            );
            #[cfg(feature = "schur-newton")]
            let source = crate::soil_newton_parallel_release::schur_inverse::source(source);
            #[cfg(feature = "norm-tree-newton")]
            let source = crate::soil_newton_parallel_release::norm_tree::source(source);
            #[cfg(feature = "active-publication-newton")]
            let source = crate::soil_newton_parallel_release::active_publication::source(source);
            #[cfg(feature = "publication-products-newton")]
            let source = crate::soil_newton_parallel_release::publication_products::source(source);
            #[cfg(feature = "retained-local-newton")]
            let source = retained_conditioned::source(source);
            #[cfg(all(
                feature = "conditioned-projection-cache-newton",
                not(feature = "shared-conditioned-core-newton")
            ))]
            let source = conditioned_projection_cache::source(source);
            #[cfg(feature = "shared-conditioned-core-newton")]
            let source = shared_conditioned_core::source(source);
            #[cfg(feature = "stream-block-newton")]
            let source = stream_block::source(source);
            #[cfg(all(
                test,
                feature = "publication-products-newton",
                not(feature = "retained-local-newton")
            ))]
            let source = if std::env::var_os("MOLLA_NEWTON_RETAINED_LOCAL").is_some() {
                for flag in [
                    "MOLLA_NEWTON_BRANCH_PIVOT",
                    "MOLLA_NEWTON_NONMONOTONE_STALL",
                    "MOLLA_NEWTON_RELEASE_REVISIT",
                    "MOLLA_NEWTON_CONDITIONED_REFINE",
                ] {
                    assert!(std::env::var_os(flag).is_none());
                }
                retained_conditioned::source(source)
            } else {
                source
            };
            #[cfg(test)]
            let source = if let Ok(selected) = std::env::var("MOLLA_NEWTON_PIVOT_INSPECT_CONTACT") {
                let selected: u32 = selected.parse().unwrap();
                assert!(selected < _capacity.contacts);
                Self::pivot_inspect_source(source, selected)
            } else {
                source
            };
            let width = if lanes == 64 {
                String::new()
            } else {
                format!("{}{lanes}\n", crate::soil_newton_width::MARKER)
            };
            let rounding = if cfg!(feature = "volatile-newton") {
                crate::soil_newton_protected::VOLATILE_MARKER
            } else {
                ""
            };
            return format!(
                "{}{rounding}{width}{source}",
                crate::soil_newton_protected::PREFIX
            );
        }
        source
    }

    pub(crate) fn conditioned_runtime_program(
        source: String,
        protected: bool,
        lanes: u32,
        capacity: SoilNewtonCapacity,
    ) -> String {
        let source = Self::conditioned_program(source, protected, lanes, capacity);
        #[cfg(all(
            feature = "grouped-sum-newton",
            not(feature = "grouped-eight-newton"),
            not(target_arch = "wasm32")
        ))]
        let source = grouped_sum::source(source, false);
        #[cfg(all(feature = "grouped-eight-newton", not(target_arch = "wasm32")))]
        let source = grouped_sum::source_eight(source, false);
        #[cfg(all(feature = "parallel-indices-newton", not(target_arch = "wasm32")))]
        let source = active_indices::source(source, false);
        #[cfg(all(feature = "soa-scratch-newton", not(target_arch = "wasm32")))]
        let source = soa_scratch::transform(source);
        source
    }

    /// Coordinates floor releases and recovers from sustained residual stagnation.
    /// With `protected-newton`, Vulkan passthrough devices use validated protected arithmetic.
    /// `grouped-sum-newton` rejects unsupported devices; other profiles retain WGSL fallback.
    /// Compiler failures are returned without changing the route.
    pub fn new_fischer_conditioned(
        engine: &Engine,
        input: SoilNewtonInputs<'_>,
        capacity: SoilNewtonCapacity,
    ) -> Result<Self> {
        let protected = protected_enabled(engine);
        let (lanes, storage) = execution_limits(engine);
        if cfg!(feature = "grouped-sum-newton") && (!protected || lanes != 256) {
            return Err(invalid());
        }
        let limits = engine.device().ok_or_else(invalid)?.limits();
        if limits.max_compute_invocations_per_workgroup < lanes
            || limits.max_compute_workgroup_size_x < lanes
            || limits.max_compute_workgroup_storage_size < storage
        {
            return Err(invalid());
        }
        let mut solver = Self::new_with_storage(
            engine,
            input,
            capacity,
            NewtonVariant::Fischer,
            |source| Self::conditioned_runtime_program(source, protected, lanes, capacity),
            "newton",
            true,
        )?;
        solver.failure_method = if protected && cfg!(feature = "soa-scratch-newton") {
            NewtonFailureMethod::FischerConditionedProtectedSoaScratch256
        } else if protected && cfg!(feature = "parallel-indices-newton") {
            NewtonFailureMethod::FischerConditionedProtectedParallelIndices256
        } else if protected && cfg!(feature = "grouped-eight-newton") {
            NewtonFailureMethod::FischerConditionedProtectedGroupedEight256
        } else if protected && cfg!(feature = "grouped-sum-newton") {
            NewtonFailureMethod::FischerConditionedProtectedGroupedSum256
        } else if protected && cfg!(feature = "stagnant-branch-newton") {
            NewtonFailureMethod::FischerConditionedProtectedStagnantBranch256
        } else if protected && cfg!(feature = "stream-block-newton") {
            NewtonFailureMethod::FischerConditionedProtectedStreamBlock256
        } else if protected && cfg!(feature = "shared-conditioned-core-newton") {
            NewtonFailureMethod::FischerConditionedProtectedSharedCore256
        } else if protected && cfg!(feature = "conditioned-projection-cache-newton") {
            NewtonFailureMethod::FischerConditionedProtectedProjectionCache256
        } else if protected && cfg!(feature = "retained-local-newton") {
            NewtonFailureMethod::FischerConditionedProtectedRetainedLocal256
        } else if protected && cfg!(feature = "publication-products-newton") {
            NewtonFailureMethod::FischerConditionedProtectedPublicationProducts256
        } else if protected && cfg!(feature = "active-publication-newton") {
            NewtonFailureMethod::FischerConditionedProtectedActivePublication256
        } else if protected && cfg!(feature = "norm-tree-newton") {
            NewtonFailureMethod::FischerConditionedProtectedNorm256
        } else if protected && cfg!(feature = "schur-newton") {
            NewtonFailureMethod::FischerConditionedProtectedSchur256
        } else if protected && cfg!(feature = "volatile-newton") {
            NewtonFailureMethod::FischerConditionedProtectedVolatile256
        } else if protected && cfg!(feature = "parallel-newton") {
            NewtonFailureMethod::FischerConditionedProtectedParallel256
        } else if protected && cfg!(feature = "cached-newton") {
            NewtonFailureMethod::FischerConditionedProtectedCached256
        } else if protected && lanes == 256 {
            NewtonFailureMethod::FischerConditionedProtectedWide256
        } else if protected {
            NewtonFailureMethod::FischerConditionedProtectedStalled
        } else {
            NewtonFailureMethod::FischerConditionedStalled
        };
        Ok(solver)
    }

    #[cfg(all(
        test,
        feature = "active-publication-newton",
        not(target_arch = "wasm32")
    ))]
    pub(crate) fn new_publication_shadow_control(
        engine: &Engine,
        input: SoilNewtonInputs<'_>,
        capacity: SoilNewtonCapacity,
    ) -> Result<Self> {
        assert!(protected_enabled(engine));
        let mut solver = Self::new_with_storage(
            engine,
            input,
            capacity,
            NewtonVariant::Fischer,
            |s| {
                use crate::soil_newton_parallel_release as p;
                let source = p::norm_tree::source(p::schur_inverse::source(p::core_source(
                    p::source(Self::cached_contact_source(Self::cached_delta_source(
                        Self::stalled_release_source(s),
                    ))),
                )));
                #[cfg(feature = "publication-products-newton")]
                let source = p::active_publication::source(source);
                format!(
                    "{}{}{}256\n{source}",
                    crate::soil_newton_protected::PREFIX,
                    crate::soil_newton_protected::VOLATILE_MARKER,
                    crate::soil_newton_width::MARKER
                )
            },
            "newton",
            true,
        )?;
        solver.failure_method = if cfg!(feature = "publication-products-newton") {
            NewtonFailureMethod::FischerConditionedProtectedActivePublication256
        } else {
            NewtonFailureMethod::FischerConditionedProtectedNorm256
        };
        Ok(solver)
    }

    pub(crate) fn conditioned_source(source: String) -> String {
        let mut source = Self::adaptive_hybrid_source(source);
        replace(
            &mut source,
            "attempt==0u && release_sliding_wall()",
            "attempt==0u && conditioned_release()",
        );
        let start = source
            .find("fn release_merit(chosen:u32,next:Six)->f32 {")
            .unwrap();
        let end = start
            + source[start..]
                .find("    var square=vec2<f32>(0.0);")
                .unwrap();
        let original = source[start..end].to_owned();
        let body = original.split_once('{').unwrap().1;
        let body = body.replacen("var q:array<vec2<f32>,32>;", "", 1);
        let prefix = format!("fn release_merit(chosen:u32,next:Six)->f32 {{\nvar q:array<vec2<f32>,32>;\nif conditioned_model {{
            for(var i=0u;i<params.width;i++) {{q[i]=free_rates[i];}}
            for(var slot=0u;slot<status.active_count;slot++) {{let c=active_indices[slot];let x=conditioned_state(c,chosen,next);
                for(var i=0u;i<params.width;i++) {{for(var a=0u;a<3u;a++) {{q[i]=rate_add(q[i],rate_mul(-response[c*params.width+i][a],vec2<f32>(x.high.k[a],x.low.k[a])));}}}}
            }}
        }} else {{{body}}}\n");
        source.replace_range(start..end, &prefix);
        replace(
            &mut source,
            "if c==chosen {x=State(next,Six());}",
            "if conditioned_model {x=conditioned_state(c,chosen,next);} else if c==chosen {x=State(next,Six());}",
        );
        let start = source
            .find("    var machine=mat3x3<f32>();var other=vec3<f32>(0.0);")
            .unwrap();
        let end = start
            + source[start..]
                .find("    var a=machine;var soil=node.soil_mass.xyz;")
                .unwrap();
        let body = source[start..end].split_once('\n').unwrap().1.to_owned();
        let prefix = format!("    var machine=mat3x3<f32>();var other=vec3<f32>(0.0);\nif conditioned_model {{
            let width=params.width;let stride=width+4u;
            for(var axis=0u;axis<3u;axis++) {{var velocity=vec2<f32>(0.0);
                for(var i=0u;i<width;i++) {{var without=rate_add(rates[i],condensed[i*stride+width]);
                    for(var j=0u;j<3u;j++) {{without=rate_add(without,fischer_product(condensed[i*stride+width+1u+j],condensed[4090u+j]));}}
                    velocity=rate_add(velocity,rate_mul(rows[c*width+i][axis],without));
                }}
                other[axis]=value(velocity);
                for(var j=0u;j<3u;j++) {{var sum=vec2<f32>(0.0);
                    for(var i=0u;i<width;i++) {{sum=rate_add(sum,rate_mul(rows[c*width+i][axis],condensed[i*stride+width+1u+j]));}}
                    machine[j][axis]=value(sum);
                }}
            }}
        }} else {{{body}}}\n");
        source.replace_range(start..end, &prefix);
        source.push_str(include_str!("../shaders/soil_newton_conditioned.wgsl"));
        source
    }

    pub(crate) fn stalled_release_source(source: String) -> String {
        let mut source = Self::conditioned_source(source);
        replace(
            &mut source,
            "candidate<merit*(1.0-1e-4*alpha)",
            "candidate<merit*(1.0-select(1e-4*alpha,0.01,fischer_smoothing==0.0 && smoothing_stalled>=2u))",
        );
        source
    }

    #[cfg(test)]
    pub(crate) fn revisited_release_source(mut source: String) -> String {
        let start = source.find("fn conditioned_release()->bool {").unwrap();
        let (prefix, body) = source.split_at(start);
        assert_eq!(body.matches("if visited {continue;}").count(), 1);
        let body = body.replace(
            "if visited {continue;}",
            "if visited && (fischer_smoothing>0.0 || smoothing_stalled<2u) {continue;}",
        );
        source = format!("{prefix}{body}");
        source
    }

    #[cfg(test)]
    pub(crate) fn convergence_control_source(mut source: String) -> String {
        replace(
            &mut source,
            "if step==params.newton_iterations {if lane==0u {atomicOr(&report.fault,8u);}return;}",
            "if step==params.newton_iterations {if lane==0u {\n\
                condensed[0]=vec2<f32>(fischer_smoothing,f32(smoothing_stalled));\n\
                condensed[1]=vec2<f32>(f32(outer.releases),f32(step));\n\
                for(var i=0u;i<8u;i++) {condensed[2u+i]=vec2<f32>(f32(release_visited[i]),select(0.0,1.0,i<outer.releases));}\n\
                atomicOr(&report.fault,8u);}return;}",
        );
        source
    }

    #[cfg(test)]
    pub(crate) fn conditioned_refine_source(mut source: String) -> String {
        let start = source.find("fn direct_contact(c:u32)->bool {").unwrap();
        let end = start + source[start..].find("\nfn direct_core()").unwrap();
        let mut body = source[start..end].to_owned();
        replace(
            &mut body,
            "    var matrix:array<array<vec2<f32>,10>,6>;",
            "    var matrix:array<array<vec2<f32>,10>,6>;\n    var permutation:array<u32,6>;for(var i=0u;i<6u;i++) {permutation[i]=i;}",
        );
        replace(
            &mut body,
            "if pivot!=k {let row=matrix[k];matrix[k]=matrix[pivot];matrix[pivot]=row;}",
            "if pivot!=k {let row=matrix[k];matrix[k]=matrix[pivot];matrix[pivot]=row;let index=permutation[k];permutation[k]=permutation[pivot];permutation[pivot]=index;}",
        );
        replace(
            &mut body,
            "for(var j=k+1u;j<10u;j++) {matrix[i][j]=rate_add(matrix[i][j],-fischer_product(ratio,matrix[k][j]));}",
            "for(var j=k+1u;j<10u;j++) {matrix[i][j]=rate_add(matrix[i][j],-fischer_product(ratio,matrix[k][j]));}\n            matrix[i][k]=ratio;",
        );
        replace(
            &mut body,
            "    return true;",
            r#"    if conditioned_refinement {
        for(var column=0u;column<4u;column++) {
            var residual:array<vec2<f32>,6>;var error=0.0;var scale=0.0;
            for(var i=0u;i<6u;i++) {
                var rhs=vec2<f32>(-blocks[c].residual[i],0.0);
                if column<3u {rhs=blocks[c].machine[i*3u+column];}
                var r=rhs;
                for(var j=0u;j<6u;j++) {
                    var x=condensed[offset+36u+j];
                    if column<3u {x=condensed[offset+j*3u+column];}
                    r=rate_add(r,-fischer_product(blocks[c].direct[i*6u+j],x));
                }
                residual[i]=r;error+=value(r)*value(r);scale+=value(rhs)*value(rhs);
            }
            if sqrt(error)/max(sqrt(scale),1e-30)<=params.tolerance {continue;}
            for(var i=0u;i<6u;i++) {
                var r=residual[permutation[i]];
                for(var j=0u;j<i;j++) {r=rate_add(r,-fischer_product(matrix[i][j],matrix[j][6u+column]));}
                matrix[i][6u+column]=r;
            }
            for(var row=6u;row>0u;row--) {
                let i=row-1u;var r=matrix[i][6u+column];
                for(var j=i+1u;j<6u;j++) {r=rate_add(r,-fischer_product(matrix[i][j],matrix[j][6u+column]));}
                let delta=fischer_div(r,matrix[i][i]);matrix[i][6u+column]=delta;
                var index=offset+36u+i;if column<3u {index=offset+i*3u+column;}
                let x=rate_add(condensed[index],delta);
                if !all(abs(x)<=vec2<f32>(3.402823e38)) {return false;}
                condensed[index]=x;
            }
        }
    }
    return true;"#,
        );
        source.replace_range(start..end, &body);
        replace(
            &mut source,
            "var valid=direct_contact(c);",
            "conditioned_refinement=true;var valid=direct_contact(c);conditioned_refinement=false;",
        );
        source.push_str("\nvar<private> conditioned_refinement:bool=false;\n");
        source
    }

    #[cfg(test)]
    pub(crate) fn pivot_inspect_source(mut source: String, selected: u32) -> String {
        let start = source.find("fn direct_contact(c:u32)->bool {").unwrap();
        let end = start + source[start..].find("\nfn direct_core()").unwrap();
        let mut direct = source[start..end].to_owned();
        replace(
            &mut direct,
            "fn direct_contact(c:u32)->bool {",
            "fn direct_contact(c:u32)->bool {\n    for(var j=42u;j<53u;j++) {condensed[4160u+54u*c+j]=vec2<f32>(-1.0,0.0);}",
        );
        replace(
            &mut direct,
            "if !finite(largest) || largest<1e-20 {return false;}",
            "if !finite(largest) || largest<1e-20 {condensed[4160u+54u*c+42u]=vec2<f32>(1.0,f32(k));condensed[4160u+54u*c+43u]=vec2<f32>(largest,f32(pivot));return false;}",
        );
        replace(
            &mut direct,
            "if !all(abs(x)<=vec2<f32>(3.402823e38)) {return false;}",
            "if !all(abs(x)<=vec2<f32>(3.402823e38)) {condensed[4160u+54u*c+42u]=vec2<f32>(2.0,f32(i));condensed[4160u+54u*c+43u]=vec2<f32>(f32(column),0.0);return false;}",
        );
        replace(
            &mut direct,
            "    return true;",
            "    condensed[4160u+54u*c+42u]=vec2<f32>(0.0,6.0);\n    return true;",
        );
        source.replace_range(start..end, &direct);
        replace(
            &mut source,
            "valid=valid && finite(relative) && relative<=params.tolerance;",
            "condensed[offset+44u+column]=vec2<f32>(relative,params.tolerance);\n                valid=valid && finite(relative) && relative<=params.tolerance;",
        );
        replace(
            &mut source,
            "var<private> fischer_smoothing:f32=0.01;",
            "var<private> fischer_smoothing:f32=0.0;",
        );
        replace(
            &mut source,
            "if certify_candidate(lane) {return;}",
            "if params.contacts>0u {inspect_conditioned_pivot();return;}\nif certify_candidate(lane) {return;}",
        );
        let marker = "    var a=machine;var soil=node.soil_mass.xyz;";
        assert_eq!(source.matches(marker).count(), 2);
        source = source.replace(marker, "    if lane==0u && mask==0u {for(var i=0u;i<3u;i++) {condensed[4029u+i]=vec2<f32>(other[i],0.0);for(var j=0u;j<3u;j++) {condensed[4020u+3u*i+j]=vec2<f32>(machine[j][i],0.0);}}}\n    var a=machine;var soil=node.soil_mass.xyz;");
        source.push_str(&format!(r#"
fn inspect_conditioned_pivot() {{
    let c={selected}u;
    conditioned_cache();conditioned_model=conditioned_prepare(c);
    if lane<8u {{release_choices[lane]=release_local(c,lane);}}
    workgroupBarrier();
    if lane==0u {{
        var invalid=0u;var first=0xffffffffu;
        for(var slot=0u;slot<status.active_count;slot++) {{let other=active_indices[slot];
            if other!=c && condensed[4160u+54u*other+53u].x!=1.0 {{invalid++;first=min(first,other);}}
        }}
        release_best=ReleaseChoice(Six(),3.402823e38,0u);var mask=0xffffffffu;
        for(var i=0u;i<8u;i++) {{let choice=release_choices[i];
            if choice.valid!=0u && choice.score<release_best.score {{release_best=choice;mask=i;}}
        }}
        condensed[4000u]=vec2<f32>(select(0.0,1.0,conditioned_model),f32(invalid));
        condensed[4001u]=vec2<f32>(f32(release_best.valid),release_best.score);
        condensed[4002u]=vec2<f32>(f32(mask),f32(first));
        let chosen=release_best.state;
        for(var i=0u;i<3u;i++) {{condensed[4003u+i]=vec2<f32>(chosen.k[i],chosen.w[i]);}}
        condensed[4006u]=vec2<f32>(f32(atomicLoad(&condensed_failed)),f32(c));
        atomicOr(&report.fault,8u);
    }}
    storageBarrier();
}}
"#));
        source
    }

    #[cfg(any(test, feature = "stagnant-branch-newton"))]
    pub(crate) fn stagnant_branch_source(mut source: String) -> String {
        replace(
            &mut source,
            "        let c=workgroupUniformLoad(&release_queue[rank]);if c==0xffffffffu {continue;}\n        conditioned_model=conditioned_prepare(c);",
            "        if branch_settle && rank>0u {break;}\n\
            let c=workgroupUniformLoad(&release_queue[rank]);if c==0xffffffffu {continue;}\n        conditioned_model=conditioned_prepare(c);",
        );
        replace(
            &mut source,
            "for(var slot=lane;slot<status.active_count;slot+=64u) {\n                let c=active_indices[slot];saved[c]=states[c];",
            "            if fischer_smoothing==0.0 && smoothing_stalled==1u && workgroupUniformLoad(&outer.releases)>0u {\n\
                if conditioned_release() {step++;if lane==0u {report.steps=step;report.trials++;}settle_pending=true;phase=0u;continue;}\n\
            }\n\
            for(var slot=lane;slot<status.active_count;slot+=64u) {\n                let c=active_indices[slot];saved[c]=states[c];",
        );
        source
    }

    #[cfg(test)]
    pub(crate) fn post_release_priority_source(mut source: String) -> String {
        replace(
            &mut source,
            "        let c=workgroupUniformLoad(&release_queue[rank]);if c==0xffffffffu {continue;}\n        conditioned_model=conditioned_prepare(c);",
            "        if branch_settle && workgroupUniformLoad(&release_contact)!=0xffffffffu {break;}\n\
            let c=workgroupUniformLoad(&release_queue[rank]);if c==0xffffffffu {continue;}\n        conditioned_model=conditioned_prepare(c);",
        );
        source
    }

    #[cfg(any(test, feature = "stagnant-branch-newton"))]
    pub(crate) fn post_release_settle_source(mut source: String) -> String {
        replace(
            &mut source,
            "var<private> conditioned_model:bool=false;",
            "var<private> conditioned_model:bool=false;\nvar<private> branch_settle:bool=false;\nvar<private> branch_merit:f32=0.0;",
        );
        replace(
            &mut source,
            "var phase=0u;var step=0u;var merit=0.0;var state_norm=0.0;",
            "var phase=0u;var step=0u;var merit=0.0;var state_norm=0.0;var settle_pending=false;",
        );
        replace(
            &mut source,
            "if attempt==0u && conditioned_release() {",
            "if attempt==0u && conditioned_release() {\nsettle_pending=true;",
        );
        replace(
            &mut source,
            "            for(var slot=lane;slot<status.active_count;slot+=64u) {\n                let c=active_indices[slot];saved[c]=states[c];",
            "            if settle_pending && fischer_smoothing==0.0 {\n\
                settle_pending=false;branch_settle=true;branch_merit=merit;\n\
                let settled=conditioned_release();branch_settle=false;\n\
                if settled {step++;if lane==0u {report.steps=step;report.trials++;}settle_pending=true;phase=0u;continue;}\n\
            }\n\
            for(var slot=lane;slot<status.active_count;slot+=64u) {\n                let c=active_indices[slot];saved[c]=states[c];",
        );
        replace(
            &mut source,
            "if !releases || !all(abs(w)<=vec3<f32>(3.402823e38)) {return;}",
            "if (!releases && !branch_settle) || !all(abs(w)<=vec3<f32>(3.402823e38)) {return;}",
        );
        replace(
            &mut source,
            "if score<(*best).score {(*best)=ReleaseChoice(next,score,1u);}",
            "if score<(*best).score && (!branch_settle || score<branch_merit) {(*best)=ReleaseChoice(next,score,1u);}",
        );
        source
    }

    #[cfg(test)]
    pub(crate) fn fallback_branch_pivot_source(mut source: String) -> String {
        replace(
            &mut source,
            "if !releases || !all(abs(w)<=vec3<f32>(3.402823e38)) {return;}",
            "if (!releases && fischer_smoothing!=0.0) || !all(abs(w)<=vec3<f32>(3.402823e38)) {return;}",
        );
        source
    }

    #[cfg(test)]
    pub(crate) fn branch_pivot_source(mut source: String) -> String {
        replace(
            &mut source,
            "if !releases || !all(abs(w)<=vec3<f32>(3.402823e38)) {return;}",
            "if (!releases && !(fischer_smoothing==0.0 && smoothing_stalled>=2u)) || !all(abs(w)<=vec3<f32>(3.402823e38)) {return;}",
        );
        source
    }

    #[cfg(test)]
    pub(crate) fn nonmonotone_stall_source(mut source: String) -> String {
        replace(
            &mut source,
            "candidate<merit*(1.0-select(1e-4*alpha,0.01,fischer_smoothing==0.0 && smoothing_stalled>=2u))",
            "candidate<merit*select(1.0-1e-4*alpha,1.0-0.1*alpha+0.1/f32(step+1u),fischer_smoothing==0.0 && smoothing_stalled>=2u)",
        );
        source
    }

    #[cfg(all(test, feature = "protected-newton", not(target_arch = "wasm32")))]
    pub(crate) fn new_stalled_release_probe(
        engine: &Engine,
        input: SoilNewtonInputs<'_>,
        capacity: SoilNewtonCapacity,
    ) -> Result<Self> {
        if !protected_enabled(engine) {
            return Err(invalid());
        }
        Self::new_with_storage(
            engine,
            input,
            capacity,
            NewtonVariant::Fischer,
            |source| {
                format!(
                    "{}{}",
                    crate::soil_newton_protected::PREFIX,
                    Self::stalled_release_source(source)
                )
            },
            "newton",
            true,
        )
    }

    pub(crate) fn failure_method(&self) -> NewtonFailureMethod {
        self.failure_method
    }

    #[cfg(all(
        any(test, feature = "cached-newton"),
        feature = "protected-newton",
        not(target_arch = "wasm32")
    ))]
    pub(crate) fn cached_delta_source(mut source: String) -> String {
        replace(
            &mut source,
            "fn conditioned_prepare(chosen:u32)->bool {",
            "fn conditioned_prepare(chosen:u32)->bool {\n    conditioned_delta_valid=false;",
        );
        replace(
            &mut source,
            "fn conditioned_state(c:u32,chosen:u32,next:Six)->State {",
            "fn conditioned_state(c:u32,chosen:u32,next:Six)->State {\n    conditioned_delta_refresh(next);",
        );
        replace(
            &mut source,
            "let dq=conditioned_delta(next,i);",
            "let dq=conditioned_delta_values[i];",
        );
        source.push_str(include_str!("../shaders/soil_newton_delta_cache.wgsl"));
        source
    }

    #[cfg(all(
        any(test, feature = "cached-newton"),
        feature = "protected-newton",
        not(target_arch = "wasm32")
    ))]
    pub(crate) fn cached_contact_source(mut source: String) -> String {
        let start = source.find("fn release_merit(").unwrap();
        let end = source[start..].find("fn release_consider(").unwrap() + start;
        let mut merit = source[start..end].to_owned();
        replace(
            &mut merit,
            "let normal_length=fischer_length(node.normal_gap.xyz);var trace=vec2<f32>(0.0);var separation=vec2<f32>(0.0);",
            "let cached=4160u+54u*c;let normal_length=condensed[cached+18u];var separation=vec2<f32>(0.0);",
        );
        replace(
            &mut merit,
            "a.paired_normal[axis]=fischer_div(vec2<f32>(node.normal_gap[axis],0.0),normal_length);",
            "a.paired_normal[axis]=condensed[cached+20u+axis];",
        );
        replace(
            &mut merit,
            "trace=rate_add(trace,rate_mul(rows[c*params.width+i][axis],vec2<f32>(response[c*params.width+i][axis],0.0)));",
            "",
        );
        replace(
            &mut merit,
            "a.paired_rho=fischer_div(vec2<f32>(1.0,0.0),rate_add(fischer_div(vec2<f32>(1.0,0.0),vec2<f32>(a.mass,0.0)),trace));",
            "a.paired_rho=condensed[cached+19u];",
        );
        source.replace_range(start..end, &merit);
        replace(
            &mut source,
            "var valid=direct_contact(c);",
            "conditioned_contact_constants(c);var valid=direct_contact(c);",
        );
        source.push_str(include_str!(
            "../shaders/soil_newton_contact_constants.wgsl"
        ));
        source
    }

    #[cfg(all(test, feature = "protected-newton", not(target_arch = "wasm32")))]
    pub(crate) fn new_cached_delta_probe(
        engine: &Engine,
        input: SoilNewtonInputs<'_>,
        capacity: SoilNewtonCapacity,
    ) -> Result<Self> {
        let (lanes, storage) = execution_limits(engine);
        let limits = engine.device().ok_or_else(invalid)?.limits();
        if !protected_enabled(engine)
            || limits.max_compute_invocations_per_workgroup < lanes
            || limits.max_compute_workgroup_size_x < lanes
            || limits.max_compute_workgroup_storage_size < storage
        {
            return Err(invalid());
        }
        Self::new_with_storage(
            engine,
            input,
            capacity,
            NewtonVariant::Fischer,
            |source| {
                let source = Self::cached_delta_source(Self::stalled_release_source(source));
                let source = if std::env::var_os("MOLLA_NEWTON_STATIC_CONTACT_CACHE").is_some() {
                    Self::cached_contact_source(source)
                } else {
                    source
                };
                let width = if lanes == 64 {
                    String::new()
                } else {
                    format!("{}{lanes}\n", crate::soil_newton_width::MARKER)
                };
                format!("{}{width}{source}", crate::soil_newton_protected::PREFIX)
            },
            "newton",
            true,
        )
    }

    #[cfg(all(test, feature = "protected-newton", not(target_arch = "wasm32")))]
    pub(crate) fn new_parallel_release_probe(
        engine: &Engine,
        input: SoilNewtonInputs<'_>,
        capacity: SoilNewtonCapacity,
    ) -> Result<Self> {
        Self::new_parallel_probe(engine, input, capacity, false)
    }

    #[cfg(all(test, feature = "protected-newton", not(target_arch = "wasm32")))]
    pub(crate) fn new_schur_inverse_edge_probe(
        engine: &Engine,
        input: SoilNewtonInputs<'_>,
        capacity: SoilNewtonCapacity,
        candidate: bool,
        adjugate: bool,
        pivot_suffix: bool,
    ) -> Result<Self> {
        let (lanes, storage) = execution_limits(engine);
        let limits = engine.device().ok_or_else(invalid)?.limits();
        if lanes != 256
            || !protected_enabled(engine)
            || limits.max_compute_invocations_per_workgroup < lanes
            || limits.max_compute_workgroup_size_x < lanes
            || limits.max_compute_workgroup_storage_size < storage
        {
            return Err(invalid());
        }
        Self::new_with_storage(
            engine,
            input,
            capacity,
            NewtonVariant::Fischer,
            |source| {
                let source = Self::cached_contact_source(Self::cached_delta_source(
                    Self::stalled_release_source(source),
                ));
                let source = crate::soil_newton_parallel_release::schur_inverse::probe_source(
                    source, candidate,
                );
                let source = if candidate && pivot_suffix {
                    distributed::mixed::pivot_suffix::source(source, false)
                } else if candidate && adjugate {
                    let mut source = distributed::mixed::adjugate::source(source);
                    replace(
                        &mut source,
                        "if adjugate_inverse(c,matrix,ratio,diagonal) {return true;}",
                        "if adjugate_inverse(c,matrix,ratio,diagonal) {probe_adjugate=true;return true;}",
                    );
                    replace(
                        &mut source,
                        "probe_reduced=false;",
                        "probe_reduced=false;probe_adjugate=false;",
                    );
                    replace(
                        &mut source,
                        "select(0.0,1.0,probe_reduced)",
                        "select(0.0,1.0,probe_adjugate)",
                    );
                    source.push_str("\nvar<private> probe_adjugate:bool=false;\n");
                    source
                } else {
                    source
                };
                format!(
                    "{}{}256\n{}",
                    crate::soil_newton_protected::PREFIX,
                    crate::soil_newton_width::MARKER,
                    source
                )
            },
            "inverse_probe",
            true,
        )
    }

    #[cfg(all(test, feature = "protected-newton", not(target_arch = "wasm32")))]
    pub(crate) fn new_volatile_rounding_probe(
        engine: &Engine,
        input: SoilNewtonInputs<'_>,
        capacity: SoilNewtonCapacity,
    ) -> Result<Self> {
        Self::new_parallel_probe(engine, input, capacity, true)
    }

    #[cfg(all(test, feature = "protected-newton", not(target_arch = "wasm32")))]
    fn new_parallel_probe(
        engine: &Engine,
        input: SoilNewtonInputs<'_>,
        capacity: SoilNewtonCapacity,
        volatile: bool,
    ) -> Result<Self> {
        let (lanes, storage) = execution_limits(engine);
        let row_products = std::env::var_os("MOLLA_NEWTON_ROW_COMPARE").is_some();
        let operation_counts = std::env::var_os("MOLLA_NEWTON_COUNT_COMPARE").is_some();
        let storage = storage
            + if row_products {
                crate::soil_newton_parallel_release::row_products::SHARED_BYTES
            } else {
                0
            };
        let storage = storage
            + if operation_counts {
                crate::soil_newton_parallel_release::operation_counts::SHARED_BYTES
            } else {
                0
            };
        let limits = engine.device().ok_or_else(invalid)?.limits();
        if lanes != 256
            || !protected_enabled(engine)
            || limits.max_compute_invocations_per_workgroup < lanes
            || limits.max_compute_workgroup_size_x < lanes
            || limits.max_compute_workgroup_storage_size < storage
        {
            return Err(invalid());
        }
        Self::new_with_storage(
            engine,
            input,
            capacity,
            NewtonVariant::Fischer,
            |source| {
                #[cfg(feature = "retained-local-newton")]
                if std::env::var("MOLLA_NEWTON_NORM_COMPARE").as_deref()
                    == Ok("publication-validation")
                {
                    return publication_validation::probe(source, true, false, false);
                }
                #[cfg(feature = "retained-local-newton")]
                if std::env::var("MOLLA_NEWTON_NORM_COMPARE").as_deref()
                    == Ok("publication-adaptive")
                {
                    return retained_conditioned::publication_adaptive_probe(source, true, false);
                }
                let source = Self::cached_contact_source(Self::cached_delta_source(
                    Self::stalled_release_source(source),
                ));
                let source = crate::soil_newton_parallel_release::source(source);
                let source = if volatile || std::env::var_os("MOLLA_NEWTON_PARALLEL_CORE").is_some()
                {
                    crate::soil_newton_parallel_release::core_source(source)
                } else {
                    source
                };
                let source = if std::env::var_os("MOLLA_NEWTON_TILED_COMPARE").is_some() {
                    crate::soil_newton_parallel_release::tiled_source(source)
                } else {
                    source
                };
                let source = if std::env::var_os("MOLLA_NEWTON_CANDIDATE_COMPARE").is_some() {
                    crate::soil_newton_parallel_release::candidate_cache::source(source)
                } else {
                    source
                };
                let source = if std::env::var_os("MOLLA_NEWTON_NONLINEAR_CACHE_COMPARE").is_some() {
                    crate::soil_newton_parallel_release::nonlinear_cache::source(source)
                } else {
                    source
                };
                let source = if let Ok(repeats) = std::env::var("MOLLA_NEWTON_PHASE_COMPARE") {
                    crate::soil_newton_parallel_release::phase_probe::source(
                        source,
                        repeats
                            .strip_prefix("linear")
                            .unwrap_or(&repeats)
                            .parse()
                            .expect("phase repetition count"),
                        repeats.starts_with("linear"),
                    )
                } else {
                    source
                };
                let source = if std::env::var_os("MOLLA_NEWTON_SPARSE_COMPARE").is_some() {
                    crate::soil_newton_parallel_release::sparse_products::source(source)
                } else {
                    source
                };
                let source = if std::env::var_os("MOLLA_NEWTON_INVERSE_COMPARE").is_some() {
                    crate::soil_newton_parallel_release::inverse_skip::source(source)
                } else {
                    source
                };
                let source = if std::env::var_os("MOLLA_NEWTON_SHARED_INVERSE_COMPARE").is_some() {
                    crate::soil_newton_parallel_release::shared_inverse::source(source)
                } else {
                    source
                };
                let source = if let Ok(scope) = std::env::var("MOLLA_NEWTON_NOINLINE_COMPARE") {
                    #[cfg(feature = "schur-newton")]
                    let source = crate::soil_newton_parallel_release::schur_inverse::source(source);
                    let marker = match scope.as_str() {
                        "division" => crate::soil_newton_protected::noinline::DIVISION_MARKER,
                        "1" => crate::soil_newton_protected::noinline::MARKER,
                        _ => panic!("unsupported noinline scope"),
                    };
                    format!("{marker}{source}")
                } else {
                    source
                };
                let source = if let Ok(prefix) = std::env::var("MOLLA_NEWTON_SCHEDULE_COMPARE") {
                    crate::soil_newton_parallel_release::schedule::source(
                        source,
                        prefix.parse().expect("experimental Krylov prefix"),
                    )
                } else {
                    source
                };
                let source = if std::env::var_os("MOLLA_NEWTON_SCHUR_COMPARE").is_some() {
                    crate::soil_newton_parallel_release::schur_contact::source(source)
                } else {
                    source
                };
                let source = if std::env::var_os("MOLLA_NEWTON_SCHUR_INVERSE_COMPARE").is_some() {
                    let source = crate::soil_newton_parallel_release::schur_inverse::source(source);
                    if std::env::var_os("MOLLA_NEWTON_SCHUR_REUSE_COMPARE").is_some() {
                        crate::soil_newton_parallel_release::schur_inverse::reuse::source(source)
                    } else {
                        source
                    }
                } else {
                    source
                };
                let source = if row_products {
                    #[cfg(feature = "schur-newton")]
                    let source = crate::soil_newton_parallel_release::schur_inverse::source(source);
                    crate::soil_newton_parallel_release::row_products::source(source)
                } else {
                    source
                };
                let source = if std::env::var_os("MOLLA_NEWTON_CGS2_COMPARE").is_some() {
                    #[cfg(feature = "schur-newton")]
                    let source = crate::soil_newton_parallel_release::schur_inverse::source(source);
                    crate::soil_newton_parallel_release::cgs2::source(source)
                } else {
                    source
                };
                let source = if operation_counts {
                    let source = crate::soil_newton_parallel_release::schur_inverse::source(source);
                    crate::soil_newton_parallel_release::operation_counts::source(source)
                } else {
                    source
                };
                let source = if let Ok(scope) = std::env::var("MOLLA_NEWTON_OPERATION_COMPARE") {
                    let source = crate::soil_newton_parallel_release::schur_inverse::source(source);
                    crate::soil_newton_parallel_release::operation_cost::source(source, &scope)
                } else {
                    source
                };
                let source = if std::env::var_os("MOLLA_NEWTON_NORM_COMPARE").is_some() {
                    let source = crate::soil_newton_parallel_release::schur_inverse::source(source);
                    let source = crate::soil_newton_parallel_release::norm_tree::source(source);
                    if std::env::var("MOLLA_NEWTON_NORM_COMPARE").as_deref()
                        == Ok("publication-products")
                    {
                        crate::soil_newton_parallel_release::publication_products::source(
                            crate::soil_newton_parallel_release::active_publication::source(source),
                        )
                    } else if std::env::var("MOLLA_NEWTON_NORM_COMPARE").as_deref()
                        == Ok("publication-zero")
                    {
                        crate::soil_newton_parallel_release::zero_publication::source(
                            crate::soil_newton_parallel_release::active_publication::source(source),
                        )
                    } else if std::env::var("MOLLA_NEWTON_NORM_COMPARE").as_deref()
                        == Ok("publication-active")
                    {
                        crate::soil_newton_parallel_release::active_publication::source(source)
                    } else if std::env::var("MOLLA_NEWTON_NORM_COMPARE").as_deref()
                        == Ok("quaternary")
                    {
                        crate::soil_newton_parallel_release::norm_quaternary::source(source)
                    } else if std::env::var("MOLLA_NEWTON_NORM_COMPARE").as_deref()
                        == Ok("private-mgs")
                    {
                        crate::soil_newton_parallel_release::private_mgs::source(source)
                    } else {
                        source
                    }
                } else {
                    source
                };
                let source = if std::env::var_os("MOLLA_NEWTON_SLICED_COMPARE").is_some() {
                    let source = crate::soil_newton_parallel_release::schur_inverse::source(source);
                    let source = crate::soil_newton_parallel_release::norm_tree::source(source);
                    crate::soil_newton_parallel_release::sliced::source(source)
                } else {
                    source
                };
                format!(
                    "{}{}{}256\n{source}",
                    crate::soil_newton_protected::PREFIX,
                    if volatile {
                        crate::soil_newton_protected::VOLATILE_MARKER
                    } else {
                        ""
                    },
                    crate::soil_newton_width::MARKER
                )
            },
            "newton",
            true,
        )
        .map(|mut solver| {
            if std::env::var_os("MOLLA_NEWTON_SLICED_COMPARE").is_some() {
                solver.slice_dispatches = Some(capacity.newton_iterations + 1);
            }
            solver
        })
    }

    #[cfg(all(test, feature = "protected-newton", not(target_arch = "wasm32")))]
    pub(crate) fn new_norm_tree_edge_probe(
        engine: &Engine,
        input: SoilNewtonInputs<'_>,
        capacity: SoilNewtonCapacity,
        candidate: bool,
    ) -> Result<Self> {
        let (lanes, storage) = execution_limits(engine);
        let limits = engine.device().ok_or_else(invalid)?.limits();
        if lanes != 256
            || !protected_enabled(engine)
            || limits.max_compute_invocations_per_workgroup < lanes
            || limits.max_compute_workgroup_size_x < lanes
            || limits.max_compute_workgroup_storage_size < storage
        {
            return Err(invalid());
        }
        Self::new_with_storage(
            engine,
            input,
            capacity,
            NewtonVariant::Fischer,
            |source| {
                #[cfg(feature = "retained-local-newton")]
                if std::env::var("MOLLA_NEWTON_NORM_COMPARE").as_deref()
                    == Ok("publication-validation")
                {
                    return publication_validation::probe(source, candidate, true, false);
                }
                #[cfg(feature = "retained-local-newton")]
                if std::env::var("MOLLA_NEWTON_NORM_COMPARE").as_deref()
                    == Ok("publication-adaptive")
                {
                    return retained_conditioned::publication_adaptive_probe(
                        source, candidate, true,
                    );
                }
                let source = Self::cached_contact_source(Self::cached_delta_source(
                    Self::stalled_release_source(source),
                ));
                if std::env::var("MOLLA_NEWTON_NORM_COMPARE").as_deref() == Ok("publication-energy")
                {
                    let source =
                        crate::soil_newton_parallel_release::publication_products::probe_source(
                            source, true,
                        );
                    let source = if candidate {
                        publication_energy::source(source, false)
                    } else {
                        source
                    };
                    return format!(
                        "{}{}{}256\n{source}",
                        crate::soil_newton_protected::PREFIX,
                        crate::soil_newton_protected::VOLATILE_MARKER,
                        crate::soil_newton_width::MARKER
                    );
                }
                format!(
                    "{}{}{}256\n{}",
                    crate::soil_newton_protected::PREFIX,
                    crate::soil_newton_protected::VOLATILE_MARKER,
                    crate::soil_newton_width::MARKER,
                    crate::soil_newton_parallel_release::norm_tree::probe_source(source, candidate)
                )
            },
            "norm_probe",
            true,
        )
    }

    #[cfg(all(test, feature = "protected-newton", not(target_arch = "wasm32")))]
    pub(crate) fn new_row_products_edge_probe(
        engine: &Engine,
        input: SoilNewtonInputs<'_>,
        capacity: SoilNewtonCapacity,
        candidate: bool,
    ) -> Result<Self> {
        let (lanes, storage) = execution_limits(engine);
        let limits = engine.device().ok_or_else(invalid)?.limits();
        if lanes != 256
            || !protected_enabled(engine)
            || limits.max_compute_invocations_per_workgroup < lanes
            || limits.max_compute_workgroup_size_x < lanes
            || limits.max_compute_workgroup_storage_size
                < storage + crate::soil_newton_parallel_release::row_products::SHARED_BYTES
        {
            return Err(invalid());
        }
        Self::new_with_storage(
            engine,
            input,
            capacity,
            NewtonVariant::Fischer,
            |source| {
                let source = Self::cached_contact_source(Self::cached_delta_source(
                    Self::stalled_release_source(source),
                ));
                format!(
                    "{}{}{}256\n{}",
                    crate::soil_newton_protected::PREFIX,
                    crate::soil_newton_protected::VOLATILE_MARKER,
                    crate::soil_newton_width::MARKER,
                    crate::soil_newton_parallel_release::row_products::probe_source(
                        source, candidate
                    )
                )
            },
            "row_probe",
            true,
        )
    }

    #[cfg(all(test, feature = "protected-newton", not(target_arch = "wasm32")))]
    pub(crate) fn new_parallel_core_probe(
        engine: &Engine,
        input: SoilNewtonInputs<'_>,
        capacity: SoilNewtonCapacity,
        parallel: bool,
        mode: u32,
    ) -> Result<Self> {
        let (lanes, storage) = execution_limits(engine);
        let limits = engine.device().ok_or_else(invalid)?.limits();
        if mode > 4
            || lanes != 256
            || !protected_enabled(engine)
            || limits.max_compute_invocations_per_workgroup < lanes
            || limits.max_compute_workgroup_size_x < lanes
            || limits.max_compute_workgroup_storage_size < storage
        {
            return Err(invalid());
        }
        Self::new_with_storage(
            engine,
            input,
            capacity,
            NewtonVariant::Fischer,
            |source| {
                let source = Self::cached_contact_source(Self::cached_delta_source(
                    Self::stalled_release_source(source),
                ));
                let source =
                    crate::soil_newton_parallel_release::probe_source(source, parallel, mode);
                let source = if parallel && std::env::var_os("MOLLA_NEWTON_TILED_EDGES").is_some() {
                    crate::soil_newton_parallel_release::tiled_source(source)
                } else {
                    source
                };
                let rounding = if parallel && cfg!(feature = "volatile-newton") {
                    crate::soil_newton_protected::VOLATILE_MARKER
                } else {
                    ""
                };
                format!(
                    "{}{rounding}{}256\n{source}",
                    crate::soil_newton_protected::PREFIX,
                    crate::soil_newton_width::MARKER
                )
            },
            "core_probe",
            true,
        )
    }

    #[cfg(all(test, feature = "protected-newton", not(target_arch = "wasm32")))]
    pub(crate) fn new_wide_probe(
        engine: &Engine,
        input: SoilNewtonInputs<'_>,
        capacity: SoilNewtonCapacity,
    ) -> Result<Self> {
        let lanes = std::env::var("MOLLA_NEWTON_WIDE_LANES")
            .map_err(|_| invalid())?
            .parse::<u32>()
            .map_err(|_| invalid())?;
        let device = engine.device().ok_or_else(invalid)?;
        let limits = device.limits();
        if !matches!(lanes, 128 | 256)
            || !protected_enabled(engine)
            || limits.max_compute_invocations_per_workgroup < lanes
            || limits.max_compute_workgroup_size_x < lanes
            || limits.max_compute_workgroup_storage_size < 24000 + 24 * (lanes - 64)
        {
            return Err(invalid());
        }
        Self::new_with_storage(
            engine,
            input,
            capacity,
            NewtonVariant::Fischer,
            |source| {
                format!(
                    "{}{}{lanes}\n{}",
                    crate::soil_newton_protected::PREFIX,
                    crate::soil_newton_width::MARKER,
                    Self::stalled_release_source(source)
                )
            },
            "newton",
            true,
        )
    }

    /// Uses an eight-iteration prefix, verified direct solve and budgeted GMRES fallback.
    pub fn new_fischer_hybrid(
        engine: &Engine,
        input: SoilNewtonInputs<'_>,
        capacity: SoilNewtonCapacity,
    ) -> Result<Self> {
        let mut solver = Self::new_with_storage(
            engine,
            input,
            capacity,
            NewtonVariant::Fischer,
            Self::hybrid_source,
            "newton",
            true,
        )?;
        solver.failure_method = NewtonFailureMethod::FischerHybrid;
        Ok(solver)
    }

    pub(crate) fn hybrid_source(source: String) -> String {
        let mut source = Self::direct_source(source);
        replace(
            &mut source,
            "fn solve(id:u32) {",
            "fn solve_bounded(id:u32,limit:u32) {",
        );
        replace(
            &mut source,
            "krylov_cycle(id,params.iterations-used);",
            "krylov_cycle(id,limit-used);",
        );
        replace(
            &mut source,
            "if used>=params.iterations ||",
            "if used>=limit ||",
        );
        replace(
            &mut source,
            "prepare_condensed();if !direct_direction() {prepare_linear();solve(lane);}",
            "prepare_linear();hybrid_direction(lane);",
        );
        source.push_str(include_str!("../shaders/soil_newton_hybrid.wgsl"));
        source
    }

    #[cfg(test)]
    pub(crate) fn new_native_f64(
        engine: &Engine,
        input: SoilNewtonInputs<'_>,
        capacity: SoilNewtonCapacity,
    ) -> Result<Self> {
        Self::new_with_arithmetic(engine, input, capacity, NewtonVariant::NativeF64)
    }

    fn new_with_arithmetic(
        engine: &Engine,
        input: SoilNewtonInputs<'_>,
        capacity: SoilNewtonCapacity,
        arithmetic: NewtonVariant,
    ) -> Result<Self> {
        Self::new_with_program(
            engine,
            input,
            capacity,
            arithmetic,
            std::convert::identity,
            "newton",
        )
    }

    fn new_with_program(
        engine: &Engine,
        input: SoilNewtonInputs<'_>,
        capacity: SoilNewtonCapacity,
        arithmetic: NewtonVariant,
        program: impl FnOnce(String) -> String,
        entry: &str,
    ) -> Result<Self> {
        Self::new_with_storage(engine, input, capacity, arithmetic, program, entry, false)
    }

    fn new_with_storage(
        engine: &Engine,
        input: SoilNewtonInputs<'_>,
        capacity: SoilNewtonCapacity,
        arithmetic: NewtonVariant,
        program: impl FnOnce(String) -> String,
        entry: &str,
        condensed: bool,
    ) -> Result<Self> {
        #[cfg(test)]
        let _setup_stage = crate::soil_setup_trace::stage("newton.solver");
        engine.check_device()?;
        let diagnostic = if matches!(
            arithmetic,
            NewtonVariant::Stagnation | NewtonVariant::Fischer
        ) {
            diagnostic_inner_limit()?
        } else {
            None
        };
        let device = engine.device().ok_or_else(invalid)?;
        #[cfg(test)]
        if arithmetic == NewtonVariant::NativeF64
            && !device.features().contains(wgpu::Features::SHADER_F64)
        {
            return Err(invalid());
        }
        let limits = device.limits();
        #[cfg(all(test, feature = "protected-newton", not(target_arch = "wasm32")))]
        let clock_trace = std::env::var_os("MOLLA_NEWTON_PHASE_CLOCK").is_some();
        #[cfg(not(all(test, feature = "protected-newton", not(target_arch = "wasm32"))))]
        let clock_trace = false;
        if clock_trace && !condensed {
            return Err(invalid());
        }
        let bytes = if condensed {
            Self::required_condensed_bytes(capacity)?
        } else if arithmetic == NewtonVariant::Fischer {
            Self::required_fischer_bytes(capacity)?
        } else {
            Self::required_bytes(capacity)?
        };
        let c = u64::from(capacity.contacts.max(1));
        let n = u64::from(capacity.dofs);
        let borrowed = [
            input.point_rows,
            input.point_response,
            input.contacts,
            input.upstream_status,
            input.free_rates,
            input.mass,
        ];
        let mut sizes = vec![
            if arithmetic == NewtonVariant::Fischer {
                456 * c
            } else {
                240 * c
            },
            144 * c,
            64 * c * u64::from(capacity.krylov_iterations + 1),
            32 * c,
            32 * c,
            if arithmetic == NewtonVariant::Fischer {
                96 * c
            } else {
                32 * c
            },
            32,
            32,
            4 * c,
            64 * c,
            64 * c,
            32,
            80 * c,
            8 * n,
            32,
        ];
        if condensed {
            sizes.push(33280 + 432 * c);
        }
        if clock_trace {
            sizes.push(128);
        }
        let bytes = bytes + u64::from(clock_trace) * 128;
        if bytes > capacity.max_owned_bytes
            || limits.max_storage_buffers_per_shader_stage
                < 20 + u32::from(condensed) + u32::from(clock_trace)
            || limits.max_bindings_per_bind_group
                < 21 + u32::from(condensed) + u32::from(clock_trace)
            || limits.max_uniform_buffers_per_shader_stage < 1
            || limits.max_compute_invocations_per_workgroup < 64
            || limits.max_compute_workgroup_size_x < 64
            || limits.max_compute_workgroup_storage_size < 24000
            || borrowed
                .iter()
                .zip([16 * c * n, 16 * c * n, 64 * c, 4, 8 * n, 4 * n * n])
                .any(|(b, s)| {
                    b.size() < s
                        || b.size() > limits.max_storage_buffer_binding_size
                        || !b.usage().contains(wgpu::BufferUsages::STORAGE)
                })
            || sizes
                .iter()
                .any(|&s| s > limits.max_buffer_size || s > limits.max_storage_buffer_binding_size)
        {
            return Err(invalid());
        }
        let buffers: Vec<_> = sizes
            .iter()
            .enumerate()
            .map(|(i, &size)| {
                molla_core::gpu_create_buffer!(
                    device,
                    &wgpu::BufferDescriptor {
                        label: Some("resident Newton storage"),
                        size,
                        usage: if i == 7 {
                            wgpu::BufferUsages::UNIFORM
                                | wgpu::BufferUsages::COPY_DST
                                | if cfg!(test) {
                                    wgpu::BufferUsages::COPY_SRC
                                } else {
                                    wgpu::BufferUsages::empty()
                                }
                        } else {
                            wgpu::BufferUsages::STORAGE
                                | wgpu::BufferUsages::COPY_SRC
                                | wgpu::BufferUsages::COPY_DST
                        },
                        mapped_at_creation: false,
                    }
                )
            })
            .collect();
        molla_core::gpu_write_buffer!(
            engine.queue().unwrap(),
            &buffers[7],
            0,
            bytemuck::cast_slice(&[
                capacity.dofs,
                capacity.contacts,
                capacity.krylov_iterations,
                1e-5_f32.to_bits(),
                capacity.newton_iterations,
                0,
                0,
                0
            ])
        );
        let source = program(match diagnostic {
            Some(limit) => diagnostic_source(shader_source_for(arithmetic), limit),
            None => shader_source_for(arithmetic),
        });
        #[cfg(all(test, feature = "protected-newton", not(target_arch = "wasm32")))]
        let source = if std::env::var_os("MOLLA_NEWTON_PUBLICATION_ENERGY").is_some() {
            assert!(!clock_trace && condensed && entry == "newton");
            assert!(std::env::var_os("MOLLA_NEWTON_PUBLICATION_PROJECTION").is_none());
            assert!(std::env::var_os("MOLLA_NEWTON_PRECONDITION_UNROLL").is_none());
            publication_energy::source(source, true)
        } else if std::env::var_os("MOLLA_NEWTON_PUBLICATION_PROJECTION").is_some() {
            assert!(!clock_trace && condensed && entry == "newton");
            publication_projection::source(source)
        } else {
            source
        };
        #[cfg(all(test, feature = "protected-newton", not(target_arch = "wasm32")))]
        let source = if clock_trace {
            crate::soil_newton_precise_tests::shader_clock::phase_source(source)
        } else {
            source
        };
        #[cfg(all(test, feature = "protected-newton", not(target_arch = "wasm32")))]
        let source = if std::env::var_os("MOLLA_NEWTON_PRECONDITION_UNROLL").is_some() {
            assert!(!clock_trace && condensed && entry == "newton");
            precondition_unroll::source(source)
        } else if std::env::var_os("MOLLA_NEWTON_CONTROL_SNAPSHOT").is_some() {
            assert!(!clock_trace && condensed && entry == "newton");
            control_snapshot::source(source)
        } else {
            source
        };
        #[cfg(all(test, feature = "protected-newton", not(target_arch = "wasm32")))]
        let source = if std::env::var_os("MOLLA_NEWTON_BARRIER_FOLD").is_some() {
            assert!(!clock_trace && condensed && entry == "newton");
            assert!(source.starts_with(crate::soil_newton_protected::PREFIX));
            for conflict in [
                "MOLLA_NEWTON_CONTROL_SNAPSHOT",
                "MOLLA_NEWTON_PRECONDITION_UNROLL",
                "MOLLA_NEWTON_PUBLICATION_ENERGY",
                "MOLLA_NEWTON_PUBLICATION_PROJECTION",
                "MOLLA_NEWTON_BOUNDED_RESET",
                "MOLLA_NEWTON_PUBLICATION_INPUT_CACHE",
            ] {
                assert!(std::env::var_os(conflict).is_none());
            }
            format!(
                "{source}\n{}",
                crate::soil_newton_protected::barrier_fold::MARKER
            )
        } else {
            source
        };
        #[cfg(all(test, feature = "soa-scratch-newton", not(target_arch = "wasm32")))]
        let source = if std::env::var_os("MOLLA_NEWTON_BOUNDED_RESET").is_some() {
            assert!(!clock_trace && condensed && entry == "newton");
            for conflict in [
                "MOLLA_NEWTON_CONTROL_SNAPSHOT",
                "MOLLA_NEWTON_PRECONDITION_UNROLL",
                "MOLLA_NEWTON_PUBLICATION_ENERGY",
                "MOLLA_NEWTON_PUBLICATION_PROJECTION",
                "MOLLA_NEWTON_PUBLICATION_INPUT_CACHE",
            ] {
                assert!(std::env::var_os(conflict).is_none());
            }
            bounded_reset::source(source)
        } else {
            source
        };
        #[cfg(all(test, feature = "soa-scratch-newton", not(target_arch = "wasm32")))]
        let source = if std::env::var_os("MOLLA_NEWTON_PUBLICATION_INPUT_CACHE").is_some() {
            assert!(!clock_trace && condensed && entry == "newton");
            for conflict in [
                "MOLLA_NEWTON_CONTROL_SNAPSHOT",
                "MOLLA_NEWTON_PRECONDITION_UNROLL",
                "MOLLA_NEWTON_PUBLICATION_ENERGY",
                "MOLLA_NEWTON_PUBLICATION_PROJECTION",
            ] {
                assert!(std::env::var_os(conflict).is_none());
            }
            publication_input_cache::source(source)
        } else {
            source
        };
        #[cfg(all(test, feature = "soa-scratch-newton", not(target_arch = "wasm32")))]
        let source =
            if std::env::var_os("MOLLA_NEWTON_SCALAR_PRODUCT").is_some() && entry == "newton" {
                assert!(!clock_trace && condensed);
                for conflict in [
                    "MOLLA_NEWTON_CONTROL_SNAPSHOT",
                    "MOLLA_NEWTON_PRECONDITION_UNROLL",
                    "MOLLA_NEWTON_PUBLICATION_ENERGY",
                    "MOLLA_NEWTON_PUBLICATION_PROJECTION",
                    "MOLLA_NEWTON_BARRIER_FOLD",
                    "MOLLA_NEWTON_BOUNDED_RESET",
                    "MOLLA_NEWTON_PUBLICATION_INPUT_CACHE",
                ] {
                    assert!(std::env::var_os(conflict).is_none());
                }
                scalar_product::source(source)
            } else {
                source
            };
        #[cfg(all(test, feature = "soa-scratch-newton", not(target_arch = "wasm32")))]
        let source = if std::env::var_os("MOLLA_NEWTON_OPERATOR_ROWS").is_some() {
            assert!(!clock_trace && condensed && entry == "newton");
            for conflict in [
                "MOLLA_NEWTON_CONTROL_SNAPSHOT",
                "MOLLA_NEWTON_PRECONDITION_UNROLL",
                "MOLLA_NEWTON_PUBLICATION_ENERGY",
                "MOLLA_NEWTON_PUBLICATION_PROJECTION",
                "MOLLA_NEWTON_BARRIER_FOLD",
                "MOLLA_NEWTON_BOUNDED_RESET",
                "MOLLA_NEWTON_PUBLICATION_INPUT_CACHE",
                "MOLLA_NEWTON_SCALAR_PRODUCT",
            ] {
                assert!(std::env::var_os(conflict).is_none());
            }
            operator_rows::source(source)
        } else {
            source
        };
        #[cfg(all(test, feature = "soa-scratch-newton", not(target_arch = "wasm32")))]
        let source = if std::env::var_os("MOLLA_NEWTON_RESIDUAL_FUSION").is_some() {
            assert!(!clock_trace && condensed && entry == "newton");
            residual_fusion::source(source)
        } else {
            source
        };
        #[cfg(all(test, feature = "soa-scratch-newton", not(target_arch = "wasm32")))]
        let source = if std::env::var_os("MOLLA_NEWTON_NUMERICAL_GUARD").is_some() {
            assert!(!clock_trace && condensed && entry == "newton");
            numerical_guard::source(source)
        } else {
            source
        };
        #[cfg(test)]
        let setup = std::env::var_os("MOLLA_NEWTON_SETUP_TRACE").map(|_| std::time::Instant::now());
        #[cfg(not(target_arch = "wasm32"))]
        let key = program_key(&source, entry, condensed);
        let create = |device: &wgpu::Device| -> Result<NewtonProgram> {
            #[cfg(test)]
            let _setup_stage = crate::soil_setup_trace::stage("newton.solver.program");
            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("resident Newton layout"),
                entries: &(0..21 + u32::from(condensed) + u32::from(clock_trace))
                    .map(|binding| wgpu::BindGroupLayoutEntry {
                        binding,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: if binding == 10 {
                                wgpu::BufferBindingType::Uniform
                            } else {
                                wgpu::BufferBindingType::Storage {
                                    read_only: matches!(binding, 1 | 2 | 3 | 12 | 13 | 17),
                                }
                            },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    })
                    .collect::<Vec<_>>(),
            });
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("resident Newton pipeline layout"),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
            #[cfg(test)]
            if setup.is_some() {
                eprintln!("Newton setup entry={entry} module begin");
            }
            let runtime = || -> Result<wgpu::ShaderModule> {
                #[cfg(test)]
                let _setup_stage = crate::soil_setup_trace::stage("newton.solver.module");
                #[cfg(all(test, feature = "protected-newton", not(target_arch = "wasm32")))]
                if clock_trace {
                    return Ok(
                        crate::soil_newton_precise_tests::shader_clock::phase_shader(
                            device, &source, entry,
                        ),
                    );
                }
                #[cfg(all(feature = "protected-newton", not(target_arch = "wasm32")))]
                if crate::soil_newton_protected::uses_compiler(&source)? {
                    return crate::soil_newton_protected::shader(device, &source, entry);
                }
                Ok(device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("resident Newton"),
                    source: wgpu::ShaderSource::Wgsl(source.clone().into()),
                }))
            };
            #[cfg(test)]
            let module = if source.starts_with("// NEWTON_PRECISE_CONTROL\n")
                || source.starts_with("// NEWTON_PRECISE_CANDIDATE\n")
            {
                crate::soil_newton_precise_tests::shader(
                    device,
                    &source,
                    entry,
                    source.starts_with("// NEWTON_PRECISE_CANDIDATE\n"),
                )
            } else {
                runtime()?
            };
            #[cfg(not(test))]
            let module = runtime()?;
            #[cfg(test)]
            if let Some(start) = setup {
                eprintln!(
                    "Newton setup entry={entry} module_ms={} pipeline begin",
                    start.elapsed().as_millis()
                );
            }
            #[cfg(test)]
            let _setup_pipeline = crate::soil_setup_trace::stage("newton.solver.pipeline");
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("resident Newton"),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: engine.pipeline_cache(),
            });
            Ok(NewtonProgram { layout, pipeline })
        };
        #[cfg(not(target_arch = "wasm32"))]
        let program = engine.cached_gpu_program(&key, create)?;
        #[cfg(target_arch = "wasm32")]
        let program = create(device)?;
        let layout = &program.layout;
        let pipeline = program.pipeline.clone();
        #[cfg(test)]
        if let Some(start) = setup {
            eprintln!(
                "Newton setup entry={entry} total_ms={}",
                start.elapsed().as_millis()
            );
        }
        let mut bindings = vec![
            &buffers[0],
            input.point_rows,
            input.point_response,
            input.contacts,
            &buffers[1],
            &buffers[2],
            &buffers[3],
            &buffers[4],
            &buffers[5],
            &buffers[6],
            &buffers[7],
            &buffers[8],
            input.upstream_status,
            input.free_rates,
            &buffers[9],
            &buffers[10],
            &buffers[11],
            input.mass,
            &buffers[12],
            &buffers[13],
            &buffers[14],
        ];
        if condensed {
            bindings.push(&buffers[15]);
        }
        if clock_trace {
            bindings.push(&buffers[16]);
        }
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("resident Newton bindings"),
            layout: &layout,
            entries: &bindings
                .iter()
                .enumerate()
                .map(|(i, b)| wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource: b.as_entire_binding(),
                })
                .collect::<Vec<_>>(),
        });
        Ok(Self {
            failure_method: match arithmetic {
                NewtonVariant::Stagnation => NewtonFailureMethod::StagnationPairs,
                NewtonVariant::Fischer => NewtonFailureMethod::FischerPairs,
                _ => NewtonFailureMethod::Unspecified,
            },
            pipeline,
            group,
            state: buffers[9].clone(),
            status: buffers[11].clone(),
            outputs: buffers[12].clone(),
            rates: buffers[13].clone(),
            publication_status: buffers[14].clone(),
            bytes,
            #[cfg(test)]
            scratch_soa: source.contains("var<storage,read_write> block_words:array<f32>;"),
            #[cfg(test)]
            scratch: buffers,
            #[cfg(test)]
            arithmetic,
            #[cfg(test)]
            slice_dispatches: None,
            #[cfg(test)]
            prefix_pipeline: None,
            #[cfg(test)]
            clock_trace,
        })
    }

    /// Clears sticky status and starts from zero impulses.
    pub fn encode_reset(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.clear_buffer(&self.status, 0, None);
        encoder.clear_buffer(&self.state, 0, None);
    }
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        #[cfg(test)]
        if self.clock_trace {
            encoder.clear_buffer(&self.scratch[16], 0, None);
        }
        encoder.clear_buffer(&self.publication_status, 0, None);
        #[cfg(all(test, feature = "protected-newton", not(target_arch = "wasm32")))]
        if let Some(dispatches) = self.slice_dispatches {
            encoder.clear_buffer(
                &self.scratch[15],
                crate::soil_newton_parallel_release::sliced::OFFSET,
                Some(crate::soil_newton_parallel_release::sliced::BYTES),
            );
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.group, &[]);
            for _ in 0..dispatches {
                pass.dispatch_workgroups(1, 1, 1);
                molla_core::gpu_metrics::record_dispatch();
            }
            return;
        }
        let mut pass = encoder.begin_compute_pass(&Default::default());
        #[cfg(test)]
        if let Some(prefix) = &self.prefix_pipeline {
            pass.set_pipeline(prefix);
            pass.set_bind_group(0, &self.group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
            molla_core::gpu_metrics::record_dispatch();
        }
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
        molla_core::gpu_metrics::record_dispatch();
    }
    pub fn state_buffer(&self) -> &wgpu::Buffer {
        &self.state
    }
    #[cfg(all(test, feature = "protected-newton", not(target_arch = "wasm32")))]
    pub(crate) fn set_publication_projection_probe(&self, engine: &Engine, enabled: bool) {
        assert!(std::env::var_os("MOLLA_NEWTON_PUBLICATION_PROJECTION").is_some());
        engine.queue().unwrap().write_buffer(
            &self.scratch[7],
            28,
            &u32::from(enabled).to_le_bytes(),
        );
    }
    #[cfg(all(test, feature = "protected-newton", not(target_arch = "wasm32")))]
    pub(crate) fn set_publication_energy_probe(&self, engine: &Engine, enabled: bool) {
        assert!(std::env::var_os("MOLLA_NEWTON_PUBLICATION_ENERGY").is_some());
        engine.queue().unwrap().write_buffer(
            &self.scratch[7],
            28,
            &u32::from(enabled).to_le_bytes(),
        );
    }
    #[cfg(all(test, feature = "protected-newton", not(target_arch = "wasm32")))]
    pub(crate) fn set_precondition_unroll_probe(&self, engine: &Engine, enabled: bool) {
        assert!(std::env::var_os("MOLLA_NEWTON_PRECONDITION_UNROLL").is_some());
        engine.queue().unwrap().write_buffer(
            &self.scratch[7],
            28,
            &u32::from(enabled).to_le_bytes(),
        );
    }
    #[cfg(all(test, feature = "protected-newton", not(target_arch = "wasm32")))]
    pub(crate) fn set_control_snapshot_probe(&self, engine: &Engine, enabled: bool) {
        assert!(std::env::var_os("MOLLA_NEWTON_CONTROL_SNAPSHOT").is_some());
        engine.queue().unwrap().write_buffer(
            &self.scratch[7],
            28,
            &u32::from(enabled).to_le_bytes(),
        );
    }
    #[cfg(all(test, feature = "soa-scratch-newton", not(target_arch = "wasm32")))]
    pub(crate) fn set_publication_input_cache_probe(&self, engine: &Engine, enabled: bool) {
        assert!(std::env::var_os("MOLLA_NEWTON_PUBLICATION_INPUT_CACHE").is_some());
        engine.queue().unwrap().write_buffer(
            &self.scratch[7],
            28,
            &u32::from(enabled).to_le_bytes(),
        );
    }
    #[cfg(all(test, feature = "soa-scratch-newton", not(target_arch = "wasm32")))]
    pub(crate) fn set_scalar_product_probe(&self, engine: &Engine, enabled: bool) {
        assert!(std::env::var_os("MOLLA_NEWTON_SCALAR_PRODUCT").is_some());
        engine.queue().unwrap().write_buffer(
            &self.scratch[7],
            28,
            &u32::from(enabled).to_le_bytes(),
        );
    }
    #[cfg(all(test, feature = "soa-scratch-newton", not(target_arch = "wasm32")))]
    pub(crate) fn set_operator_rows_probe(&self, engine: &Engine, enabled: bool) {
        assert!(std::env::var_os("MOLLA_NEWTON_OPERATOR_ROWS").is_some());
        engine.queue().unwrap().write_buffer(
            &self.scratch[7],
            28,
            &u32::from(enabled).to_le_bytes(),
        );
    }
    #[cfg(all(test, feature = "soa-scratch-newton", not(target_arch = "wasm32")))]
    pub(crate) fn set_residual_fusion_probe(&self, engine: &Engine, enabled: bool) {
        assert!(std::env::var_os("MOLLA_NEWTON_RESIDUAL_FUSION").is_some());
        engine.queue().unwrap().write_buffer(
            &self.scratch[7],
            28,
            &u32::from(enabled).to_le_bytes(),
        );
    }
    #[cfg(all(test, feature = "soa-scratch-newton", not(target_arch = "wasm32")))]
    pub(crate) fn set_numerical_guard_probe(&self, engine: &Engine, enabled: bool) {
        assert!(std::env::var_os("MOLLA_NEWTON_NUMERICAL_GUARD").is_some());
        engine.queue().unwrap().write_buffer(
            &self.scratch[7],
            28,
            &u32::from(enabled).to_le_bytes(),
        );
    }
    #[cfg(all(test, feature = "soa-scratch-newton", not(target_arch = "wasm32")))]
    pub(crate) fn numerical_guard_probe_parameters(&self) -> &wgpu::Buffer {
        &self.scratch[7]
    }
    #[cfg(all(test, feature = "protected-newton", not(target_arch = "wasm32")))]
    pub(crate) fn clock_buffer(&self) -> &wgpu::Buffer {
        assert!(self.clock_trace);
        &self.scratch[16]
    }
    #[cfg(all(test, feature = "protected-newton", not(target_arch = "wasm32")))]
    pub(crate) fn set_phase_clock_probe(&self, engine: &Engine, enabled: bool) {
        assert!(self.clock_trace);
        engine.queue().unwrap().write_buffer(
            &self.scratch[7],
            28,
            &u32::from(!enabled).to_le_bytes(),
        );
    }
    pub fn outputs_buffer(&self) -> &wgpu::Buffer {
        &self.outputs
    }
    pub fn rates_buffer(&self) -> &wgpu::Buffer {
        &self.rates
    }
    pub fn publication_status_buffer(&self) -> &wgpu::Buffer {
        &self.publication_status
    }
    /// Fault, accepted steps, Krylov iterations plus condensed solves, trials,
    /// residual norm, physical stopping estimate, last damping, diagnostic limit.
    /// Fault bits: input=1, nonfinite=2, iteration=8, line search=16, diagnostic=32.
    pub fn status_buffer(&self) -> &wgpu::Buffer {
        &self.status
    }
    pub fn owned_buffer_bytes(&self) -> u64 {
        self.bytes
    }
    #[cfg(test)]
    pub(crate) fn replay_dispatch_count(&self) -> u64 {
        self.slice_dispatches
            .map(u64::from)
            .unwrap_or(1 + u64::from(self.prefix_pipeline.is_some()))
    }
    #[cfg(test)]
    pub(crate) fn use_identity_preconditioner(&mut self, engine: &Engine) {
        let mut source = shader_source_for(self.arithmetic);
        let start = source.find("fn build_inverse(c:u32)->bool {").unwrap();
        let end = start + source[start..].find("@compute").unwrap();
        source.replace_range(
            start..end,
            "fn build_inverse(c:u32)->bool {return false;}\n",
        );
        if let Some(limit) = diagnostic_inner_limit().unwrap() {
            source = diagnostic_source(source, limit);
        }
        self.use_probe(engine, source, "newton");
    }
    #[cfg(test)]
    pub(crate) fn use_serial_publication(&mut self, engine: &Engine) {
        let mut source = shader_source_for(self.arithmetic);
        replace(
            &mut source,
            &crate::soil_newton_publish_gpu::resident_source(),
            &crate::soil_newton_publish_gpu::resident_serial_source(),
        );
        self.use_probe(engine, source, "newton");
    }
    #[cfg(test)]
    pub(crate) fn use_stopping_probe(&mut self, engine: &Engine, singular: bool) {
        let mut shader = shader_source_for(self.arithmetic);
        if self.arithmetic == NewtonVariant::Fischer {
            replace(
                &mut shader,
                "krylov_cycle(id,params.iterations-used);",
                "krylov_cycle(id,params.iterations-used);if lane==0u {report.trials++;}",
            );
        } else {
            replace(
                &mut shader,
                "if stagnant {break;}",
                "if stagnant {if lane==0u {report.trials++;}break;}",
            );
        }
        let mut source = format!(
            "{}\n@compute @workgroup_size(64) fn stopping_probe(@builtin(local_invocation_index) id:u32) {{
                lane=id;rate_lane=id;
                if lane==0u {{atomicStore(&status.fault,0u);status.active_count=params.contacts;}}
                for(var c=lane;c<params.contacts;c+=64u) {{
                    active_indices[c]=c;blocks[c]=Block();inverses[c]=Inverse();direction[c]=Six();
                    for(var i=0u;i<6u;i++) {{
                        let first=c==0u && i==0u;
                        blocks[c].residual[i]=select(0.01,1.0,first);
                        blocks[c].direct[7u*i]=select(1.0+f32(6u*c+i)/42.0,0.0,{singular} && first);
                        inverses[c].a[7u*i]=1.0;
                    }}
                }}
                storageBarrier();solve(lane);
                if lane==0u {{atomicStore(&report.fault,atomicLoad(&status.fault));report.linear_iterations=status.iterations;report.residual=status.residual;report.physical=status.relative;report.diagnostic_limit=bitcast<u32>(status.pad);}}
            }}",
            shader
        );
        if self.arithmetic == NewtonVariant::Fischer {
            replace(
                &mut source,
                "blocks[c].direct[7u*i]=select(",
                "blocks[c].direct[7u*i]=vec2<f32>(select(",
            );
            replace(
                &mut source,
                &format!("{singular} && first);"),
                &format!("{singular} && first),0.0);"),
            );
        }
        self.use_probe(engine, source, "stopping_probe");
    }
    #[cfg(test)]
    pub(crate) fn new_direct_probe(
        engine: &Engine,
        input: SoilNewtonInputs<'_>,
        capacity: SoilNewtonCapacity,
        singular: bool,
    ) -> Result<Self> {
        Self::new_with_storage(
            engine,
            input,
            capacity,
            NewtonVariant::Fischer,
            |mut source| {
                replace(
                    &mut source,
                    "var<private> fischer_smoothing:f32=0.0;",
                    "var<private> fischer_smoothing:f32=0.01;",
                );
                format!("{source}\n{}\n{}\n@compute @workgroup_size(64) fn direct_probe(@builtin(local_invocation_index) id:u32) {{
                lane=id;rate_lane=id;
                if lane==0u {{atomicStore(&status.fault,0u);status.active_count=0u;
                    for(var c=0u;c<params.contacts;c++) {{if contacts[c].soil_mass.w>0.0 {{active_indices[status.active_count]=c;status.active_count++;}}}}
                }}
                storageBarrier();evaluate();
                if {singular} {{for(var slot=lane;slot<status.active_count;slot+=64u) {{
                    let c=active_indices[slot];blocks[c]=Block();
                    for(var i=0u;i<6u;i++) {{blocks[c].residual[i]=0.1*f32(i+1u);blocks[c].direct[7u*i]=vec2<f32>(select(0.0,1.0,i>=3u),0.0);}}
                    for(var i=0u;i<3u;i++) {{blocks[c].machine[4u*i]=vec2<f32>(1.0,0.0);}}
                }}}}
                storageBarrier();prepare_condensed();let accepted=direct_direction();
                if !accepted {{prepare_linear();solve(lane);}}
                if lane==0u {{atomicStore(&report.fault,atomicLoad(&status.fault));report.linear_iterations=status.iterations;report.trials=select(1u,0u,accepted);report.physical=status.relative;}}
            }}", include_str!("../shaders/soil_newton_condensed.wgsl"), include_str!("../shaders/soil_newton_direct.wgsl"))
            },
            "direct_probe",
            true,
        )
    }

    #[cfg(test)]
    pub(crate) fn new_hybrid_budget_probe(
        engine: &Engine,
        input: SoilNewtonInputs<'_>,
        capacity: SoilNewtonCapacity,
    ) -> Result<Self> {
        Self::new_with_storage(
            engine,
            input,
            capacity,
            NewtonVariant::Fischer,
            |source| {
                let mut source = Self::hybrid_source(source);
                replace(
                    &mut source,
                    "if fault!=8u {return;}",
                    "if fault!=8u {return;} if lane==0u {report.trials+=1u;}",
                );
                replace(
                    &mut source,
                    "prepare_linear();solve_bounded(id,params.iterations-used);",
                    "if lane==0u {report.diagnostic_limit=1u;} prepare_linear();solve_bounded(id,params.iterations-used);",
                );
                format!("{source}\n@compute @workgroup_size(64) fn hybrid_budget_probe(@builtin(local_invocation_index) id:u32) {{
                    lane=id;rate_lane=id;
                    if lane==0u {{atomicStore(&status.fault,0u);status.active_count=0u;
                        for(var c=0u;c<params.contacts;c++) {{if contacts[c].soil_mass.w>0.0 {{active_indices[status.active_count]=c;status.active_count++;}}}}
                    }}
                    storageBarrier();
                    for(var slot=lane;slot<status.active_count;slot+=64u) {{
                        let c=active_indices[slot];blocks[c]=Block();
                        for(var i=0u;i<6u;i++) {{blocks[c].residual[i]=0.1*f32(6u*c+i+1u);blocks[c].direct[7u*i]=vec2<f32>(select(1.0,0.0,i==0u),0.0);}}
                        blocks[c].machine[0]=vec2<f32>(1.0,0.0);
                    }}
                    storageBarrier();prepare_linear();hybrid_direction(lane);
                    if lane==0u {{atomicStore(&report.fault,atomicLoad(&status.fault));report.linear_iterations=status.iterations;report.physical=status.relative;}}
                }}")
            },
            "hybrid_budget_probe",
            true,
        )
    }

    #[cfg(test)]
    pub(crate) fn new_precise_control_probe(
        engine: &Engine,
        input: SoilNewtonInputs<'_>,
        capacity: SoilNewtonCapacity,
    ) -> Result<Self> {
        Self::new_precise_probe(engine, input, capacity, false, None)
    }

    #[cfg(test)]
    pub(crate) fn new_precise_candidate_probe(
        engine: &Engine,
        input: SoilNewtonInputs<'_>,
        capacity: SoilNewtonCapacity,
    ) -> Result<Self> {
        Self::new_precise_probe(engine, input, capacity, true, None)
    }

    #[cfg(test)]
    pub(crate) fn new_precise_local_probe(
        engine: &Engine,
        input: SoilNewtonInputs<'_>,
        capacity: SoilNewtonCapacity,
        candidate: bool,
        contact: u32,
    ) -> Result<Self> {
        if contact >= capacity.contacts {
            return Err(invalid());
        }
        Self::new_precise_probe(engine, input, capacity, candidate, Some(contact))
    }

    #[cfg(test)]
    fn new_precise_probe(
        engine: &Engine,
        input: SoilNewtonInputs<'_>,
        capacity: SoilNewtonCapacity,
        candidate: bool,
        local: Option<u32>,
    ) -> Result<Self> {
        engine.check_device()?;
        let device = engine.device().ok_or_else(invalid)?;
        if engine.adapter_info().is_none_or(|a| a.backend != "Vulkan")
            || !device
                .features()
                .contains(wgpu::Features::PASSTHROUGH_SHADERS)
        {
            return Err(invalid());
        }
        let mode = if candidate { "CANDIDATE" } else { "CONTROL" };
        let mut solver = Self::new_with_storage(
            engine,
            input,
            capacity,
            NewtonVariant::Fischer,
            |source| {
                let mut source = Self::conditioned_source(source);
                if let Some(contact) = local {
                    source = crate::soil_newton_precise_tests::local_trace_source(source, contact);
                }
                let boundary = if candidate {
                    crate::soil_newton_precise_tests::boundary_variant()
                } else {
                    "control"
                };
                format!(
                    "// NEWTON_PRECISE_{mode}\n// boundary={boundary}\n{}",
                    source
                )
            },
            if local.is_some() {
                "precision_local"
            } else {
                "newton"
            },
            true,
        )?;
        solver.failure_method = NewtonFailureMethod::Unspecified;
        Ok(solver)
    }

    #[cfg(test)]
    pub(crate) fn new_without_release_probe(
        engine: &Engine,
        input: SoilNewtonInputs<'_>,
        capacity: SoilNewtonCapacity,
    ) -> Result<Self> {
        let mut solver = Self::new_with_storage(
            engine,
            input,
            capacity,
            NewtonVariant::Fischer,
            |source| {
                let mut source = Self::conditioned_source(source);
                replace(&mut source, "attempt==0u && conditioned_release()", "false");
                source
            },
            "newton",
            true,
        )?;
        solver.failure_method = NewtonFailureMethod::Unspecified;
        Ok(solver)
    }

    #[cfg(test)]
    pub(crate) fn new_conditioned_probe(
        engine: &Engine,
        input: SoilNewtonInputs<'_>,
        capacity: SoilNewtonCapacity,
        selected: u32,
        singular: Option<u32>,
    ) -> Result<Self> {
        let corrupt = singular
            .map(|c| format!("if lane==0u {{blocks[{c}u].direct=array<vec2<f32>,36>();}}"))
            .unwrap_or_default();
        Self::new_with_storage(
            engine,
            input,
            capacity,
            NewtonVariant::Fischer,
            |source| {
                let source = Self::conditioned_source(source);
                format!("{source}\n@compute @workgroup_size(64) fn conditioned_probe(@builtin(local_invocation_index) id:u32) {{
                lane=id;rate_lane=id;
                if lane==0u {{atomicStore(&status.fault,0u);status.active_count=0u;
                    for(var c=0u;c<params.contacts;c++) {{if contacts[c].soil_mass.w>0.0 {{active_indices[status.active_count]=c;status.active_count++;}}}}
                }}
                storageBarrier();
                for(var slot=lane;slot<status.active_count;slot+=64u) {{let c=active_indices[slot];states[c]=State(Six(vec4<f32>(0.1,0.1,0.1,0.0),vec4<f32>(0.1,0.1,0.1,0.0)),Six());}}
                storageBarrier();evaluate();
                {corrupt}
                storageBarrier();conditioned_cache();let accepted=conditioned_prepare({selected}u);
                if accepted {{for(var slot=lane;slot<status.active_count;slot+=64u) {{let c=active_indices[slot];states[c]=conditioned_state(c,{selected}u,Six(vec4<f32>(0.17,-0.01,0.12,0.0),vec4<f32>(0.04,0.15,0.1,0.0)));}}}}
                storageBarrier();if lane==0u {{atomicStore(&report.fault,select(8u,0u,accepted));}}
            }}")
            },
            "conditioned_probe",
            true,
        )
    }

    #[cfg(test)]
    pub(crate) fn new_condensed_probe(
        engine: &Engine,
        input: SoilNewtonInputs<'_>,
        capacity: SoilNewtonCapacity,
        damping: f32,
    ) -> Result<Self> {
        Self::new_with_storage(
            engine,
            input,
            capacity,
            NewtonVariant::Fischer,
            |source| {
                format!("{source}\n{}\n@compute @workgroup_size(64) fn condensed_probe(@builtin(local_invocation_index) id:u32) {{
                lane=id;rate_lane=id;
                if lane==0u {{atomicStore(&status.fault,0u);status.active_count=0u;
                    for(var c=0u;c<params.contacts;c++) {{if contacts[c].soil_mass.w>0.0 {{active_indices[status.active_count]=c;status.active_count++;}}}}
                }}
                storageBarrier();evaluate();prepare_linear();let accepted=condensed_direction({damping:?});
                if lane==0u {{atomicStore(&report.fault,select(8u,0u,accepted));report.linear_iterations=status.iterations;report.physical=status.relative;}}
            }}", include_str!("../shaders/soil_newton_condensed.wgsl"))
            },
            "condensed_probe",
            true,
        )
    }

    #[cfg(test)]
    pub(crate) fn use_damped_probe(&mut self, engine: &Engine, damping: f32) {
        assert!(damping.is_finite() && damping > 0.0);
        let source = format!(
            "{}\n@compute @workgroup_size(64) fn damped_probe(@builtin(local_invocation_index) id:u32) {{
                lane=id;rate_lane=id;
                if lane==0u {{
                    atomicStore(&status.fault,0u);status.active_count=0u;
                    for(var c=0u;c<params.contacts;c++) {{if contacts[c].soil_mass.w>0.0 {{active_indices[status.active_count]=c;status.active_count++;}}}}
                }}
                storageBarrier();evaluate();prepare_linear();least_squares(lane,{damping:?});
                if lane==0u {{atomicStore(&report.fault,atomicLoad(&status.fault));report.linear_iterations=status.iterations;report.residual=status.residual;report.physical=status.relative;}}
            }}",
            shader_source_for(self.arithmetic)
        );
        self.use_probe(engine, source, "damped_probe");
    }
    #[cfg(test)]
    pub(crate) fn use_linear_probe(&mut self, engine: &Engine) {
        assert!(self.arithmetic == NewtonVariant::Fischer);
        self.use_probe(
            engine,
            Self::linear_probe_source(shader_source_for(self.arithmetic)),
            "linear_probe",
        );
    }
    #[cfg(test)]
    pub(crate) fn new_fischer_linear_probe(
        engine: &Engine,
        input: SoilNewtonInputs<'_>,
        capacity: SoilNewtonCapacity,
    ) -> Result<Self> {
        Self::new_with_program(
            engine,
            input,
            capacity,
            NewtonVariant::Fischer,
            Self::linear_probe_source,
            "linear_probe",
        )
    }
    #[cfg(test)]
    fn linear_probe_source(source: String) -> String {
        format!("{source}\n@compute @workgroup_size(64) fn linear_probe(@builtin(local_invocation_index) id:u32) {{
            lane=id;rate_lane=id;
            if lane==0u {{atomicStore(&status.fault,0u);status.active_count=0u;
                for(var c=0u;c<params.contacts;c++) {{if contacts[c].soil_mass.w>0.0 {{active_indices[status.active_count]=c;status.active_count++;}}}}
            }}
            storageBarrier();evaluate();prepare_linear();solve(lane);
            if lane==0u {{atomicStore(&report.fault,atomicLoad(&status.fault));report.linear_iterations=status.iterations;report.residual=status.residual;report.physical=status.relative;}}
        }}")
    }
    #[cfg(test)]
    pub(crate) fn use_paired_direction_probe(&mut self, engine: &Engine) {
        assert!(self.arithmetic == NewtonVariant::Fischer);
        let source = format!("{}\n@compute @workgroup_size(64) fn paired_direction_probe(@builtin(local_invocation_index) id:u32) {{
            lane=id;rate_lane=id;
            if lane==0u {{status.active_count=params.contacts;}}
            let stride=max(params.contacts,1u);let last=params.iterations*stride;
            for(var c=lane;c<params.contacts;c+=64u) {{
                active_indices[c]=c;blocks[c]=Block();
                for(var i=0u;i<6u;i++) {{blocks[c].direct[7u*i]=vec2<f32>(1.0,0.0);}}
                basis[last+c].z=Six(vec4<f32>(7.0),vec4<f32>(7.0));
                direction[stride+c]=Six(vec4<f32>(17.0),vec4<f32>(17.0));direction[2u*stride+c]=Six(vec4<f32>(19.0),vec4<f32>(19.0));
                basis[stride+c].v=Six(vec4<f32>(11.0),vec4<f32>(13.0));
            }}
            storageBarrier();prepare_linear();
            for(var c=lane;c<params.contacts;c+=64u) {{
                for(var i=0u;i<6u;i++) {{if component(basis[last+c].z,i)!=0.0 {{atomicOr(&report.fault,1u);}}}}
                for(var i=0u;i<6u;i++) {{if any(base_direction_pair(c,i)!=vec2<f32>(0.0)) {{atomicOr(&report.fault,4u);}}}}
                if any(basis[stride+c].v.k!=vec4<f32>(11.0)) || any(basis[stride+c].v.w!=vec4<f32>(13.0)) {{atomicOr(&report.fault,2u);}}
                blocks[c]=Block();blocks[c].direct[0]=vec2<f32>(1.0,0.0);blocks[c].direct[1]=vec2<f32>(1.0,0.0);
                blocks[c].direct[6]=vec2<f32>(1.0,2e-8);blocks[c].direct[7]=vec2<f32>(1.0,-1e-8);
                direction[c]=Six(vec4<f32>(1.0,-1.0,0.0,0.0),vec4<f32>(0.0));
                basis[last+c].z=Six(vec4<f32>(2e-8,1e-8,0.0,0.0),vec4<f32>(0.0));
            }}
            storageBarrier();apply_direction();
        }}", shader_source_for(self.arithmetic));
        self.use_probe(engine, source, "paired_direction_probe");
    }
    #[cfg(test)]
    pub(crate) fn use_transpose_probe(&mut self, engine: &Engine) {
        let source = format!(
            "{}\n@compute @workgroup_size(64) fn transpose_probe(@builtin(local_invocation_index) id:u32) {{
                lane=id;rate_lane=id;
                if lane==0u {{atomicStore(&status.fault,0u);status.active_count=0u;
                    for(var c=0u;c<params.contacts;c++) {{if contacts[c].soil_mass.w>0.0 {{active_indices[status.active_count]=c;status.active_count++;}}}}
                }}
                storageBarrier();evaluate();
                for(var slot=lane;slot<status.active_count;slot+=64u) {{let c=active_indices[slot];vector[c]=direction[c];}}
                storageBarrier();apply_operator();
                for(var slot=lane;slot<status.active_count;slot+=64u) {{let c=active_indices[slot];saved[c]=State(scratch[c],Six());}}
                storageBarrier();apply_transpose();
                if lane==0u {{atomicStore(&report.fault,atomicLoad(&status.fault));}}
            }}", shader_source_for(self.arithmetic)
        );
        self.use_probe(engine, source, "transpose_probe");
    }
    #[cfg(test)]
    fn use_probe(&mut self, engine: &Engine, source: String, entry: &str) {
        self.scratch_soa = source.contains("var<storage,read_write> block_words:array<f32>;");
        let device = engine.device().unwrap();
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(entry),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(entry),
            bind_group_layouts: &[Some(&self.pipeline.get_bind_group_layout(0))],
            immediate_size: 0,
        });
        self.pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry),
            layout: Some(&layout),
            module: &module,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: engine.pipeline_cache(),
        });
    }
    #[cfg(test)]
    pub(crate) fn linear_probe_buffers(&self) -> [&wgpu::Buffer; 2] {
        assert!(
            !self.scratch_soa,
            "use logical_scratch_values for SoA diagnostics"
        );
        [&self.scratch[0], &self.scratch[5]]
    }
    #[cfg(test)]
    pub(crate) fn direction_buffer(&self) -> &wgpu::Buffer {
        &self.scratch[5]
    }
    #[cfg(test)]
    pub(crate) fn logical_scratch_values(&self, engine: &Engine, index: usize) -> Vec<f32> {
        assert!(matches!(index, 0 | 1));
        let bytes =
            crate::soil_shared_wall_friction_gpu::capture::read(engine, &self.scratch[index]);
        let values: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
            .collect();
        #[cfg(all(feature = "protected-newton", not(target_arch = "wasm32")))]
        if self.scratch_soa {
            return soa_scratch::logical_values(&values, index == 1);
        }
        assert!(!self.scratch_soa);
        values
    }
    #[cfg(test)]
    pub(crate) fn set_iteration_prefix_probe(&self, engine: &Engine, limit: u32) {
        assert!(limit <= 64);
        engine
            .queue()
            .unwrap()
            .write_buffer(&self.scratch[7], 16, &limit.to_le_bytes());
    }
    #[cfg(test)]
    pub(crate) fn precision_local_buffer(&self) -> &wgpu::Buffer {
        &self.scratch[15]
    }
    #[cfg(test)]
    pub(crate) fn basis_buffer(&self) -> &wgpu::Buffer {
        &self.scratch[2]
    }
    #[cfg(test)]
    pub(crate) fn inverse_buffer(&self) -> &wgpu::Buffer {
        assert!(
            !self.scratch_soa,
            "use logical_scratch_values for SoA diagnostics"
        );
        &self.scratch[1]
    }
    #[cfg(test)]
    pub(crate) fn transpose_probe_output(&self) -> &wgpu::Buffer {
        &self.scratch[4]
    }
    #[cfg(test)]
    pub(crate) fn forward_probe_output(&self) -> &wgpu::Buffer {
        &self.scratch[10]
    }
    #[cfg(test)]
    pub(crate) fn scratch_buffers(&self) -> impl Iterator<Item = &wgpu::Buffer> {
        self.scratch.iter().enumerate().filter_map(|(i, b)| {
            if matches!(i, 7 | 9 | 11 | 14) {
                None
            } else {
                Some(b)
            }
        })
    }

    #[cfg(test)]
    pub(crate) fn shares_program_with(&self, other: &Self) -> bool {
        self.pipeline == other.pipeline
    }
}
