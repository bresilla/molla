//! Bounded shared-machine block sweeps with coupled wall/Coulomb certification.

use molla_compute::Engine;
use molla_core::{Error, Result};

pub const SHARED_WALL_INPUT: u32 = 1;
pub const SHARED_WALL_MASS: u32 = 2;
pub const SHARED_WALL_NONFINITE: u32 = 4;
pub const SHARED_WALL_SWEEPS: u32 = 16;
pub const SHARED_WALL_LOCAL: u32 = 64;
pub(crate) const SHARED_WALL_WORKGROUP_BYTES: u32 = 848;

#[cfg(test)]
pub(crate) fn redistribution_selected() -> bool {
    [
        "SOIL_WALL_TEST_REPRESENTED_RATES",
        "SOIL_WALL_TEST_COUPLED_ROUNDING",
        "SOIL_WALL_TEST_REDISTRIBUTE",
    ]
    .iter()
    .all(|key| std::env::var_os(key).is_some())
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SoilSharedWallContact {
    /// Free world velocity and nonnegative node mass; zero mass is inactive.
    pub soil_mass: [f32; 4],
    /// Unit body-to-soil normal and nonnegative gap divided by timestep.
    pub normal_gap: [f32; 4],
    /// Nonnegative Coulomb coefficient followed by zero padding.
    pub friction: [f32; 4],
    /// Two bits per XYZ axis: 0 absent, 1 lower, 2 upper; remaining words zero.
    pub boundary: [u32; 4],
}

pub struct SoilSharedWallInputs<'a> {
    /// Symmetric positive-definite generalized mass, row-major.
    pub mass: &'a wgpu::Buffer,
    pub free_rates: &'a wgpu::Buffer,
    /// Contact-major world point Jacobian columns, XYZ plus zero padding.
    pub point_rows: &'a wgpu::Buffer,
    /// Distinct soil nodes, each with one machine contact.
    pub contacts: &'a wgpu::Buffer,
}

#[derive(Clone, Copy, Debug)]
pub struct SoilSharedWallCapacity {
    pub dofs: u32,
    pub contacts: u32,
    pub max_sweeps: u32,
    pub local_iterations: u32,
    pub max_owned_bytes: u64,
}

/// Shared contact solve with interleaved [high, low] generalized-rate inputs and outputs.
/// This component does not enable compensated rates in the rigid runtime.
/// Contact preparation requires positive-definite three-axis point mobility.
pub struct ExperimentalSoilPairedWallFrictionGpu {
    inner: ExperimentalSoilSharedWallFrictionGpu,
}

impl ExperimentalSoilPairedWallFrictionGpu {
    pub(crate) fn history_bytes(contacts: u32) -> u64 {
        336 * u64::from(contacts.max(1)) + full_state_history_bytes(contacts)
    }
    pub(crate) fn status_bytes() -> u64 {
        432 + full_state_status_bytes()
    }
    pub(crate) fn into_inner(self) -> ExperimentalSoilSharedWallFrictionGpu {
        self.inner
    }
    pub fn required_bytes(capacity: SoilSharedWallCapacity) -> Result<u64> {
        Ok(
            ExperimentalSoilSharedWallFrictionGpu::required_bytes(capacity)?
                + 4 * u64::from(capacity.dofs)
                + full_state_history_bytes(capacity.contacts)
                + full_state_status_bytes(),
        )
    }

    /// `input.free_rates` contains eight bytes per DOF, ordered high then low.
    pub fn new(
        engine: &Engine,
        input: SoilSharedWallInputs<'_>,
        capacity: SoilSharedWallCapacity,
    ) -> Result<Self> {
        Ok(Self {
            inner: ExperimentalSoilSharedWallFrictionGpu::with_rate_preparation(
                engine,
                input,
                capacity,
                [true, true, true, false, false],
                [true, true, true, false, true],
                0,
                true,
                true,
            )?,
        })
    }

    /// Interleaved f32 high/low pairs, eight bytes per generalized rate.
    pub fn rates_buffer(&self) -> &wgpu::Buffer {
        &self.inner.rates
    }
    pub fn output_buffer(&self) -> &wgpu::Buffer {
        self.inner.output_buffer()
    }
    pub fn status_buffer(&self) -> &wgpu::Buffer {
        self.inner.status_buffer()
    }
    pub fn owned_buffer_bytes(&self) -> u64 {
        self.inner.owned_buffer_bytes()
    }
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        self.inner.encode(encoder);
    }
}

/// Sequential contact blocks with parallel preparation and shared-rate reductions.
/// Only zero-status outputs are consumable; callers clear sticky status for reuse.
/// Point response must be SPD. No global convergence or throughput guarantee.
pub struct ExperimentalSoilSharedWallFrictionGpu {
    pipelines: [wgpu::ComputePipeline; 3],
    prepare_cache: Option<wgpu::ComputePipeline>,
    group: wgpu::BindGroup,
    rates: wgpu::Buffer,
    conditional_inputs: wgpu::Buffer,
    outputs: wgpu::Buffer,
    status: wgpu::Buffer,
    contacts: u32,
    solve_dispatches: u32,
    bytes: u64,
    capture_inputs: [wgpu::Buffer; 4],
    capture_capacity: SoilSharedWallCapacity,
    paired_rates: bool,
    capture_storage: [wgpu::Buffer; 4],
    #[cfg(test)]
    capture: Option<capture::SharedWallCapture>,
    #[cfg(test)]
    trace: Option<capture::SharedWallTrace>,
}

fn invalid() -> Error {
    Error::Solver("shared wall friction: invalid capacity, usage or device limit".into())
}

fn full_state_history_bytes(contacts: u32) -> u64 {
    if full_state::selected() {
        return 64 * full_state::depth() * u64::from(contacts.max(1));
    }
    0
}

fn full_state_status_bytes() -> u64 {
    if full_state::depth() == 2 {
        return 128;
    }
    0
}

#[path = "soil_shared_wall_full_state.rs"]
mod full_state;

/// Materializes each f32 rounding result through the supplied runtime-zero word.
pub(crate) fn register_rounding_source(source: &str, key: &str) -> String {
    let reference = "    atomicStore(&arithmetic_round[rounding_slot],bitcast<u32>(value));\n    return bitcast<f32>(atomicAdd(&arithmetic_round[rounding_slot],0u));";
    assert_eq!(source.matches(reference).count(), 1);
    source.replace(
        reference,
        &format!("    return bitcast<f32>(bitcast<u32>(value) ^ ({key}));"),
    )
}

pub(crate) fn unrolled_direction_source(source: &str) -> String {
    let start = source.find("fn direction(").unwrap();
    let end = source.find("fn small_physical_correction(").unwrap();
    let mut body = String::from(
        "fn direction(f:array<f32,6>)->array<f32,6> {\n    var rhs:array<f32,6>;\n    linear_ok=true;\n",
    );
    for i in 0..6 {
        body.push_str(&format!("    rhs[{i}]=-f[{i}];\n"));
    }
    for k in 0..6 {
        body.push_str(&format!(
            "    {{ var pivot={k}u; var largest=abs(jac[{k}][{k}]);\n"
        ));
        for i in k + 1..6 {
            body.push_str(&format!(
                "    if abs(jac[{i}][{k}])>largest {{ largest=abs(jac[{i}][{k}]); pivot={i}u; }}\n"
            ));
        }
        body.push_str("    if !finite(largest) || largest==0.0 { linear_ok=false; return rhs; }\n");
        for i in k + 1..6 {
            body.push_str(&format!("    if pivot=={i}u {{\n"));
            for j in k..6 {
                body.push_str(&format!(
                    "    {{ let t=jac[{k}][{j}]; jac[{k}][{j}]=jac[{i}][{j}]; jac[{i}][{j}]=t; }}\n"
                ));
            }
            body.push_str(&format!(
                "    let t=rhs[{k}]; rhs[{k}]=rhs[{i}]; rhs[{i}]=t; }}\n"
            ));
        }
        for i in k + 1..6 {
            body.push_str(&format!("    {{ let factor=jac[{i}][{k}]/jac[{k}][{k}];\n"));
            for j in k + 1..6 {
                body.push_str(&format!(
                    "    jac[{i}][{j}]=fma(-factor,jac[{k}][{j}],jac[{i}][{j}]);\n"
                ));
            }
            body.push_str(&format!(
                "    rhs[{i}]=fma(-factor,rhs[{k}],rhs[{i}]); }}\n"
            ));
        }
        body.push_str("    }\n");
    }
    for i in (0..6).rev() {
        for j in i + 1..6 {
            body.push_str(&format!(
                "    rhs[{i}]=fma(-jac[{i}][{j}],rhs[{j}],rhs[{i}]);\n"
            ));
        }
        body.push_str(&format!(
            "    rhs[{i}]/=jac[{i}][{i}];\n    if !finite(rhs[{i}]) {{ linear_ok=false; }}\n"
        ));
    }
    body.push_str("    return rhs;\n}\n");
    let mut candidate = source.to_owned();
    candidate.replace_range(start..end, &body);
    candidate
}

fn ordered_contact_source(source: &str) -> String {
    let mut candidate = source
        .replace("scale:array<f32,6> }", "scale:array<f32,6>, active_index:u32 }")
        .replace("for (var c=0u;c<shared_params.contacts;c++) {", "for (var slot=0u;slot<shared_status.control.y;slot++) { let c=shared_history[slot].active_index;")
        .replace("for (var c=coop_lane;c<shared_params.contacts;c+=64u) {", "for (var slot=coop_lane;slot<shared_status.control.y;slot+=64u) { let c=shared_history[slot].active_index;")
        .replace("        coop_ready=0u;", "        ordered_total=shared_status.control.y;\n        coop_ready=0u;");
    let begin = candidate.find("fn coop_natural_merit()").unwrap();
    let end = candidate.find("fn coop_accelerate(").unwrap();
    candidate.replace_range(
        begin..end,
        include_str!("../shaders/soil_shared_wall_ordered_merit.wgsl"),
    );
    let reset = "    shared_status.control=vec4<u32>(0u);";
    assert_eq!(candidate.matches(reset).count(), 1);
    candidate = candidate.replace(reset, &format!("{reset}\n    for (var c=0u;c<shared_params.contacts;c++) {{\n        if shared_contacts[c].soil_mass.w>0.0 {{\n            shared_history[shared_status.control.y].active_index=c;\n            shared_status.control.y+=1u;\n        }}\n    }}"));
    candidate
}

fn chunked_contact_source(source: &str, chunk: u32) -> String {
    assert!((1..=64).contains(&chunk));
    let mut candidate = source.replace(
        "control:vec4<u32> }",
        "control:vec4<u32>, sweep_start:array<vec2<f32>,32> }",
    );
    let begin = candidate.find("fn coop_local_sweep()").unwrap();
    let end = candidate[begin..].find("fn coop_finish()").unwrap() + begin;
    let local = candidate[begin..end].replace(
        "var slot=0u;slot<shared_status.control.y;slot++",
        &format!("var slot=shared_status.control.z;slot<min(shared_status.control.z+{chunk}u,shared_status.control.y);slot++"),
    );
    assert_ne!(local, candidate[begin..end]);
    candidate.replace_range(begin..end, &local);
    let init = "            coop_start=shared_q;\n            shared_status.sweeps+=1u;";
    assert_eq!(candidate.matches(init).count(), 1);
    candidate = candidate.replace(init, "            if shared_status.control.z==0u {\n                shared_status.sweep_start=shared_q;\n                shared_status.sweeps+=1u;\n            }\n            coop_start=shared_status.sweep_start;");
    let ready = "            if atomicLoad(&shared_status.fault)==0u { coop_ready=1u; }";
    assert_eq!(candidate.matches(ready).count(), 1);
    candidate.replace(ready, &format!("            if atomicLoad(&shared_status.fault)==0u {{\n                shared_status.control.z=min(shared_status.control.z+{chunk}u,shared_status.control.y);\n                if shared_status.control.z==shared_status.control.y {{\n                    shared_status.control.z=0u;\n                    coop_ready=1u;\n                }} else {{\n                    for (var i=0u;i<n;i++) {{ shared_status.rates[i]=shared_q[i]; }}\n                }}\n            }}"))
}

fn local_cooperative_source(source: &str) -> String {
    let call = "            coop_local_sweep();\n";
    let ready = "    if workgroupUniformLoad(&coop_ready)==0u { return; }\n    storageBarrier();";
    assert_eq!(source.matches(call).count(), 1);
    assert_eq!(source.matches(ready).count(), 1);
    let candidate = source
        .replace("fn coop_local_sweep()", "fn coop_local_reference_sweep()")
        .replace(call, "")
        .replace(ready, "    if workgroupUniformLoad(&coop_ready)==0u { return; }\n    coop_local_sweep();\n    if workgroupUniformLoad(&coop_ready)==0u { return; }\n    storageBarrier();");
    format!(
        "{candidate}\n{}",
        include_str!("../shaders/soil_shared_wall_local_cooperative.wgsl")
    )
}

fn cooperative_source(source: &str) -> String {
    let merit_start = source.find("fn shared_natural_merit()").unwrap();
    let merit = &source[merit_start..];
    let term = &merit[merit.find("        let node=shared_contacts[c];").unwrap()
        ..merit
            .find("        value=two_sum(value.x,value.y+sum);")
            .unwrap()];
    let coefficient_start = source
        .find("    let n=shared_params.width; var current:")
        .unwrap();
    let coefficient_end = source
        .find("    let baseline=shared_natural_merit();")
        .unwrap();
    let solve_start = source.find("fn shared_solve(").unwrap();
    let solve = &source[solve_start..];
    let local_start = solve
        .find("        for (var c=0u;c<shared_params.contacts;c++) {")
        .unwrap();
    let local_end = solve.find("        shared_rebuild_rates();").unwrap();
    let finish_start = solve.find("        if shared_certify() {").unwrap();
    let finish_end = solve
        .find("    if shared_status.sweeps>=shared_params.sweeps")
        .unwrap();
    let save_end = solve
        .find("        shared_status.previous_f[i]=shared_previous_f[i];\n    }\n}")
        .unwrap()
        + "        shared_status.previous_f[i]=shared_previous_f[i];\n    }\n}".len();
    let candidate = source
        .replace("var<private> shared_q:", "var<workgroup> shared_q:")
        .replace("fn shared_solve(", "fn shared_reference_solve(");
    format!(
        "{candidate}\nfn coop_contact_merit(c:u32)->f32 {{ {} return sum; }}\nfn coop_coefficient(start:array<vec2<f32>,32>,has_previous:bool)->f32 {{ {} shared_previous_f=current; return coefficient; }}\nfn coop_local_sweep() {{ let n=shared_params.width; {} }}\nfn coop_finish() {{ let n=shared_params.width; {} {}\n{}",
        term.replace("if m==0.0 { continue; }", "if m==0.0 { return 0.0; }"),
        &source[coefficient_start..coefficient_end],
        &solve[local_start..local_end],
        solve[finish_start..finish_end]
            .trim_end()
            .strip_suffix('}')
            .unwrap(),
        &solve[finish_end..save_end],
        include_str!("../shaders/soil_shared_wall_cooperative.wgsl")
    )
}

fn cached_setup_source(source: &str) -> String {
    let anchor = "fn solve_block(input:Input,iterations:u32) {";
    let begin = source.find(anchor).unwrap() + anchor.len();
    let end = source[begin..]
        .find("    var x:array<f32,6>; var tail:array<f32,6>;")
        .unwrap()
        + begin;
    let setup = &source[begin..end];
    let mut candidate = source.to_owned();
    candidate.replace_range(
        begin..end,
        include_str!("../shaders/soil_wall_cached_setup.wgsl"),
    );
    let history = "struct SharedHistory { previous_k:vec4<f32>, previous_w:vec4<f32>, baseline_k:vec4<f32>, baseline_w:vec4<f32> }";
    assert_eq!(candidate.matches(history).count(), 1);
    candidate = candidate.replace(history, "struct SharedHistory { previous_k:vec4<f32>, previous_w:vec4<f32>, baseline_k:vec4<f32>, baseline_w:vec4<f32>, basis:array<vec4<f32>,6>, metric:array<array<f32,6>,6>, scale:array<f32,6> }");
    let prepared = "        vec4<f32>(0.0,0.0,0.0,node.friction.x),node.soil_mass,node.normal_gap,node.boundary);";
    assert_eq!(candidate.matches(prepared).count(), 1);
    candidate = candidate.replace(prepared, &format!("{prepared}\n    shared_cache_prepare(shared_inputs[c],c);\n    if block_output.status.x!=0u {{ shared_fault(64u,c+1u,block_output.status.x); }}"));
    candidate = candidate.replace(
        "            solve_block(input,shared_params.iterations);",
        "            cache_index=c;\n            solve_block(input,shared_params.iterations);",
    );
    format!(
        "{candidate}\nvar<private> cache_index:u32;\nfn shared_cache_prepare(input:Input,c:u32) {{{setup}\n    for (var i=0u;i<6u;i++) {{ shared_history[c].basis[i]=vec4<f32>(axes[i],0.0); }}\n    shared_history[c].metric=h;\n    shared_history[c].scale=rho;\n    block_output.status.x=0u;\n}}\n"
    )
}

fn split_preparation_source(source: &str) -> String {
    let setup = "    shared_cache_prepare(shared_inputs[c],c);\n    if block_output.status.x!=0u { shared_fault(64u,c+1u,block_output.status.x); }";
    assert_eq!(source.matches(setup).count(), 1);
    format!(
        "{}\n@compute @workgroup_size(64)\nfn shared_prepare_cache(@builtin(global_invocation_id) id:vec3<u32>,@builtin(local_invocation_index) lane:u32) {{\n    rounding_slot=lane;\n    let c=id.x;\n    if c>=shared_params.contacts || atomicLoad(&shared_status.fault)!=0u {{ return; }}\n    if shared_contacts[c].soil_mass.w==0.0 {{ return; }}\n{setup}\n}}\n",
        source.replace(setup, "")
    )
}

#[cfg(test)]
fn redistributed_source(source: &str) -> String {
    let mut candidate = source.to_owned();
    for (anchor, replacement) in [
        (
            "active_index:u32 }",
            "active_index:u32, correction:vec4<f32> }",
        ),
        (
            "if !represented_coupled_factor() { return false; }",
            "if !represented_redistribute() { return false; }",
        ),
        (
            "fn represented_coupled_delta(c:u32)->vec3<f32>",
            "fn redistributed_raw_delta(c:u32)->vec3<f32>",
        ),
        (
            "represented_rhs[i]=value.x-clamp(value.x,-0.5e-6*value.y,0.5e-6*value.y);",
            "let remaining=two_sum(value.x-clamp(value.x,-0.5e-6*value.y,0.5e-6*value.y),represented_fixed_momentum(i)); represented_rhs[i]=rounded(remaining.x+remaining.y);",
        ),
        ("return rank>0u;", "return true;"),
    ] {
        assert_eq!(candidate.matches(anchor).count(), 1);
        candidate = candidate.replace(anchor, replacement);
    }
    let start = candidate.find("fn represented_coupled_factor()").unwrap();
    let end = candidate.find("fn redistributed_raw_delta(").unwrap();
    let body = candidate[start..end].replace(
        "if !represented_active(c) { continue; }",
        "if !represented_active(c) || shared_history[c].correction.w!=0.0 { continue; }",
    );
    candidate.replace_range(start..end, &body);
    format!(
        "{candidate}\n{}",
        include_str!("../shaders/soil_shared_wall_redistribution.wgsl")
    )
}

#[cfg(test)]
fn bounded_dual_source(source: &str) -> String {
    let anchor = "if !represented_redistribute() { return false; }";
    assert_eq!(source.matches(anchor).count(), 1);
    let gate = "scale+=dot(abs(shared_rows[c*n+i].xyz),abs(k));";
    assert_eq!(source.matches(gate).count(), 1);
    let candidate = source
        .replace(anchor, "if !represented_bounded_dual() { return false; }")
        .replace(gate, "scale+=abs(dot(shared_rows[c*n+i].xyz,k));")
        .replace(
            "let value=represented_momentum(i);",
            "let value=dual_momentum(i);",
        );
    format!(
        "{}\n{}",
        candidate,
        include_str!("../shaders/soil_shared_wall_bounded_dual.wgsl")
    )
}

fn stable_merit_source(source: &str) -> String {
    let mut candidate = source.to_owned();
    for (anchor, replacement) in [
        (
            "lambda-max(0.0,lambda-rho*(dot(normal,relative)+node.normal_gap.w))",
            "min(lambda,rho*(dot(normal,relative)+node.normal_gap.w))",
        ),
        (
            "beta-max(0.0,beta-m*sign*soil[axis])",
            "min(beta,m*sign*soil[axis])",
        ),
        (
            "let ft=(tangent-projected)/scale;",
            "var ft=(rho/scale)*slip; if radius>cap { ft=(tangent-projected)/scale; }",
        ),
    ] {
        assert_eq!(candidate.matches(anchor).count(), 1);
        candidate = candidate.replace(anchor, replacement);
    }
    candidate
}

#[cfg(test)]
fn backtracked_acceleration_source(source: &str) -> String {
    let anchor = "        let candidate=coop_natural_merit();";
    assert_eq!(source.matches(anchor).count(), 1);
    source.replace(
        anchor,
        include_str!("../shaders/soil_shared_wall_backtrack.wgsl"),
    )
}

fn coverage_certificate_source(source: &str) -> String {
    let mut candidate = shared_friction_accuracy_source(source);
    for (anchor, strict) in [
        (
            "separation< -1e-6*vs",
            "separation< -1e-6*max(1.0,max(abs(dot(normal,soil)),max(abs(dot(normal,machine)),node.normal_gap.w)))",
        ),
        (
            "separation>=-1e-5*velocity_scale",
            "separation>=-1e-6*max(1.0,max(abs(dot(axes[0],soil)),max(abs(dot(axes[0],machine)),input.normal_gap.w)))",
        ),
    ] {
        if !candidate.contains(anchor) {
            assert_eq!(candidate.matches(strict).count(), 1);
        } else {
            assert_eq!(candidate.matches(anchor).count(), 1);
            candidate = candidate.replace(anchor, strict);
        }
    }
    candidate
}

#[cfg(test)]
fn coupled_represented_source(source: &str) -> String {
    let source = coverage_certificate_source(source);
    let source = represented_rate_source(&source);
    let start = source.find("fn shared_certify()->bool {").unwrap();
    let mut body = source[start..].to_owned();
    let begin = body.find("    var denominator=vec2<f32>(0.0);").unwrap();
    let end = body
        .find("    for (var i=0u;i<n;i++) {\n        represented_start[i]")
        .unwrap();
    body.replace_range(
        begin..end,
        "    if !represented_coupled_factor() { return false; }\n",
    );
    body = body.replace("    var safe=true;", "    let projection_step=represented_coupled_step();\n    var safe=finite(projection_step) && projection_step>0.0;");
    body = body.replace("        let delta=-(mass*residual/weight)*shared_rows[c*n+selected].xyz;", "        if !safe { break; }\n        let delta=projection_step*represented_coupled_delta(c);");
    format!(
        "{}{}\n{}",
        &source[..start],
        body,
        include_str!("../shaders/soil_shared_wall_coupled_rounding.wgsl")
    )
}

#[cfg(test)]
fn represented_rate_source(source: &str) -> String {
    let anchor = "fn shared_certify()->bool {";
    assert_eq!(source.matches(anchor).count(), 1);
    format!(
        "{}\n{}",
        source.replace(anchor, "fn shared_certify_raw()->bool {"),
        include_str!("../shaders/soil_shared_wall_represented.wgsl")
    )
}

fn certificate_trace_source(source: &str) -> String {
    if source.contains("fn shared_certify()->bool {\n    shared_status.pad=0u;") {
        return source.to_owned();
    }
    let start = source.find("fn shared_certify()->bool {").unwrap();
    let end = start + source[start..].find("\n@compute").unwrap();
    let mut body = source[start..end].replace(
        "fn shared_certify()->bool {",
        "fn shared_certify()->bool {\n    shared_status.pad=0u;",
    );
    for (anchor, code, index) in [
        (
            "if abs(momentum.x+momentum.y)>1e-6*scale { return false; }",
            1,
            "i+1u",
        ),
        (
            "|| length(tangent)>cap+1e-6*ps { return false; }",
            2,
            "c+1u",
        ),
        (
            "if length(slip)>1e-6*vs && length(tangent+cap*normalize(slip))>1e-6*ps { return false; }",
            3,
            "c+1u",
        ),
        ("if w[axis]!=0.0 { return false; }", 4, "c+1u"),
        (
            "if sign*w[axis]< -1e-6*ps || sign*soil[axis]< -1e-6*vs || abs(w[axis]*soil[axis])>1e-6*ps*vs { return false; }",
            5,
            "c+1u",
        ),
        (
            "|| length(momentum_error)>1e-6*momentum_scale { return false; }",
            7,
            "c+1u",
        ),
    ] {
        assert_eq!(body.matches(anchor).count(), 1);
        body = body.replace(
            anchor,
            &anchor.replace(
                "return false;",
                &format!("shared_status.pad={code}u | (({index})<<8u); return false;"),
            ),
        );
    }
    let paired_gate = "if lambda< -1e-6*ps || !shared_normal_covered(c,soil) || abs(lambda*separation)>1e-6*ps*vs\n            || length(tangent)>cap+1e-6*ps { shared_status.pad=2u | ((c+1u)<<8u); return false; }";
    if body.contains("!shared_normal_covered(c,soil)") {
        assert_eq!(body.matches(paired_gate).count(), 1);
        body = body.replace(paired_gate, "if lambda< -1e-6*ps { shared_status.pad=8u | ((c+1u)<<8u); return false; }\n        if !shared_normal_covered(c,soil) { shared_status.pad=9u | ((c+1u)<<8u); return false; }\n        if abs(lambda*separation)>1e-6*ps*vs { shared_status.pad=10u | ((c+1u)<<8u); return false; }\n        if length(tangent)>cap+1e-6*ps { shared_status.pad=11u | ((c+1u)<<8u); return false; }");
    }
    body = body.replace("    return energy.x+energy.y<=1e-6*energy_scale;", "    let accepted=energy.x+energy.y<=1e-6*energy_scale;\n    if !accepted { shared_status.pad=6u; }\n    return accepted;");
    let mut candidate = source.to_owned();
    candidate.replace_range(start..end, &body);
    candidate
}

fn storage_setup_source(source: &str) -> String {
    let function = source
        .find("fn shared_cache_prepare(input:Input,c:u32) {")
        .unwrap();
    let start = function
        + source[function..]
            .find("    axes[0]=normalize(input.normal_gap.xyz);")
            .unwrap();
    let tail = "    block_output.status.x=0u;\n}";
    let end = start + source[start..].find(tail).unwrap() + tail.len() - 1;
    let mut candidate = source.to_owned();
    candidate.replace_range(
        start..end,
        include_str!("../shaders/soil_wall_storage_setup.wgsl"),
    );
    candidate
}

fn storage_response_source(source: &str) -> String {
    let start = source
        .find("    var response:array<vec3<f32>,32>;")
        .unwrap();
    let end = source[start..].find("    shared_inputs[c]=Input(").unwrap() + start;
    let response = source[start..end]
        .replace("    var response:array<vec3<f32>,32>;\n", "")
        .replace("response[j]", "shared_response[c*n+j].xyz")
        .replace("response[d][j]", "shared_response[c*n+d][j]")
        .replace(
            "response[i]=value/shared_lower[i*n+i];",
            "shared_response[c*n+i]=vec4<f32>(value/shared_lower[i*n+i],0.0);",
        )
        .replace("response[i]", "shared_response[c*n+i].xyz")
        .replace(
            "        shared_response[c*n+i]=vec4<f32>(shared_response[c*n+i].xyz,0.0);\n",
            "",
        );
    let mut candidate = source.to_owned();
    candidate.replace_range(start..end, &response);
    candidate
}

pub(crate) fn shared_friction_accuracy_source(source: &str) -> String {
    let original = "const LOCAL_FRICTION_TOLERANCE:f32=1e-5;";
    let shared = "const LOCAL_FRICTION_TOLERANCE:f32=1e-6;";
    if source.contains(shared) {
        assert_eq!(source.matches(shared).count(), 1);
        source.to_owned()
    } else {
        assert_eq!(source.matches(original).count(), 1);
        source.replace(original, shared)
    }
}

#[cfg(test)]
fn local_publication_refinement_source(source: &str, stationary: bool) -> String {
    if source.contains("var publication_trial:array<f32,6>;") {
        return source.to_owned();
    }
    let anchor = "if old_merit<=1e-12 && small_physical_correction(x,delta) { publish(input,x,tail,a,sqrt(old_merit)); if block_output.status.x!=32u { return; } }";
    assert_eq!(source.matches(anchor).count(), 1);
    let candidate = source.replace(anchor, r#"if old_merit<=1e-12 && small_physical_correction(x,delta) {
            if (shared_status.pad&255u)==9u && shared_status.control.w==0u {
                var stationary=true;
                for (var i=0u;i<shared_params.width;i++) {
                    let speed=shared_status.rates[i].x+shared_status.rates[i].y;
                    stationary=stationary && abs(shared_status.previous_f[i])<=1e-6*max(1.0,abs(speed));
                }
                if stationary { shared_status.control.w=1u; }
            }
            if shared_status.control.w!=0u {
            var publication_trial:array<f32,6>; var publication_tail:array<f32,6>;
            for (var i=0u;i<6u;i++) {
                let sum=two_sum(x[i],delta[i]); let next=two_sum(sum.x,sum.y+tail[i]);
                publication_trial[i]=next.x; publication_tail[i]=next.y;
            }
            let refined_merit=merit(residual(publication_trial,publication_tail,false));
            if finite(refined_merit) && refined_merit<old_merit && refined_merit<=(1.0-2e-4)*old_merit {
                publish(input,publication_trial,publication_tail,a,sqrt(refined_merit));
                if block_output.status.x==0u { return; }
            }
            }
            publish(input,x,tail,a,sqrt(old_merit)); if block_output.status.x!=32u { return; }
        }"#);
    if stationary {
        candidate
    } else {
        candidate.replace("if shared_status.control.w!=0u {", "if true {")
    }
}

#[cfg(test)]
fn paired_merit_diagnostic_source(source: &str, soil: bool) -> String {
    let mut source = source.to_owned();
    let merit_normal = "min(lambda,rho*(dot(normal,relative)+node.normal_gap.w))";
    let original_normal = "lambda-max(0.0,lambda-rho*(dot(normal,relative)+node.normal_gap.w))";
    assert!(source.contains(merit_normal) || source.contains(original_normal));
    source = source.replace(
        original_normal,
        "min(lambda,rho*shared_normal_residual(c,soil).x)",
    );
    source = source.replace(
        merit_normal,
        "min(lambda,rho*shared_normal_residual(c,soil).x)",
    );
    if soil {
        assert!(source.contains("let soil=node.soil_mass.xyz+(k+w)/m;"));
        source = source.replace(
            "let soil=node.soil_mass.xyz+(k+w)/m;",
            "let soil=shared_merit_soil_velocity(c,k,w);",
        );
    }
    source
}

fn paired_rate_source(source: &str) -> String {
    let mut source = source.to_owned();
    for (anchor, replacement) in [
        ("shared_free:array<f32>", "shared_free:array<vec2<f32>>"),
        ("shared_rates:array<f32>", "shared_rates:array<vec2<f32>>"),
        (
            "if !finite(shared_free[i])",
            "if !all(abs(shared_free[i])<=vec2<f32>(3.402823e38)) || !finite(shared_free[i].x+shared_free[i].y)",
        ),
        ("vec2<f32>(shared_free[i],0.0)", "shared_free[i]"),
        (
            "shared_rates[i]=rounded(shared_q[i].x+shared_q[i].y)",
            "shared_rates[i]=two_sum(shared_q[i].x,shared_q[i].y)",
        ),
    ] {
        assert!(source.contains(anchor), "paired rate anchor: {anchor}");
        source = source.replace(anchor, replacement);
    }
    let start = source.find("fn shared_certify()->bool {").unwrap();
    let begin = start
        + source[start..]
            .find("    for (var i=0u;i<n;i++) {")
            .unwrap();
    let node = begin + source[begin..].find("        let node=shared_contacts[c]; if node.soil_mass.w==0.0 { continue; }\n        let output=shared_outputs[c];").unwrap();
    let end = begin + source[begin..node].rfind("    for (").unwrap();
    source.replace_range(
        begin..end,
        include_str!("../shaders/soil_shared_wall_paired_certificate.wgsl"),
    );
    let anchor = "separation< -1e-6*max(1.0,max(abs(dot(normal,soil)),max(abs(dot(normal,machine)),node.normal_gap.w)))";
    assert_eq!(source.matches(anchor).count(), 1);
    source = source.replace(anchor, "!shared_normal_covered(c,soil)");
    format!(
        "{}\n{}\n{}\n{source}",
        crate::soil_rate_encoding::arithmetic(1),
        include_str!("../shaders/soil_normal_rate_contract.wgsl"),
        include_str!("../shaders/soil_shared_wall_normal_contract.wgsl")
    )
}

impl ExperimentalSoilSharedWallFrictionGpu {
    pub fn required_bytes(capacity: SoilSharedWallCapacity) -> Result<u64> {
        if !(1..=32).contains(&capacity.dofs)
            || capacity.contacts > 16384
            || !(1..=256).contains(&capacity.max_sweeps)
            || !(1..=64).contains(&capacity.local_iterations)
        {
            return Err(invalid());
        }
        let n = u64::from(capacity.dofs);
        let c = u64::from(capacity.contacts.max(1));
        Ok((528 + 16 * n) * c + 4 * n * n + 4 * n + 448)
    }

    pub fn new(
        engine: &Engine,
        input: SoilSharedWallInputs<'_>,
        capacity: SoilSharedWallCapacity,
    ) -> Result<Self> {
        #[cfg(test)]
        if std::env::var_os("SOIL_WALL_TEST_SPLIT_PREPARE").is_some() {
            eprintln!("test-only shared-wall split preparation selected");
            return Self::with_preparation(
                engine,
                input,
                capacity,
                [true, true, true, false, false],
                [true, true, true, false, true],
                0,
                true,
            );
        }
        Self::with_ordered(
            engine,
            input,
            capacity,
            [true, true, true, false, false],
            [true, true, true, false, true],
        )
    }

    fn with_ordered(
        engine: &Engine,
        input: SoilSharedWallInputs<'_>,
        capacity: SoilSharedWallCapacity,
        options: [bool; 5],
        variants: [bool; 5],
    ) -> Result<Self> {
        Self::with_chunks(engine, input, capacity, options, variants, 0)
    }

    fn with_chunks(
        engine: &Engine,
        input: SoilSharedWallInputs<'_>,
        capacity: SoilSharedWallCapacity,
        options: [bool; 5],
        variants: [bool; 5],
        chunk: u32,
    ) -> Result<Self> {
        Self::with_preparation(engine, input, capacity, options, variants, chunk, false)
    }

    fn with_preparation(
        engine: &Engine,
        input: SoilSharedWallInputs<'_>,
        capacity: SoilSharedWallCapacity,
        options: [bool; 5],
        variants: [bool; 5],
        chunk: u32,
        split_prepare: bool,
    ) -> Result<Self> {
        Self::with_rate_preparation(
            engine,
            input,
            capacity,
            options,
            variants,
            chunk,
            split_prepare,
            false,
        )
    }

    fn with_rate_preparation(
        engine: &Engine,
        input: SoilSharedWallInputs<'_>,
        capacity: SoilSharedWallCapacity,
        options: [bool; 5],
        variants: [bool; 5],
        chunk: u32,
        split_prepare: bool,
        paired: bool,
    ) -> Result<Self> {
        let [
            cache_setup,
            unroll_linear,
            cooperative,
            local_parallel,
            ordered,
        ] = variants;
        let [
            accelerate,
            segmented,
            register_rounding,
            direct_wall_axes,
            warm_start,
        ] = options;
        #[cfg(test)]
        let represented = !paired
            && split_prepare
            && std::env::var_os("SOIL_WALL_TEST_REPRESENTED_RATES").is_some();
        #[cfg(not(test))]
        let represented = false;
        #[cfg(test)]
        let coupled = represented && std::env::var_os("SOIL_WALL_TEST_COUPLED_ROUNDING").is_some();
        #[cfg(not(test))]
        let coupled = false;
        #[cfg(test)]
        let redistribute = !paired && split_prepare && redistribution_selected();
        #[cfg(not(test))]
        let redistribute = false;
        #[cfg(test)]
        let bounded_dual =
            redistribute && std::env::var_os("SOIL_WALL_TEST_BOUNDED_DUAL").is_some();
        #[cfg(not(test))]
        let bounded_dual = false;
        if represented && !ordered {
            return Err(invalid());
        }
        if split_prepare && !cache_setup {
            return Err(invalid());
        }
        if chunk > 64 || (chunk != 0 && !ordered) {
            return Err(invalid());
        }
        if cache_setup && (direct_wall_axes || warm_start) {
            return Err(invalid());
        }
        if local_parallel && !cooperative {
            return Err(invalid());
        }
        if ordered && (!cooperative || local_parallel) {
            return Err(invalid());
        }
        if cooperative
            && (!segmented
                || !register_rounding
                || !cache_setup
                || !unroll_linear
                || direct_wall_axes
                || warm_start)
        {
            return Err(invalid());
        }
        engine.check_device()?;
        let continuation_bytes = if chunk == 0 { 0 } else { 256 };
        let full_status_bytes = if paired { full_state_status_bytes() } else { 0 };
        let full_state_bytes = if paired {
            full_state_history_bytes(capacity.contacts)
        } else {
            0
        };
        let redistribution_bytes = if redistribute {
            16 * u64::from(capacity.contacts.max(1))
        } else {
            0
        };
        let bytes = Self::required_bytes(capacity)?
            + continuation_bytes
            + full_state_bytes
            + full_status_bytes
            + redistribution_bytes
            + if paired {
                4 * u64::from(capacity.dofs)
            } else {
                0
            }
            - if cache_setup {
                0
            } else {
                272 * u64::from(capacity.contacts.max(1))
            };
        let device = engine.device().ok_or_else(invalid)?;
        let limits = device.limits();
        if represented
            && limits.max_compute_workgroup_storage_size
                < SHARED_WALL_WORKGROUP_BYTES
                    + if coupled { 4864 } else { 256 }
                    + if bounded_dual { 640 } else { 0 }
        {
            return Err(invalid());
        }
        let n = u64::from(capacity.dofs);
        let c = u64::from(capacity.contacts.max(1));
        if bytes > capacity.max_owned_bytes
            || limits.max_storage_buffers_per_shader_stage < 11
            || limits.max_compute_workgroup_storage_size
                < if cooperative {
                    if ordered {
                        SHARED_WALL_WORKGROUP_BYTES
                    } else if local_parallel {
                        864
                    } else {
                        832
                    }
                } else {
                    256
                }
            || limits.max_compute_workgroup_size_x < 64
            || limits.max_compute_invocations_per_workgroup < 64
            || capacity.contacts.div_ceil(64) > limits.max_compute_workgroups_per_dimension
        {
            return Err(invalid());
        }
        let owned_sizes = [
            4 * n * n,
            16 * c * n,
            112 * c,
            80 * c,
            if paired { 8 * n } else { 4 * n },
            432 + continuation_bytes + full_status_bytes,
            16,
            (if cache_setup { 336 * c } else { 64 * c }) + redistribution_bytes + full_state_bytes,
        ];
        if owned_sizes
            .iter()
            .any(|&s| s > limits.max_buffer_size || s > limits.max_storage_buffer_binding_size)
        {
            return Err(invalid());
        }
        for (buffer, required) in [
            input.mass,
            input.free_rates,
            input.point_rows,
            input.contacts,
        ]
        .into_iter()
        .zip([
            4 * n * n,
            if paired { 8 * n } else { 4 * n },
            16 * c * n,
            64 * c,
        ]) {
            if buffer.size() < required
                || buffer.size() > limits.max_storage_buffer_binding_size
                || !buffer.usage().contains(wgpu::BufferUsages::STORAGE)
            {
                return Err(invalid());
            }
        }
        let buffers = owned_sizes.map(|size| {
            molla_core::gpu_create_buffer!(
                device,
                &wgpu::BufferDescriptor {
                    label: Some("shared wall friction retained storage"),
                    size,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::UNIFORM,
                    mapped_at_creation: false,
                }
            )
        });
        molla_core::gpu_write_buffer!(
            engine.queue().unwrap(),
            &buffers[6],
            0,
            bytemuck::cast_slice(&[
                capacity.dofs,
                capacity.contacts,
                capacity.max_sweeps,
                capacity.local_iterations
            ])
        );
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shared wall friction layout"),
            entries: &(0..12)
                .map(|binding| wgpu::BindGroupLayoutEntry {
                    binding,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: if binding == 10 {
                            wgpu::BufferBindingType::Uniform
                        } else {
                            wgpu::BufferBindingType::Storage {
                                read_only: binding < 4,
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
            label: Some("shared wall friction pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let source = [
            include_str!("../shaders/soil_wall_friction_block.wgsl"),
            include_str!("../shaders/soil_shared_wall_friction.wgsl"),
        ]
        .join("\n")
        .replace(
            "const SHARED_ACCELERATE:bool=true;",
            &format!("const SHARED_ACCELERATE:bool={accelerate};"),
        )
        .replace(
            "const SHARED_SEGMENTED:bool=true;",
            &format!("const SHARED_SEGMENTED:bool={segmented};"),
        );
        let source = if paired {
            stable_merit_source(&source)
        } else {
            source
        };
        #[cfg(test)]
        let source = if !paired && std::env::var_os("SOIL_WALL_TEST_STABLE_MERIT").is_some() {
            stable_merit_source(&source)
        } else {
            source
        };
        let source = if cache_setup {
            cached_setup_source(&source)
        } else {
            source
        };
        let source = if direct_wall_axes {
            let start = source.find("fn impulse_pair(").unwrap();
            let end = source.find("fn finite(").unwrap();
            source.replace(
                &source[start..end],
                include_str!("../shaders/soil_wall_axes_candidate.wgsl"),
            )
        } else {
            source
        };
        let source = if warm_start {
            let anchor = "    var x:array<f32,6>; var tail:array<f32,6>;";
            assert_eq!(source.matches(anchor).count(), 1);
            format!("{}\nvar<private> warm_seed:Output;\n", source
                .replace(anchor, &format!("{anchor}\n{}", include_str!("../shaders/soil_wall_warm_seed.wgsl")))
                .replace("            solve_block(input,shared_params.iterations);", "            warm_seed=shared_outputs[c];\n            solve_block(input,shared_params.iterations);"))
        } else {
            source
        };
        let source = if register_rounding {
            register_rounding_source(&source, "shared_params.width >> 16u")
        } else {
            source
        };
        let source = if unroll_linear {
            unrolled_direction_source(&source)
        } else {
            source
        };
        let source = if cooperative {
            cooperative_source(&source)
        } else {
            source
        };
        let source = if local_parallel {
            local_cooperative_source(&source)
        } else {
            source
        };
        let source = if ordered {
            ordered_contact_source(&source)
        } else {
            source
        };
        let source = if chunk != 0 {
            chunked_contact_source(&source, chunk)
        } else {
            source
        };
        let source = if split_prepare {
            split_preparation_source(&source)
        } else {
            source
        };
        let source = if paired {
            certificate_trace_source(&paired_rate_source(&coverage_certificate_source(
                &storage_setup_source(&storage_response_source(&source)),
            )))
        } else {
            source
        };
        let source = if full_state_bytes != 0 {
            assert!(paired && cooperative && ordered && cache_setup && chunk == 0);
            full_state::source(&source)
        } else {
            source
        };
        #[cfg(test)]
        let source = if paired {
            match std::env::var("MOLLA_SHARED_MERIT_DIAGNOSTIC").as_deref() {
                Ok("normal") => paired_merit_diagnostic_source(&source, false),
                Ok("normal-and-soil") => paired_merit_diagnostic_source(&source, true),
                Err(_) => source,
                Ok(_) => panic!("unknown shared merit diagnostic"),
            }
        } else {
            source
        };
        #[cfg(test)]
        let source = if paired {
            match std::env::var("MOLLA_SHARED_REFINEMENT_DIAGNOSTIC").as_deref() {
                Ok("always") => local_publication_refinement_source(&source, false),
                Ok("stationary") => local_publication_refinement_source(&source, true),
                Err(_) => source,
                Ok(_) => panic!("unknown shared refinement diagnostic"),
            }
        } else {
            source
        };
        #[cfg(test)]
        let source = if !paired
            && split_prepare
            && std::env::var_os("SOIL_WALL_TEST_STORAGE_RESPONSE").is_some()
        {
            eprintln!("test-only shared-wall storage response selected");
            storage_response_source(&source)
        } else {
            source
        };
        #[cfg(test)]
        let source = if !paired
            && split_prepare
            && std::env::var_os("SOIL_WALL_TEST_STORAGE_SETUP").is_some()
        {
            eprintln!("test-only shared-wall storage setup selected");
            storage_setup_source(&source)
        } else {
            source
        };
        #[cfg(test)]
        let source = if cooperative && std::env::var_os("SOIL_WALL_TEST_BACKTRACK").is_some() {
            backtracked_acceleration_source(&source)
        } else {
            source
        };
        #[cfg(test)]
        let source = if std::env::var_os("SOIL_WALL_TEST_COVERAGE_SCALE").is_some() {
            coverage_certificate_source(&source)
        } else {
            source
        };
        #[cfg(test)]
        let source = if std::env::var_os("SOIL_WALL_TEST_CERTIFICATE_TRACE").is_some() {
            certificate_trace_source(&source)
        } else {
            source
        };
        #[cfg(test)]
        let source = if coupled {
            coupled_represented_source(&source)
        } else if represented {
            represented_rate_source(&source)
        } else {
            source
        };
        #[cfg(test)]
        let source = if redistribute {
            redistributed_source(&source)
        } else {
            source
        };
        #[cfg(test)]
        let source = if bounded_dual {
            bounded_dual_source(&source)
        } else {
            source
        };
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shared wall friction"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipelines = ["shared_factor", "shared_prepare", "shared_solve"].map(|entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        });
        let prepare_cache = split_prepare.then(|| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("shared_prepare_cache"),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some("shared_prepare_cache"),
                compilation_options: Default::default(),
                cache: None,
            })
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shared wall friction buffers"),
            layout: &layout,
            entries: &[
                input.mass,
                input.free_rates,
                input.point_rows,
                input.contacts,
            ]
            .into_iter()
            .chain(buffers.iter())
            .enumerate()
            .map(|(binding, buffer)| wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect::<Vec<_>>(),
        });
        let capture_storage = [
            buffers[0].clone(),
            buffers[1].clone(),
            buffers[6].clone(),
            buffers[7].clone(),
        ];
        let [_, _, conditional_inputs, outputs, rates, status, _, _] = buffers;
        Ok(Self {
            pipelines,
            prepare_cache,
            group,
            rates,
            conditional_inputs,
            outputs,
            status,
            contacts: capacity.contacts,
            solve_dispatches: if segmented && capacity.contacts > 0 {
                capacity.max_sweeps
                    * if chunk == 0 {
                        1
                    } else {
                        capacity.contacts.div_ceil(chunk)
                    }
            } else {
                1
            },
            bytes,
            capture_inputs: [
                input.mass.clone(),
                input.free_rates.clone(),
                input.point_rows.clone(),
                input.contacts.clone(),
            ],
            capture_capacity: capacity,
            paired_rates: paired,
            capture_storage,
            #[cfg(test)]
            capture: None,
            #[cfg(test)]
            trace: None,
        })
    }

    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        #[cfg(test)]
        if let Some(capture) = &self.capture {
            capture.encode(encoder);
            return;
        }
        #[cfg(test)]
        let traced = self
            .trace
            .as_ref()
            .map(|trace| trace.before(encoder, &self.capture_inputs, self.capture_capacity));
        for (_index, (pipeline, groups)) in self.passes().enumerate() {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &self.group, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
            molla_core::gpu_metrics::record_dispatch();
            drop(pass);
            #[cfg(test)]
            if let (Some(trace), Some(path)) = (&self.trace, &traced) {
                trace.checkpoint(
                    encoder,
                    path,
                    _index,
                    &self.status,
                    self.prepare_cache.is_some(),
                );
                if _index == 0 {
                    trace.factor_state(path, &self.capture_storage);
                }
                if _index == 1 + usize::from(self.prepare_cache.is_some()) && self.contacts != 0 {
                    trace.stop_prepared(path, self);
                }
            }
        }
        #[cfg(test)]
        if let (Some(trace), Some(path)) = (&self.trace, &traced) {
            trace.finish(path, self);
        }
    }

    fn passes(&self) -> impl Iterator<Item = (&wgpu::ComputePipeline, u32)> {
        self.pipelines
            .iter()
            .take(2)
            .zip([1, self.contacts.div_ceil(64)])
            .chain(
                self.prepare_cache
                    .iter()
                    .map(|p| (p, self.contacts.div_ceil(64))),
            )
            .chain(std::iter::repeat_n(
                (&self.pipelines[2], 1),
                self.solve_dispatches as usize,
            ))
            .filter(|(_, groups)| *groups != 0)
    }
    pub fn rates_buffer(&self) -> &wgpu::Buffer {
        &self.rates
    }
    /// Last attempted conditional inputs in `SoilWallFrictionInput` layout.
    /// Diagnostic storage only; contents before an attempted solve are incomplete.
    pub fn conditional_inputs_buffer(&self) -> &wgpu::Buffer {
        &self.conditional_inputs
    }
    /// Local output layout; energy W is the soil-node kinetic-energy change.
    /// Machine XYZ is the final shared point velocity; residual W is zero.
    pub fn output_buffer(&self) -> &wgpu::Buffer {
        &self.outputs
    }
    /// Fault bits, completed sweeps, local fault bits, highest failing contact plus one.
    /// Next four words: accepted, rejected, attempted extrapolations, certificate diagnostic.
    /// The paired diagnostic packs the reason in the low byte and a one-based index above it.
    /// Remaining storage holds private sweep continuation state.
    pub fn status_buffer(&self) -> &wgpu::Buffer {
        &self.status
    }
    pub fn owned_buffer_bytes(&self) -> u64 {
        self.bytes
    }

    #[cfg(test)]
    pub(crate) fn capture_first_input(&mut self, engine: &Engine) -> capture::SharedWallSnapshot {
        let capture = capture::SharedWallCapture::new(
            engine,
            &self.capture_inputs,
            &self.status,
            self.capture_capacity,
            self.paired_rates,
        );
        let snapshot = capture.snapshot.clone();
        self.capture = Some(capture);
        snapshot
    }

    #[cfg(test)]
    pub(crate) fn trace_stages(
        &mut self,
        engine: std::sync::Arc<Engine>,
        path: std::path::PathBuf,
    ) {
        assert!(self.capture.is_none());
        self.trace = Some(capture::SharedWallTrace::new(
            engine,
            path,
            &self.capture_inputs,
            &self.status,
            self.capture_capacity,
            self.paired_rates,
        ));
    }
}

#[cfg(test)]
#[path = "soil_shared_wall_friction_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "soil_shared_wall_capture.rs"]
pub(crate) mod capture;

#[path = "soil_shared_wall_failure.rs"]
mod failure;
