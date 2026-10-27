//! Retained device factorization and checked contact response preparation.

use molla_compute::Engine;
use molla_core::{Error, Result};

#[derive(Clone, Copy)]
pub(crate) enum PreparationPhase {
    Reset,
    Geometry,
    Contacts,
    Factor,
    Response,
    Cleanup,
}

pub struct SoilNewtonPrepareInputs<'a> {
    pub mass: &'a wgpu::Buffer,
    pub point_rows: &'a wgpu::Buffer,
    pub contacts: &'a wgpu::Buffer,
    pub upstream_status: &'a wgpu::Buffer,
}

/// Factors an exactly symmetric mass matrix and verifies M R = J^T at 1e-6
/// componentwise backward error after two full residual corrections and up to
/// six further corrections restricted to uncertified residual components.
/// Failed preparation clears every response and latches diagnostics until reset.
pub struct ExperimentalSoilNewtonPrepareGpu {
    program: std::sync::Arc<PreparationProgram>,
    group: wgpu::BindGroup,
    response: wgpu::Buffer,
    #[cfg(test)]
    factor: wgpu::Buffer,
    status: wgpu::Buffer,
    failure: wgpu::Buffer,
    groups: u32,
    bytes: u64,
}

struct PreparationProgram {
    layout: wgpu::BindGroupLayout,
    pipelines: [wgpu::ComputePipeline; 3],
}

fn program_key(source: &str) -> String {
    format!(
        "soil-preparation-v1;wgsl={}",
        blake3::hash(source.as_bytes())
    )
}

fn invalid() -> Error {
    Error::Solver("soil Newton preparation: invalid capacity, usage or device limit".into())
}

fn create_program(engine: &Engine, source: &str) -> Result<PreparationProgram> {
    #[cfg(test)]
    let _setup = crate::soil_setup_trace::stage("newton.preparation.program");
    let device = engine.device().ok_or_else(invalid)?;
    #[cfg(all(
        any(test, feature = "strict-rounding-newton-preparation"),
        feature = "protected-newton",
        not(target_arch = "wasm32")
    ))]
    let specialized = {
        use crate::soil_newton_protected::preparation_rounding as rounding;
        if source.starts_with(rounding::ATOMIC)
            || source.starts_with(rounding::VOLATILE)
            || source.starts_with(rounding::STRICT)
        {
            Some(rounding::shaders(
                engine,
                source,
                !source.starts_with(rounding::ATOMIC),
            )?)
        } else {
            None
        }
    };
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("soil Newton preparation layout"),
        entries: &(0..10)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: if binding == 8 {
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
        label: Some("soil Newton preparation pipeline layout"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("soil Newton preparation"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipelines = ["factor", "prepare", "finish"].map(|entry| {
        #[cfg(test)]
        let _setup = crate::soil_setup_trace::stage(match entry {
            "factor" => "newton.preparation.factor",
            "prepare" => "newton.preparation.response",
            _ => "newton.preparation.finish",
        });
        #[cfg(all(
            any(test, feature = "strict-rounding-newton-preparation"),
            feature = "protected-newton",
            not(target_arch = "wasm32")
        ))]
        let module = specialized
            .as_ref()
            .and_then(|modules| match entry {
                "factor" => Some(&modules[0]),
                "prepare" => Some(&modules[1]),
                _ => None,
            })
            .unwrap_or(&module);
        #[cfg(not(all(
            any(test, feature = "strict-rounding-newton-preparation"),
            feature = "protected-newton",
            not(target_arch = "wasm32")
        )))]
        let module = &module;
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry),
            layout: Some(&pipeline_layout),
            module,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: engine.pipeline_cache(),
        })
    });
    Ok(PreparationProgram { layout, pipelines })
}

#[test]
fn newton_preparation_program_key_covers_source() {
    assert_eq!(program_key("a"), program_key("a"));
    assert_ne!(program_key("a"), program_key("b"));
    let axis = axis_parallel_source();
    assert_ne!(
        program_key(&axis),
        program_key(&format!("// PREPARATION_STRICT_VOLATILE_ROUNDING\n{axis}"))
    );
    assert_ne!(
        program_key(&axis_shared_source(3)),
        program_key(&axis_shared_source(10))
    );
}

#[test]
fn newton_preparation_selected_execution_limits() {
    type Prepare = ExperimentalSoilNewtonPrepareGpu;
    let (threads, shared, contacts_per_group) = Prepare::selected_execution();
    let expected = if cfg!(feature = "strict-rounding-newton-preparation") {
        (64, 384, 16)
    } else if cfg!(feature = "row-parallel-newton-preparation") {
        (128, 2048, 1)
    } else if cfg!(feature = "axis-parallel-newton-preparation") {
        (64, 384, 16)
    } else {
        (64, 256, 64)
    };
    assert_eq!((threads, shared, contacts_per_group), expected);
    let limits = wgpu::Limits {
        max_compute_invocations_per_workgroup: threads,
        max_compute_workgroup_size_x: threads,
        max_compute_workgroup_storage_size: shared,
        max_compute_workgroups_per_dimension: 16,
        ..Default::default()
    };
    for contacts in [0, 1, 16 * contacts_per_group] {
        assert!(Prepare::selected_limits_supported(&limits, contacts));
    }
    assert!(!Prepare::selected_limits_supported(
        &limits,
        16 * contacts_per_group + 1
    ));
    let mut short = limits.clone();
    short.max_compute_invocations_per_workgroup -= 1;
    assert!(!Prepare::selected_limits_supported(&short, 1));
    let mut short = limits.clone();
    short.max_compute_workgroup_size_x -= 1;
    assert!(!Prepare::selected_limits_supported(&short, 1));
    let mut short = limits.clone();
    short.max_compute_workgroup_storage_size -= 1;
    assert!(!Prepare::selected_limits_supported(&short, 1));
    let mut short = limits;
    short.max_compute_workgroups_per_dimension = 0;
    assert!(!Prepare::selected_limits_supported(&short, 0));
}

#[cfg(any(test, feature = "axis-parallel-newton-preparation"))]
pub(crate) fn axis_parallel_source() -> String {
    let source = sparse_residual_source();
    let start = source
        .find("@compute @workgroup_size(64)\nfn prepare(")
        .unwrap();
    let end = source
        .find("@compute @workgroup_size(64)\nfn finish(")
        .unwrap();
    format!(
        "{}\n{}\n{}",
        &source[..start],
        include_str!("../shaders/soil_newton_prepare_axes.wgsl"),
        source[end..].replace("@workgroup_size(64)", "@workgroup_size(16)")
    )
}

#[cfg(test)]
pub(crate) fn axis_shared_bytes(dofs: u32) -> Result<u32> {
    if !(1..=32).contains(&dofs) {
        return Err(invalid());
    }
    Ok(384 + 512 * dofs)
}

#[cfg(test)]
pub(crate) fn axis_shared_source(dofs: u32) -> String {
    axis_shared_bytes(dofs).unwrap();
    let source = axis_parallel_source();
    let end = source
        .find("@compute @workgroup_size(16)\nfn finish(")
        .unwrap();
    let mut body = source[..end].to_owned();
    for name in ["response", "correction"] {
        for index in ["i", "j"] {
            body = body.replace(
                &format!("{name}[c*n+{index}][axis]"),
                &format!("axis_{name}[64u*{index}+4u*(c%16u)+axis]"),
            );
        }
    }
    let marker =
        "    storageBarrier();\n    if axis==0u && (atomicLoad(&prepare_state[slot])&3u)==1u {";
    assert_eq!(body.matches(marker).count(), 1);
    body = body.replace(
        marker,
        "    workgroupBarrier();\n\
        if c<max(params.contacts,1u) {for(var i=0u;i<n;i++) {\n\
            response[c*n+i][axis]=axis_response[64u*i+lane];\n\
            correction[c*n+i][axis]=axis_correction[64u*i+lane];\n\
        }}\n\
        storageBarrier();\n\
        if axis==0u && (atomicLoad(&prepare_state[slot])&3u)==1u {",
    );
    let count = 64 * dofs;
    format!(
        "var<workgroup> axis_response:array<f32,{count}>;\nvar<workgroup> axis_correction:array<f32,{count}>;\n{body}{}",
        &source[end..]
    )
}

#[cfg(test)]
pub(crate) fn certificate_mask_source() -> String {
    let source = sparse_residual_source();
    let start = source.find("fn certified(c:u32)->bool {").unwrap();
    let end = source[start..].find("@compute").unwrap() + start;
    let original = &source[start..end];
    let cached = "var<private> certificate_masks:array<u32,32>;\n\
fn certified(c:u32)->bool {\n\
    var all_valid=true;\n\
    for(var i=0u;i<params.width;i++) {\n\
        var mask=0u;\n\
        for(var axis=0u;axis<3u;axis++) {\n\
            let error=component_error(c,i,axis);\n\
            let valid=error>=0.0 && error<=1e-6;\n\
            mask|=select(0u,1u<<axis,valid);\n\
            all_valid=all_valid && valid;\n\
        }\n\
        certificate_masks[i]=mask;\n\
    }\n\
    return all_valid;\n\
}\n";
    let check = "let error=component_error(c,i,axis);\n                if error>=0.0 && error<=1e-6 {correction[c*n+i][axis]=0.0;continue;}";
    assert_eq!(source.matches(check).count(), 1);
    source.replace(original, cached).replace(
        check,
        "if (certificate_masks[i] & (1u<<axis))!=0u {correction[c*n+i][axis]=0.0;continue;}",
    )
}

#[cfg(test)]
pub(crate) fn narrow_workgroup_source() -> String {
    let source = sparse_residual_source();
    let arithmetic = crate::soil_rate_encoding::sorted_arithmetic(64);
    assert_eq!(source.matches(&arithmetic).count(), 1);
    assert_eq!(source.matches("@workgroup_size(64)").count(), 2);
    source
        .replace(
            &arithmetic,
            &crate::soil_rate_encoding::sorted_arithmetic(8),
        )
        .replace("@workgroup_size(64)", "@workgroup_size(8)")
}

#[cfg(test)]
pub(crate) fn native_pair_source() -> String {
    let source = sparse_residual_source();
    let arithmetic = crate::soil_rate_encoding::sorted_arithmetic(64);
    assert_eq!(source.matches(&arithmetic).count(), 1);
    source.replace(
        &arithmetic,
        &crate::soil_rate_encoding::native_arithmetic(64),
    )
}

#[cfg(any(test, feature = "row-parallel-newton-preparation"))]
pub(crate) fn row_parallel_source() -> String {
    let source = sparse_residual_source().replacen(
        &crate::soil_rate_encoding::sorted_arithmetic(64),
        &crate::soil_rate_encoding::sorted_arithmetic(128),
        1,
    );
    let start = source
        .find("@compute @workgroup_size(64)\nfn prepare(")
        .unwrap();
    let end = source
        .find("@compute @workgroup_size(64)\nfn finish(")
        .unwrap();
    format!(
        "{}\n{}\n{}",
        &source[..start],
        include_str!("../shaders/soil_newton_prepare_rows.wgsl"),
        source[end..].replace("@workgroup_size(64)", "@workgroup_size(1)")
    )
}

#[test]
fn newton_preparation_row_parallel_contract() {
    let text = row_parallel_source();
    let module = naga::front::wgsl::parse_str(&text)
        .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&text)));
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&text)));
    assert!(text.contains("refinement<8u") && text.contains("refinement>=2u"));
    assert!(text.contains("for(var j=i+1u;j<n;j++)"));
}

#[cfg(any(test, feature = "row-parallel-newton-preparation"))]
pub(crate) fn row_factor_source() -> String {
    let source = row_parallel_source();
    let start = source
        .find("@compute @workgroup_size(1)\nfn factor(")
        .unwrap();
    let end = source.find("var<workgroup> row_solution:").unwrap();
    assert!(end > start);
    format!(
        "{}\n{}\n{}",
        &source[..start],
        include_str!("../shaders/soil_newton_prepare_factor.wgsl"),
        &source[end..]
    )
}

#[test]
fn newton_preparation_row_factor_contract() {
    let text = row_factor_source();
    let module = naga::front::wgsl::parse_str(&text)
        .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&text)));
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&text)));
    assert!(text.contains("for(var k=0u;k<j;k++)"));
    assert!(text.contains("refinement<8u") && text.contains("refinement>=2u"));
}

#[cfg(test)]
fn row_sized_lanes(dofs: u32) -> Result<u32> {
    match dofs {
        1..=10 => Ok(32),
        11..=21 => Ok(64),
        22..=32 => Ok(128),
        _ => Err(invalid()),
    }
}

#[cfg(test)]
fn row_sized_source(dofs: u32) -> Result<String> {
    let lanes = row_sized_lanes(dofs)?;
    let source = row_factor_source();
    let marker = "@compute @workgroup_size(128)\nfn prepare(";
    assert_eq!(source.matches(marker).count(), 1);
    Ok(source.replace(
        marker,
        &format!("@compute @workgroup_size({lanes})\nfn prepare("),
    ))
}

#[test]
fn newton_preparation_row_sized_contract() {
    for dofs in [0, 33, u32::MAX] {
        assert!(row_sized_source(dofs).is_err());
    }
    for dofs in 1..=32 {
        let lanes = row_sized_lanes(dofs).unwrap();
        assert!(lanes >= 3 * dofs && [32, 64, 128].contains(&lanes));
        let text = row_sized_source(dofs).unwrap();
        assert_eq!(
            text.replace(
                &format!("@workgroup_size({lanes})\nfn prepare("),
                "@workgroup_size(128)\nfn prepare("
            ),
            row_factor_source()
        );
        let module = naga::front::wgsl::parse_str(&text).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
        let prepare = module
            .entry_points
            .iter()
            .find(|e| e.name == "prepare")
            .unwrap();
        assert_eq!(prepare.workgroup_size, [lanes, 1, 1]);
    }
}

pub(crate) fn shader_source() -> String {
    format!(
        "{}\n{}",
        crate::soil_rate_encoding::sorted_arithmetic(64),
        include_str!("../shaders/soil_newton_prepare.wgsl")
    )
}

#[cfg(test)]
pub(crate) fn native_accumulator_source() -> String {
    let source = include_str!("../shaders/soil_newton_prepare.wgsl");
    let value = "fn value(x:vec2<f32>)->f32 {return rate_round(x.x+x.y);}";
    assert_eq!(source.matches(value).count(), 1);
    let source = source
        .replace(value, "fn value(x:f64)->f32 {return f32(x);}")
        .replace("vec2<f32>(", "f64(")
        .replace(",0.0)", ")")
        .replace("rate_lane=0u;", "")
        .replace("rate_lane=lane;", "");
    assert!(!source.contains("vec2<f32>") && !source.contains("rate_round"));
    format!(
        "fn rate_add(a:f64,b:f64)->f64 {{return a+b;}}\nfn rate_mul(a:f32,b:f64)->f64 {{return f64(a)*b;}}\nfn rate_sum(a:f32,b:f32)->f64 {{return f64(a)+f64(b);}}\n{source}"
    )
}

#[cfg(test)]
pub(crate) fn private_response_source() -> String {
    let source = shader_source();
    let start = source
        .find("@compute @workgroup_size(64)\nfn prepare(")
        .unwrap();
    let end = source
        .find("@compute @workgroup_size(64)\nfn finish(")
        .unwrap();
    let prefix = source[..start].replace("response[c*n+j]", "local_response[j]");
    let body = &source[start..end];
    let init =
        "for(var i=0u;i<n;i++) {response[c*n+i]=vec4<f32>(0.0);correction[c*n+i]=vec4<f32>(0.0);}";
    assert_eq!(body.matches(init).count(), 1);
    let mut body = body.replace(init, "__GLOBAL_CLEAR__");
    for name in ["response", "correction"] {
        for index in ["i", "j"] {
            body = body.replace(
                &format!("{name}[c*n+{index}]"),
                &format!("local_{name}[{index}]"),
            );
        }
    }
    assert!(!body.contains("response[c*n+") && !body.contains("correction[c*n+"));
    body = body.replace("__GLOBAL_CLEAR__", &format!("{init}\n    for(var i=0u;i<n;i++) {{local_response[i]=vec4<f32>(0.0);local_correction[i]=vec4<f32>(0.0);}}"));
    let end_body = body.rfind('}').unwrap();
    body.insert_str(end_body, "    for(var i=0u;i<n;i++) {response[c*n+i]=local_response[i];correction[c*n+i]=local_correction[i];}\n");
    format!(
        "var<private> local_response:array<vec4<f32>,32>;\nvar<private> local_correction:array<vec4<f32>,32>;\n{prefix}{body}{}",
        &source[end..]
    )
}

#[cfg(any(test, feature = "zero-term-newton-preparation"))]
pub(crate) fn zero_terms_source() -> String {
    let mut source = shader_source();
    for (coefficient, term) in [
        ("-lower[i*n+k]", "lower[j*n+k]"),
        ("-lower[i*n+j]", "response[c*n+j][axis]"),
        ("-lower[j*n+i]", "response[c*n+j][axis]"),
        ("-lower[i*n+j]", "correction[c*n+j][axis]"),
        ("-lower[j*n+i]", "correction[c*n+j][axis]"),
    ] {
        let original = format!("sum=rate_add(sum,rate_mul({coefficient},vec2<f32>({term},0.0)))");
        assert_eq!(source.matches(&original).count(), 1);
        source = source.replace(
            &original,
            &format!("sum=add_nonzero_term(sum,{coefficient},{term})"),
        );
    }
    format!(
        "fn add_nonzero_term(sum:vec2<f32>,coefficient:f32,term:f32)->vec2<f32> {{\n    if coefficient==0.0 && finite(term) {{return sum;}}\n    return rate_add(sum,rate_mul(coefficient,vec2<f32>(term,0.0)));\n}}\n{source}"
    )
}

#[cfg(any(test, feature = "zero-residual-newton-preparation"))]
pub(crate) fn sparse_residual_source() -> String {
    let mut source = zero_terms_source();
    for (coefficient, term) in [("m", "r"), ("-mass[i*n+j]", "response[c*n+j][axis]")] {
        let original =
            format!("residual=rate_add(residual,rate_mul({coefficient},vec2<f32>({term},0.0)))");
        assert_eq!(source.matches(&original).count(), 1);
        source = source.replace(
            &original,
            &format!("residual=add_nonzero_term(residual,{coefficient},{term})"),
        );
    }
    source
}

impl ExperimentalSoilNewtonPrepareGpu {
    #[cfg(test)]
    pub(crate) fn new_row_sized(
        engine: &Engine,
        input: SoilNewtonPrepareInputs<'_>,
        dofs: u32,
        contacts: u32,
        max_owned_bytes: u64,
    ) -> Result<Self> {
        let lanes = row_sized_lanes(dofs)?;
        engine.check_device()?;
        let limits = engine.device().ok_or_else(invalid)?.limits();
        if limits.max_compute_invocations_per_workgroup < lanes
            || limits.max_compute_workgroup_size_x < lanes
            || limits.max_compute_workgroup_storage_size < 2048
        {
            return Err(invalid());
        }
        Self::new_with_execution(
            engine,
            input,
            dofs,
            contacts,
            max_owned_bytes,
            row_sized_source(dofs)?,
            1,
        )
    }

    #[cfg(any(test, feature = "row-parallel-newton-preparation"))]
    pub(crate) fn new_row_factor(
        engine: &Engine,
        input: SoilNewtonPrepareInputs<'_>,
        dofs: u32,
        contacts: u32,
        max_owned_bytes: u64,
    ) -> Result<Self> {
        engine.check_device()?;
        let limits = engine.device().ok_or_else(invalid)?.limits();
        if limits.max_compute_invocations_per_workgroup < 128
            || limits.max_compute_workgroup_size_x < 128
            || limits.max_compute_workgroup_storage_size < 2048
        {
            return Err(invalid());
        }
        Self::new_with_execution(
            engine,
            input,
            dofs,
            contacts,
            max_owned_bytes,
            row_factor_source(),
            1,
        )
    }

    #[cfg(test)]
    pub(crate) fn new_row_parallel(
        engine: &Engine,
        input: SoilNewtonPrepareInputs<'_>,
        dofs: u32,
        contacts: u32,
        max_owned_bytes: u64,
    ) -> Result<Self> {
        engine.check_device()?;
        let limits = engine.device().ok_or_else(invalid)?.limits();
        if limits.max_compute_invocations_per_workgroup < 128
            || limits.max_compute_workgroup_size_x < 128
            || limits.max_compute_workgroup_storage_size < 2048
        {
            return Err(invalid());
        }
        Self::new_with_execution(
            engine,
            input,
            dofs,
            contacts,
            max_owned_bytes,
            row_parallel_source(),
            1,
        )
    }

    pub const fn backend_label() -> &'static str {
        if cfg!(feature = "strict-rounding-newton-preparation") {
            "compensated-axis-parallel-strict-volatile-rounding"
        } else if cfg!(feature = "row-parallel-newton-preparation") {
            "compensated-row-parallel-cooperative-factor"
        } else if cfg!(feature = "axis-parallel-newton-preparation") {
            "compensated-axis-parallel-zero-term-and-residual-elision"
        } else if cfg!(feature = "zero-residual-newton-preparation") {
            "compensated-zero-term-and-residual-elision"
        } else if cfg!(feature = "zero-term-newton-preparation") {
            "compensated-zero-term-elision"
        } else {
            "compensated-dense-terms"
        }
    }

    fn selected_execution() -> (u32, u32, u32) {
        if cfg!(feature = "strict-rounding-newton-preparation") {
            (64, 384, 16)
        } else if cfg!(feature = "row-parallel-newton-preparation") {
            (128, 2048, 1)
        } else if cfg!(feature = "axis-parallel-newton-preparation") {
            (64, 384, 16)
        } else {
            (64, 256, 64)
        }
    }

    fn selected_limits_supported(limits: &wgpu::Limits, contacts: u32) -> bool {
        let (threads, shared, contacts_per_group) = Self::selected_execution();
        limits.max_compute_invocations_per_workgroup >= threads
            && limits.max_compute_workgroup_size_x >= threads
            && limits.max_compute_workgroup_storage_size >= shared
            && contacts.max(1).div_ceil(contacts_per_group)
                <= limits.max_compute_workgroups_per_dimension
    }

    pub(crate) fn check_selected_execution(engine: &Engine, contacts: u32) -> Result<()> {
        engine.check_device()?;
        let device = engine.device().ok_or_else(invalid)?;
        if !Self::selected_limits_supported(&device.limits(), contacts) {
            return Err(invalid());
        }
        #[cfg(feature = "strict-rounding-newton-preparation")]
        if cfg!(target_arch = "wasm32") || !crate::soil_newton_gpu::protected_enabled(engine) {
            return Err(invalid());
        }
        Ok(())
    }

    pub fn required_bytes(dofs: u32, contacts: u32) -> Result<u64> {
        if !(1..=32).contains(&dofs) {
            return Err(invalid());
        }
        let n = u64::from(dofs);
        let c = u64::from(contacts.max(1));
        Ok(8 * n * n + 32 * c * n + 16 * n + 176)
    }

    pub fn new(
        engine: &Engine,
        input: SoilNewtonPrepareInputs<'_>,
        dofs: u32,
        contacts: u32,
        max_owned_bytes: u64,
    ) -> Result<Self> {
        Self::check_selected_execution(engine, contacts)?;
        #[cfg(all(
            feature = "strict-rounding-newton-preparation",
            not(target_arch = "wasm32")
        ))]
        {
            Self::new_axis_rounding::<2>(engine, input, dofs, contacts, max_owned_bytes)
        }
        #[cfg(all(feature = "strict-rounding-newton-preparation", target_arch = "wasm32"))]
        {
            Err(invalid())
        }
        #[cfg(all(
            feature = "axis-parallel-newton-preparation",
            not(feature = "strict-rounding-newton-preparation")
        ))]
        {
            #[cfg(feature = "row-parallel-newton-preparation")]
            {
                Self::new_row_factor(engine, input, dofs, contacts, max_owned_bytes)
            }
            #[cfg(not(feature = "row-parallel-newton-preparation"))]
            {
                Self::new_axis_parallel(engine, input, dofs, contacts, max_owned_bytes)
            }
        }
        #[cfg(not(feature = "axis-parallel-newton-preparation"))]
        {
            #[cfg(feature = "zero-residual-newton-preparation")]
            let source = sparse_residual_source();
            #[cfg(all(
                feature = "zero-term-newton-preparation",
                not(feature = "zero-residual-newton-preparation")
            ))]
            let source = zero_terms_source();
            #[cfg(not(feature = "zero-term-newton-preparation"))]
            let source = shader_source();
            Self::new_with_source(engine, input, dofs, contacts, max_owned_bytes, source)
        }
    }

    #[cfg(any(test, feature = "axis-parallel-newton-preparation"))]
    pub(crate) fn new_axis_parallel(
        engine: &Engine,
        input: SoilNewtonPrepareInputs<'_>,
        dofs: u32,
        contacts: u32,
        max_owned_bytes: u64,
    ) -> Result<Self> {
        engine.check_device()?;
        let limits = engine.device().ok_or_else(invalid)?.limits();
        if limits.max_compute_invocations_per_workgroup < 64
            || limits.max_compute_workgroup_size_x < 64
            || limits.max_compute_workgroup_storage_size < 384
        {
            return Err(invalid());
        }
        Self::new_with_execution(
            engine,
            input,
            dofs,
            contacts,
            max_owned_bytes,
            axis_parallel_source(),
            16,
        )
    }

    #[cfg(test)]
    pub(crate) fn new_axis_shared(
        engine: &Engine,
        input: SoilNewtonPrepareInputs<'_>,
        dofs: u32,
        contacts: u32,
        max_owned_bytes: u64,
    ) -> Result<Self> {
        let shared_bytes = axis_shared_bytes(dofs)?;
        engine.check_device()?;
        let limits = engine.device().ok_or_else(invalid)?.limits();
        if limits.max_compute_invocations_per_workgroup < 64
            || limits.max_compute_workgroup_size_x < 64
            || limits.max_compute_workgroup_storage_size < shared_bytes
        {
            return Err(invalid());
        }
        Self::new_with_execution(
            engine,
            input,
            dofs,
            contacts,
            max_owned_bytes,
            axis_shared_source(dofs),
            16,
        )
    }

    #[cfg(all(
        any(test, feature = "strict-rounding-newton-preparation"),
        feature = "protected-newton",
        not(target_arch = "wasm32")
    ))]
    pub(crate) fn new_axis_rounding<const MODE: u32>(
        engine: &Engine,
        input: SoilNewtonPrepareInputs<'_>,
        dofs: u32,
        contacts: u32,
        max_owned_bytes: u64,
    ) -> Result<Self> {
        use crate::soil_newton_protected::preparation_rounding as rounding;
        engine.check_device()?;
        let limits = engine.device().ok_or_else(invalid)?.limits();
        if !crate::soil_newton_gpu::protected_enabled(engine)
            || MODE > 2
            || (!cfg!(test) && MODE != 2)
            || limits.max_compute_invocations_per_workgroup < 64
            || limits.max_compute_workgroup_size_x < 64
            || limits.max_compute_workgroup_storage_size < 384
        {
            return Err(invalid());
        }
        let marker = match MODE {
            0 => rounding::ATOMIC,
            1 => rounding::VOLATILE,
            _ => rounding::STRICT,
        };
        Self::new_with_execution(
            engine,
            input,
            dofs,
            contacts,
            max_owned_bytes,
            format!("{marker}{}", axis_parallel_source()),
            16,
        )
    }

    #[cfg(test)]
    pub(crate) fn new_certificate_mask(
        engine: &Engine,
        input: SoilNewtonPrepareInputs<'_>,
        dofs: u32,
        contacts: u32,
        max_owned_bytes: u64,
    ) -> Result<Self> {
        Self::new_with_source(
            engine,
            input,
            dofs,
            contacts,
            max_owned_bytes,
            certificate_mask_source(),
        )
    }

    #[cfg(test)]
    pub(crate) fn new_native_pairs(
        engine: &Engine,
        input: SoilNewtonPrepareInputs<'_>,
        dofs: u32,
        contacts: u32,
        max_owned_bytes: u64,
    ) -> Result<Self> {
        if !engine
            .device()
            .ok_or_else(invalid)?
            .features()
            .contains(wgpu::Features::SHADER_F64)
        {
            return Err(invalid());
        }
        Self::new_with_source(
            engine,
            input,
            dofs,
            contacts,
            max_owned_bytes,
            native_pair_source(),
        )
    }

    #[cfg(test)]
    pub(crate) fn new_narrow_workgroup(
        engine: &Engine,
        input: SoilNewtonPrepareInputs<'_>,
        dofs: u32,
        contacts: u32,
        max_owned_bytes: u64,
    ) -> Result<Self> {
        Self::new_with_execution(
            engine,
            input,
            dofs,
            contacts,
            max_owned_bytes,
            narrow_workgroup_source(),
            8,
        )
    }

    #[cfg(test)]
    pub(crate) fn new_native_accumulator(
        engine: &Engine,
        input: SoilNewtonPrepareInputs<'_>,
        dofs: u32,
        contacts: u32,
        max_owned_bytes: u64,
    ) -> Result<Self> {
        if !engine
            .device()
            .ok_or_else(invalid)?
            .features()
            .contains(wgpu::Features::SHADER_F64)
        {
            return Err(invalid());
        }
        Self::new_with_source(
            engine,
            input,
            dofs,
            contacts,
            max_owned_bytes,
            native_accumulator_source(),
        )
    }

    #[cfg(test)]
    pub(crate) fn new_reference(
        engine: &Engine,
        input: SoilNewtonPrepareInputs<'_>,
        dofs: u32,
        contacts: u32,
        max_owned_bytes: u64,
    ) -> Result<Self> {
        Self::new_with_source(
            engine,
            input,
            dofs,
            contacts,
            max_owned_bytes,
            shader_source(),
        )
    }

    #[cfg(test)]
    pub(crate) fn new_fixed_width(
        engine: &Engine,
        input: SoilNewtonPrepareInputs<'_>,
        dofs: u32,
        contacts: u32,
        max_owned_bytes: u64,
    ) -> Result<Self> {
        Self::required_bytes(dofs, contacts)?;
        let source = shader_source();
        assert_eq!(source.matches("params.width").count(), 8);
        Self::new_with_source(
            engine,
            input,
            dofs,
            contacts,
            max_owned_bytes,
            source.replace("params.width", &format!("{dofs}u")),
        )
    }

    #[cfg(test)]
    pub(crate) fn new_private_response(
        engine: &Engine,
        input: SoilNewtonPrepareInputs<'_>,
        dofs: u32,
        contacts: u32,
        max_owned_bytes: u64,
    ) -> Result<Self> {
        Self::new_with_source(
            engine,
            input,
            dofs,
            contacts,
            max_owned_bytes,
            private_response_source(),
        )
    }

    #[cfg(test)]
    pub(crate) fn new_zero_terms(
        engine: &Engine,
        input: SoilNewtonPrepareInputs<'_>,
        dofs: u32,
        contacts: u32,
        max_owned_bytes: u64,
    ) -> Result<Self> {
        Self::new_with_source(
            engine,
            input,
            dofs,
            contacts,
            max_owned_bytes,
            zero_terms_source(),
        )
    }

    #[cfg(test)]
    pub(crate) fn new_sparse_residual(
        engine: &Engine,
        input: SoilNewtonPrepareInputs<'_>,
        dofs: u32,
        contacts: u32,
        max_owned_bytes: u64,
    ) -> Result<Self> {
        Self::new_with_source(
            engine,
            input,
            dofs,
            contacts,
            max_owned_bytes,
            sparse_residual_source(),
        )
    }

    fn new_with_source(
        engine: &Engine,
        input: SoilNewtonPrepareInputs<'_>,
        dofs: u32,
        contacts: u32,
        max_owned_bytes: u64,
        source: String,
    ) -> Result<Self> {
        Self::new_with_execution(engine, input, dofs, contacts, max_owned_bytes, source, 64)
    }

    #[allow(clippy::too_many_arguments)]
    fn new_with_execution(
        engine: &Engine,
        input: SoilNewtonPrepareInputs<'_>,
        dofs: u32,
        contacts: u32,
        max_owned_bytes: u64,
        source: String,
        lanes: u32,
    ) -> Result<Self> {
        engine.check_device()?;
        let device = engine.device().ok_or_else(invalid)?;
        let limits = device.limits();
        let bytes = Self::required_bytes(dofs, contacts)?;
        let n = u64::from(dofs);
        let c = u64::from(contacts.max(1));
        if !lanes.is_power_of_two() || lanes > 64 {
            return Err(invalid());
        }
        let groups = contacts.max(1).div_ceil(lanes);
        let borrowed = [
            input.mass,
            input.point_rows,
            input.contacts,
            input.upstream_status,
        ];
        let sizes = [
            4 * n * n,
            16 * c * n,
            16 * c * n,
            32,
            16,
            4 * n * n + 16 * n + 128,
        ];
        if bytes > max_owned_bytes
            || limits.max_storage_buffers_per_shader_stage < 9
            || limits.max_bindings_per_bind_group < 10
            || limits.max_uniform_buffers_per_shader_stage < 1
            || limits.max_compute_invocations_per_workgroup < lanes
            || limits.max_compute_workgroup_size_x < lanes
            || limits.max_compute_workgroup_storage_size < 4 * lanes
            || groups > limits.max_compute_workgroups_per_dimension
            || borrowed
                .iter()
                .zip([4 * n * n, 16 * c * n, 64 * c, 4])
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
        let key = program_key(&source);
        let setup = std::env::var_os("MOLLA_NEWTON_SETUP_TRACE").map(|_| std::time::Instant::now());
        let create = |_: &wgpu::Device| {
            if setup.is_some() {
                eprintln!("preparation setup program={key} build begin");
            }
            create_program(engine, &source)
        };
        #[cfg(not(target_arch = "wasm32"))]
        let program = engine.cached_gpu_program(&key, create)?;
        #[cfg(target_arch = "wasm32")]
        let program = std::sync::Arc::new(create(device)?);
        if let Some(start) = setup {
            eprintln!(
                "preparation setup program={key} ready_ms={}",
                start.elapsed().as_millis()
            );
        }
        let buffers: Vec<_> = sizes
            .iter()
            .enumerate()
            .map(|(i, &size)| {
                molla_core::gpu_create_buffer!(
                    device,
                    &wgpu::BufferDescriptor {
                        label: Some("soil Newton preparation storage"),
                        size,
                        usage: if i == 4 {
                            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST
                        } else {
                            wgpu::BufferUsages::STORAGE
                                | wgpu::BufferUsages::COPY_DST
                                | wgpu::BufferUsages::COPY_SRC
                        },
                        mapped_at_creation: false
                    }
                )
            })
            .collect();
        molla_core::gpu_write_buffer!(
            engine.queue().unwrap(),
            &buffers[4],
            0,
            bytemuck::cast_slice(&[dofs, contacts, 0, 0])
        );
        let bindings = [
            input.mass,
            input.point_rows,
            input.contacts,
            input.upstream_status,
            &buffers[0],
            &buffers[1],
            &buffers[2],
            &buffers[3],
            &buffers[4],
            &buffers[5],
        ];
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("soil Newton preparation bindings"),
            layout: &program.layout,
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
            program,
            group,
            response: buffers[1].clone(),
            #[cfg(test)]
            factor: buffers[0].clone(),
            status: buffers[3].clone(),
            failure: buffers[5].clone(),
            groups,
            bytes,
        })
    }

    pub fn encode_reset(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.clear_buffer(&self.status, 0, None);
    }
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        self.encode_traced(encoder, &mut |_, _, _| {});
    }
    pub(crate) fn encode_traced(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        trace: &mut impl FnMut(&mut wgpu::CommandEncoder, PreparationPhase, bool),
    ) {
        for ((pipeline, groups), phase) in self
            .program
            .pipelines
            .iter()
            .zip([1, self.groups, self.groups])
            .zip([
                PreparationPhase::Factor,
                PreparationPhase::Response,
                PreparationPhase::Cleanup,
            ])
        {
            trace(encoder, phase, true);
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &self.group, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
            molla_core::gpu_metrics::record_dispatch();
            drop(pass);
            trace(encoder, phase, false);
        }
    }
    pub fn response_buffer(&self) -> &wgpu::Buffer {
        &self.response
    }
    #[cfg(test)]
    pub(crate) fn factor_buffer(&self) -> &wgpu::Buffer {
        &self.factor
    }
    #[cfg(test)]
    pub(crate) fn shares_program(&self, other: &Self) -> bool {
        std::sync::Arc::ptr_eq(&self.program, &other.program)
    }
    #[cfg(test)]
    pub(crate) fn encode_uncleared_diagnostic(&self, encoder: &mut wgpu::CommandEncoder) {
        for (pipeline, groups) in self.program.pipelines[..2].iter().zip([1, self.groups]) {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &self.group, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
    }
    /// Sticky fault, failing contact (one-based), active count, maximum backward
    /// error as f32 bits, padding. Fault bits: input=1, mass=2, response=4.
    pub fn status_buffer(&self) -> &wgpu::Buffer {
        &self.status
    }
    /// Little-endian words: schema, width, source slots, one-based contact,
    /// eight status words, upstream fault, three reserved words; then mass,
    /// one contact's point rows and its 64-byte contact record. Schema zero is
    /// empty. The packet survives upstream rejection until a clean preparation.
    pub fn failure_buffer(&self) -> &wgpu::Buffer {
        &self.failure
    }
    pub fn owned_buffer_bytes(&self) -> u64 {
        self.bytes
    }
}

pub(crate) fn capture_failure(
    engine: &Engine,
    buffer: &wgpu::Buffer,
    path: &std::path::Path,
    map: impl FnOnce(&wgpu::Buffer) -> Result<()>,
) -> Result<()> {
    use std::io::Write;
    let device = engine.device().ok_or_else(invalid)?;
    let staging = molla_core::gpu_create_buffer!(
        device,
        &wgpu::BufferDescriptor {
            label: Some("Newton preparation failure readback"),
            size: buffer.size(),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        }
    );
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, buffer.size());
    engine.queue().unwrap().submit([encoder.finish()]);
    map(&staging)?;
    let mapped = molla_core::gpu_readback!(staging.slice(..));
    let bytes = mapped.to_vec();
    drop(mapped);
    staging.unmap();
    if bytes[..4] != 1u32.to_le_bytes() {
        return Err(invalid());
    }
    let io = |e: std::io::Error| Error::Solver(format!("Newton preparation capture: {e}"));
    std::fs::create_dir(path).map_err(io)?;
    let write = |name: &str, data: &[u8]| -> Result<()> {
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path.join(name))
            .map_err(io)?
            .write_all(data)
            .map_err(io)
    };
    write("preparation.bin", &bytes)?;
    write("capture.json", format!(
        r#"{{"schema":1,"stage":"newton-preparation","acceptance":false,"byte_order":"little","bytes":{},"blake3":"{}"}}"#,
        bytes.len(), blake3::hash(&bytes).to_hex()).as_bytes())
}
