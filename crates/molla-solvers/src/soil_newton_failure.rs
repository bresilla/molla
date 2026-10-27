use molla_compute::Engine;
use molla_core::{Error, Result};

#[path = "soil_newton_certificate.rs"]
pub(crate) mod certificate;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub(crate) enum NewtonFailureMethod {
    Unspecified = 0,
    StagnationPairs = 1,
    FischerPairs = 2,
    FischerContinuation = 3,
    FischerContinuationRelease = 4,
    FischerCondensed = 5,
    FischerDirect = 6,
    FischerHybrid = 7,
    FischerAdaptiveDirect = 8,
    FischerAdaptiveHybrid = 9,
    FischerConditioned = 10,
    FischerConditionedProtected = 11,
    FischerConditionedProtectedLow = 12,
    FischerConditionedStalled = 13,
    FischerConditionedProtectedStalled = 14,
    FischerConditionedProtectedWide256 = 15,
    FischerConditionedProtectedCached256 = 16,
    FischerConditionedProtectedParallel256 = 17,
    FischerConditionedProtectedVolatile256 = 18,
    FischerConditionedProtectedSchur256 = 19,
    FischerConditionedProtectedNorm256 = 20,
    FischerConditionedProtectedActivePublication256 = 21,
    FischerConditionedProtectedPublicationProducts256 = 22,
    FischerConditionedProtectedRetainedLocal256 = 23,
    FischerConditionedProtectedProjectionCache256 = 24,
    FischerConditionedProtectedSharedCore256 = 25,
    FischerConditionedProtectedStreamBlock256 = 26,
    FischerConditionedProtectedStagnantBranch256 = 27,
    FischerConditionedProtectedGroupedSum256 = 28,
    FischerConditionedProtectedGroupedEight256 = 29,
    FischerConditionedProtectedParallelIndices256 = 30,
    FischerConditionedProtectedSoaScratch256 = 31,
}

impl NewtonFailureMethod {
    fn from_tag(tag: u32) -> Result<Self> {
        match tag {
            0 => Ok(Self::Unspecified),
            1 => Ok(Self::StagnationPairs),
            2 => Ok(Self::FischerPairs),
            3 => Ok(Self::FischerContinuation),
            4 => Ok(Self::FischerContinuationRelease),
            5 => Ok(Self::FischerCondensed),
            6 => Ok(Self::FischerDirect),
            7 => Ok(Self::FischerHybrid),
            8 => Ok(Self::FischerAdaptiveDirect),
            9 => Ok(Self::FischerAdaptiveHybrid),
            10 => Ok(Self::FischerConditioned),
            11 => Ok(Self::FischerConditionedProtected),
            12 => Ok(Self::FischerConditionedProtectedLow),
            13 => Ok(Self::FischerConditionedStalled),
            14 => Ok(Self::FischerConditionedProtectedStalled),
            15 => Ok(Self::FischerConditionedProtectedWide256),
            16 => Ok(Self::FischerConditionedProtectedCached256),
            17 => Ok(Self::FischerConditionedProtectedParallel256),
            18 => Ok(Self::FischerConditionedProtectedVolatile256),
            19 => Ok(Self::FischerConditionedProtectedSchur256),
            20 => Ok(Self::FischerConditionedProtectedNorm256),
            21 => Ok(Self::FischerConditionedProtectedActivePublication256),
            22 => Ok(Self::FischerConditionedProtectedPublicationProducts256),
            23 => Ok(Self::FischerConditionedProtectedRetainedLocal256),
            24 => Ok(Self::FischerConditionedProtectedProjectionCache256),
            25 => Ok(Self::FischerConditionedProtectedSharedCore256),
            26 => Ok(Self::FischerConditionedProtectedStreamBlock256),
            27 => Ok(Self::FischerConditionedProtectedStagnantBranch256),
            28 => Ok(Self::FischerConditionedProtectedGroupedSum256),
            29 => Ok(Self::FischerConditionedProtectedGroupedEight256),
            30 => Ok(Self::FischerConditionedProtectedParallelIndices256),
            31 => Ok(Self::FischerConditionedProtectedSoaScratch256),
            _ => Err(invalid()),
        }
    }

    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Unspecified => "unspecified-pairs",
            Self::StagnationPairs => "stagnation-pairs",
            Self::FischerPairs => "fischer-pairs",
            Self::FischerContinuation => "fischer-continuation-pairs",
            Self::FischerContinuationRelease => "fischer-continuation-release-pairs",
            Self::FischerCondensed => "fischer-condensed-pairs",
            Self::FischerDirect => "fischer-direct-pairs",
            Self::FischerHybrid => "fischer-hybrid-pairs",
            Self::FischerAdaptiveDirect => "fischer-adaptive-direct-pairs",
            Self::FischerAdaptiveHybrid => "fischer-adaptive-hybrid-pairs",
            Self::FischerConditioned => "fischer-conditioned-pairs",
            Self::FischerConditionedProtected => "fischer-conditioned-protected-pairs",
            Self::FischerConditionedProtectedLow => "fischer-conditioned-protected-low-pairs",
            Self::FischerConditionedStalled => "fischer-conditioned-stalled-pairs",
            Self::FischerConditionedProtectedStalled => {
                "fischer-conditioned-protected-stalled-pairs"
            }
            Self::FischerConditionedProtectedWide256 => {
                "fischer-conditioned-protected-stalled-256-pairs"
            }
            Self::FischerConditionedProtectedCached256 => {
                "fischer-conditioned-protected-stalled-cached-256-pairs"
            }
            Self::FischerConditionedProtectedParallel256 => {
                "fischer-conditioned-protected-stalled-parallel-256-pairs"
            }
            Self::FischerConditionedProtectedVolatile256 => {
                "fischer-conditioned-protected-stalled-volatile-256-pairs"
            }
            Self::FischerConditionedProtectedSchur256 => {
                "fischer-conditioned-protected-stalled-schur-256-pairs"
            }
            Self::FischerConditionedProtectedNorm256 => {
                "fischer-conditioned-protected-stalled-schur-norm-256-pairs"
            }
            Self::FischerConditionedProtectedActivePublication256 => {
                "fischer-conditioned-protected-stalled-schur-norm-active-publication-256-pairs"
            }
            Self::FischerConditionedProtectedPublicationProducts256 => {
                "fischer-conditioned-protected-stalled-schur-norm-publication-products-256-pairs"
            }
            Self::FischerConditionedProtectedRetainedLocal256 => {
                "fischer-conditioned-protected-stalled-schur-norm-publication-products-retained-local-256-pairs"
            }
            Self::FischerConditionedProtectedProjectionCache256 => {
                "fischer-conditioned-protected-stalled-schur-norm-publication-products-retained-local-projection-cache-256-pairs"
            }
            Self::FischerConditionedProtectedSharedCore256 => {
                "fischer-conditioned-protected-stalled-schur-norm-publication-products-retained-local-shared-core-256-pairs"
            }
            Self::FischerConditionedProtectedStreamBlock256 => {
                "fischer-conditioned-protected-stalled-schur-norm-publication-products-retained-local-shared-core-stream-block-256-pairs"
            }
            Self::FischerConditionedProtectedGroupedSum256 => {
                "fischer-conditioned-protected-stalled-schur-norm-publication-products-retained-local-shared-core-stream-block-stagnant-branch-grouped-sum-256-pairs"
            }
            Self::FischerConditionedProtectedGroupedEight256 => {
                "fischer-conditioned-protected-stalled-schur-norm-publication-products-retained-local-shared-core-stream-block-stagnant-branch-grouped-eight-256-pairs"
            }
            Self::FischerConditionedProtectedParallelIndices256 => {
                "fischer-conditioned-protected-stalled-schur-norm-publication-products-retained-local-shared-core-stream-block-stagnant-branch-grouped-eight-parallel-indices-256-pairs"
            }
            Self::FischerConditionedProtectedSoaScratch256 => {
                "fischer-conditioned-protected-stalled-schur-norm-publication-products-retained-local-shared-core-stream-block-stagnant-branch-grouped-eight-parallel-indices-soa-scratch-256-pairs"
            }
            Self::FischerConditionedProtectedStagnantBranch256 => {
                "fischer-conditioned-protected-stalled-schur-norm-publication-products-retained-local-shared-core-stream-block-stagnant-branch-256-pairs"
            }
        }
    }
}

fn shader_source(method: NewtonFailureMethod) -> String {
    include_str!("../shaders/soil_newton_failure.wgsl").replace(
        "const METHOD_TAG:u32=0u;",
        &format!("const METHOD_TAG:u32={}u;", method as u32),
    )
}

#[cfg(test)]
fn success_source(method: NewtonFailureMethod) -> String {
    shader_source(method)
        .replace("if report[0]==0u {if c==0u {packet[0]=0u;}return;}",
            "if report[0]!=0u || publication[0]!=0u || publication[2]==0u || report[1]==0u {return;}")
        .replace("for(var i=8u;i<16u;i++) {packet[i]=0u;}",
            "for(var i=8u;i<16u;i++) {packet[i]=0u;}packet[8]=1u;")
}

#[cfg(test)]
fn success_seed_source(method: NewtonFailureMethod) -> String {
    success_source(method)
        .replace("packet[8]=1u;", "packet[8]=1u;packet[9]=1u;")
        .replace("@compute @workgroup_size(64)",
            "@group(0) @binding(12) var<storage,read> initial:array<u32>;\n@group(0) @binding(13) var<storage,read_write> saved_initial:array<u32>;\n@compute @workgroup_size(64)")
        .replace("let base=32u+n*n+2u*n+c*(8u*n+32u);",
            "for(var i=0u;i<16u;i++) {saved_initial[c*16u+i]=initial[c*16u+i];}\nlet base=32u+n*n+2u*n+c*(8u*n+32u);")
}

#[cfg(test)]
struct SuccessSeed {
    source: wgpu::Buffer,
    initial: wgpu::Buffer,
    saved: wgpu::Buffer,
}

pub(crate) struct NewtonFailureCapture {
    pipeline: wgpu::ComputePipeline,
    group: wgpu::BindGroup,
    packet: wgpu::Buffer,
    groups: u32,
    #[cfg(test)]
    seed: Option<SuccessSeed>,
}

fn invalid() -> Error {
    Error::Solver("Newton failure capture: invalid capacity, inputs or device limit".into())
}

impl NewtonFailureCapture {
    pub(crate) fn required_bytes(n: u32, c: u32) -> Result<u64> {
        if !(1..=32).contains(&n) || c > 16384 {
            return Err(invalid());
        }
        let n = u64::from(n);
        let c = u64::from(c.max(1));
        Ok(144 + 4 * n * n + 8 * n + (32 * n + 128) * c)
    }

    /// Sources: mass, paired rates, rows, response, contacts, state, report,
    /// publication, preparation and common status. Dimensions include solve limits.
    #[cfg(test)]
    pub(crate) fn new(
        engine: &Engine,
        sources: [&wgpu::Buffer; 10],
        dimensions: [u32; 4],
    ) -> Result<Self> {
        Self::new_with_method(
            engine,
            sources,
            dimensions,
            NewtonFailureMethod::Unspecified,
        )
    }

    pub(crate) fn new_with_method(
        engine: &Engine,
        sources: [&wgpu::Buffer; 10],
        dimensions: [u32; 4],
        method: NewtonFailureMethod,
    ) -> Result<Self> {
        #[cfg(test)]
        let _setup = crate::soil_setup_trace::stage("newton.capture");
        engine.check_device()?;
        let [n, c, newton, krylov] = dimensions;
        let bytes = Self::required_bytes(n, c)?;
        if !(1..=64).contains(&newton) || !(1..=64).contains(&krylov) {
            return Err(invalid());
        }
        let device = engine.device().ok_or_else(invalid)?;
        let limits = device.limits();
        #[cfg(test)]
        let seed_capture = cfg!(feature = "warm-newton")
            && std::env::var("MOLLA_NEWTON_CAPTURE_SUCCESS").as_deref() == Ok("1");
        #[cfg(test)]
        if seed_capture
            && (limits.max_storage_buffers_per_shader_stage < 13
                || limits.max_bindings_per_bind_group < 14
                || !sources[5].usage().contains(wgpu::BufferUsages::COPY_SRC))
        {
            return Err(invalid());
        }
        let width = u64::from(n);
        let slots = u64::from(c.max(1));
        let sizes = [
            4 * width * width,
            8 * width,
            16 * slots * width,
            16 * slots * width,
            64 * slots,
            64 * slots,
            32,
            32,
            4,
            4,
        ];
        if limits.max_storage_buffers_per_shader_stage < 11
            || limits.max_bindings_per_bind_group < 12
            || limits.max_uniform_buffers_per_shader_stage < 1
            || limits.max_compute_invocations_per_workgroup < 64
            || limits.max_compute_workgroup_size_x < 64
            || c.max(1).div_ceil(64) > limits.max_compute_workgroups_per_dimension
            || bytes - 16 > limits.max_buffer_size
            || bytes - 16 > limits.max_storage_buffer_binding_size
            || sources.iter().zip(sizes).any(|(b, s)| {
                b.size() < s
                    || b.size() > limits.max_storage_buffer_binding_size
                    || !b.usage().contains(wgpu::BufferUsages::STORAGE)
            })
        {
            return Err(invalid());
        }
        let packet = molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("latched Newton failure"),
                size: bytes - 16,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }
        );
        let params = molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("Newton failure dimensions"),
                size: 16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }
        );
        molla_core::gpu_write_buffer!(
            engine.queue().unwrap(),
            &params,
            0,
            bytemuck::cast_slice(&dimensions)
        );
        let source = shader_source(method);
        #[cfg(test)]
        let source = if seed_capture {
            success_seed_source(method)
        } else if std::env::var("MOLLA_NEWTON_CAPTURE_SUCCESS").as_deref() == Ok("1") {
            success_source(method)
        } else {
            source
        };
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Newton failure capture"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Newton failure capture"),
            layout: None,
            module: &module,
            entry_point: Some("capture"),
            compilation_options: Default::default(),
            cache: None,
        });
        #[cfg(test)]
        let seed = seed_capture.then(|| {
            let create = |label| {
                molla_core::gpu_create_buffer!(
                    device,
                    &wgpu::BufferDescriptor {
                        label: Some(label),
                        size: 64 * slots,
                        usage: wgpu::BufferUsages::STORAGE
                            | wgpu::BufferUsages::COPY_SRC
                            | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    }
                )
            };
            SuccessSeed {
                source: sources[5].clone(),
                initial: create("Newton success pending initial guess"),
                saved: create("Newton success latched initial guess"),
            }
        });
        let bindings = sources.into_iter().chain([&packet, &params]);
        #[cfg(test)]
        let bindings = bindings.chain(seed.iter().flat_map(|s| [&s.initial, &s.saved]));
        let bindings: Vec<_> = bindings
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.as_entire_binding(),
            })
            .collect();
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Newton failure capture"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &bindings,
        });
        Ok(Self {
            pipeline,
            group,
            packet,
            groups: c.max(1).div_ceil(64),
            #[cfg(test)]
            seed,
        })
    }

    #[cfg(test)]
    pub(crate) fn encode_initial(&self, encoder: &mut wgpu::CommandEncoder) {
        if let Some(seed) = &self.seed {
            encoder.copy_buffer_to_buffer(&seed.source, 0, &seed.initial, 0, seed.initial.size());
        }
    }

    #[cfg(test)]
    pub(crate) fn success_seed_buffer(&self) -> Option<&wgpu::Buffer> {
        self.seed.as_ref().map(|s| &s.saved)
    }

    pub(crate) fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.group, &[]);
        pass.dispatch_workgroups(self.groups, 1, 1);
        molla_core::gpu_metrics::record_dispatch();
    }

    pub(crate) fn buffer(&self) -> &wgpu::Buffer {
        &self.packet
    }
}

#[cfg(test)]
pub(crate) fn capture_failure(
    engine: &Engine,
    buffer: &wgpu::Buffer,
    path: &std::path::Path,
    map: impl FnOnce(&wgpu::Buffer) -> Result<()>,
) -> Result<()> {
    capture_packet(engine, buffer, path, map, false, None)
}

pub(crate) fn capture_failure_with_seed(
    engine: &Engine,
    buffer: &wgpu::Buffer,
    seed: Option<&wgpu::Buffer>,
    path: &std::path::Path,
    map: impl FnOnce(&wgpu::Buffer) -> Result<()>,
) -> Result<()> {
    capture_packet(engine, buffer, path, map, false, seed)
}

#[cfg(test)]
pub(crate) fn capture_success(
    engine: &Engine,
    buffer: &wgpu::Buffer,
    path: &std::path::Path,
    map: impl FnOnce(&wgpu::Buffer) -> Result<()>,
) -> Result<()> {
    capture_packet(engine, buffer, path, map, true, None)
}

#[cfg(test)]
pub(crate) fn capture_success_with_seed(
    engine: &Engine,
    buffer: &wgpu::Buffer,
    seed: Option<&wgpu::Buffer>,
    path: &std::path::Path,
    map: impl FnOnce(&wgpu::Buffer) -> Result<()>,
) -> Result<()> {
    capture_packet(engine, buffer, path, map, true, seed)
}

fn capture_packet(
    engine: &Engine,
    buffer: &wgpu::Buffer,
    path: &std::path::Path,
    map: impl FnOnce(&wgpu::Buffer) -> Result<()>,
    success: bool,
    seed: Option<&wgpu::Buffer>,
) -> Result<()> {
    use std::io::Write;
    let device = engine.device().ok_or_else(invalid)?;
    let staging = molla_core::gpu_create_buffer!(
        device,
        &wgpu::BufferDescriptor {
            label: Some("Newton failure readback"),
            size: buffer.size() + seed.map_or(0, |b| b.size()),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        }
    );
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, buffer.size());
    if let Some(seed) = seed {
        encoder.copy_buffer_to_buffer(seed, 0, &staging, buffer.size(), seed.size());
    }
    engine.queue().unwrap().submit([encoder.finish()]);
    map(&staging)?;
    let mapped = molla_core::gpu_readback!(staging.slice(..));
    let bytes = mapped[..buffer.size() as usize].to_vec();
    let seed_bytes = mapped[buffer.size() as usize..].to_vec();
    drop(mapped);
    staging.unmap();
    if bytes.len() < 128 || bytes[..4] != 1u32.to_le_bytes() {
        return Err(invalid());
    }
    let word = |i: usize| u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap());
    if word(3) != 0
        || word(4) != 0
        || word(8) != u32::from(success)
        || (success && word(9) != u32::from(seed.is_some()))
        || if success {
            word(16) != 0 || word(24) != 0 || word(26) == 0 || word(17) == 0
        } else {
            word(16) == 0
        }
    {
        return Err(invalid());
    }
    let method = NewtonFailureMethod::from_tag(word(7))?.name();
    let certificate = certificate::Certificate::from_packet(&bytes)?;
    if seed.is_some() && seed_bytes.len() != 64 * word(2).max(1) as usize {
        return Err(invalid());
    }
    let stage = if success {
        "newton-success"
    } else {
        "newton-solve"
    };
    let io = |e: std::io::Error| Error::Solver(format!("Newton failure capture: {e}"));
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
    write("newton.bin", &bytes)?;
    let seed_meta = if seed.is_some() {
        write("seed.bin", &seed_bytes)?;
        format!(
            r#", "initial_guess":"retained-seed", "seed_bytes":{}, "seed_blake3":"{}""#,
            seed_bytes.len(),
            blake3::hash(&seed_bytes).to_hex()
        )
    } else {
        String::new()
    };
    write("capture.json",format!(r#"{{"schema":1,"stage":"{stage}","acceptance":false,"byte_order":"little","method":"{method}","certificate_version":{},"certificate":"{}","dofs":{},"contacts":{},"newton_iterations":{},"krylov_iterations":{},"diagnostic_inner_limit":{},"fault":{},"bytes":{},"blake3":"{}"{seed_meta}}}"#,certificate.tag(),certificate.name(),word(1),word(2),word(5),word(6),word(23),word(16),bytes.len(),blake3::hash(&bytes).to_hex()).as_bytes())
}

#[cfg(test)]
pub(crate) fn replay_seed(
    path: &std::path::Path,
    meta: &serde_json::Value,
    contacts: usize,
) -> Result<Option<Vec<u8>>> {
    let file = path.join("seed.bin");
    let Some(hash) = meta["seed_blake3"].as_str() else {
        return if meta.get("initial_guess").is_some()
            || meta.get("seed_bytes").is_some()
            || file.exists()
        {
            Err(invalid())
        } else {
            Ok(None)
        };
    };
    if contacts > 16384
        || meta["initial_guess"] != "retained-seed"
        || meta["seed_bytes"].as_u64() != Some(64 * contacts.max(1) as u64)
    {
        return Err(invalid());
    }
    let expected = 64 * contacts.max(1);
    if std::fs::metadata(&file).map_err(|_| invalid())?.len() != expected as u64 {
        return Err(invalid());
    }
    let bytes = std::fs::read(file).map_err(|_| invalid())?;
    if bytes.len() != expected
        || blake3::hash(&bytes).to_hex().as_str() != hash
        || bytes.chunks_exact(4).enumerate().any(|(i, b)| {
            let value = f32::from_le_bytes(b.try_into().unwrap());
            !value.is_finite() || (i % 4 == 3 && value != 0.0)
        })
    {
        return Err(invalid());
    }
    Ok(Some(bytes))
}

#[cfg(test)]
#[path = "soil_newton_failure_tests.rs"]
mod tests;
