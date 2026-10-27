//! `Engine` — owns the wgpu runtime.

use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};

use molla_core::{Error, Result};

/// Backend selector for [`Engine::new`].
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Backend {
    /// Pure CPU — no `wgpu` resources allocated. Useful for tests and for
    /// the CPU-fallback paths that every solver must provide.
    Cpu,
    /// Pick whatever GPU backend `wgpu` finds best on this host.
    AutoGpu,
    /// Force a specific `wgpu` backend.
    Gpu(wgpu::Backends),
    /// A device and queue adopted from another owner through
    /// [`Engine::from_device`]; never selectable through [`Engine::new`].
    External,
}

/// Human-readable info about the selected adapter. Useful in `--version`
/// output and as a sanity check at startup.
#[derive(Clone, Debug)]
pub struct AdapterInfo {
    /// Vendor PCI id, when reported by the driver.
    pub vendor: u32,
    /// Device name as reported by the driver.
    pub name: String,
    /// Backend in use (`"Vulkan"`, `"Metal"`, `"Dx12"`, `"Gl"`, `"BrowserWebGpu"`).
    pub backend: String,
    /// Whether this adapter is a real GPU or a software rasteriser.
    pub is_software: bool,
    /// Driver name, when reported.
    pub driver: String,
    /// Driver version and implementation details, when reported.
    pub driver_info: String,
}

impl AdapterInfo {
    pub fn from_wgpu(info: wgpu::AdapterInfo) -> Self {
        Self {
            vendor: info.vendor,
            name: info.name,
            backend: format!("{:?}", info.backend),
            is_software: matches!(info.device_type, wgpu::DeviceType::Cpu),
            driver: info.driver,
            driver_info: info.driver_info,
        }
    }
}

/// The runtime that owns the `wgpu::Device` / `Queue` and feeds them
/// to the kernel dispatch wrappers. Construct
/// via [`Engine::new_cpu`] / [`Engine::new_gpu`] / [`Engine::new`].
pub struct Engine {
    programs: crate::program_cache::ProgramCache,
    pipeline_cache: Option<wgpu::PipelineCache>,
    backend: Backend,
    adapter_info: Option<AdapterInfo>,
    device: Option<Arc<wgpu::Device>>,
    queue: Option<Arc<wgpu::Queue>>,
    health: Option<Arc<DeviceHealth>>,
}

/// Sticky first-loss diagnostic for one logical GPU device.
#[derive(Debug, Clone)]
pub struct DeviceLoss {
    pub reason: wgpu::DeviceLostReason,
    pub message: String,
}

/// Shared loss signal without references to GPU resources.
#[derive(Default)]
pub struct DeviceHealth {
    lost: AtomicBool,
    loss: Mutex<Option<DeviceLoss>>,
}

impl DeviceHealth {
    pub fn is_lost(&self) -> bool {
        self.lost.load(Ordering::Acquire)
    }

    pub fn loss(&self) -> Option<DeviceLoss> {
        if !self.is_lost() {
            return None;
        }
        self.loss.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    pub fn check(&self) -> Result<()> {
        if let Some(loss) = self.loss() {
            return Err(Error::Gpu(format!(
                "GPU device lost ({:?}): {}",
                loss.reason, loss.message
            )));
        }
        Ok(())
    }

    /// Records the first loss; later calls keep the original diagnostic.
    pub fn record(&self, reason: wgpu::DeviceLostReason, message: String) {
        let mut loss = self.loss.lock().unwrap_or_else(|e| e.into_inner());
        if loss.is_none() {
            *loss = Some(DeviceLoss { reason, message });
        }
        self.lost.store(true, Ordering::Release);
    }
}

impl Engine {
    /// Construct a CPU-only engine. No `wgpu` resources are allocated.
    ///
    /// # Examples
    ///
    /// ```
    /// use molla_compute::Engine;
    /// let engine = Engine::new_cpu();
    /// assert!(engine.device().is_none());
    /// ```
    #[must_use]
    pub fn new_cpu() -> Self {
        Self {
            backend: Backend::Cpu,
            programs: Default::default(),
            pipeline_cache: None,
            adapter_info: None,
            device: None,
            queue: None,
            health: None,
        }
    }

    /// Construct a GPU engine, picking whatever adapter `wgpu` finds.
    /// Returns [`Error::Gpu`] if no compatible adapter is available.
    pub fn new_gpu() -> Result<Self> {
        Self::new(Backend::AutoGpu)
    }

    /// Construct a GPU engine requiring the specified device features; unsupported features fail.
    pub fn new_gpu_with_features(features: wgpu::Features) -> Result<Self> {
        init_gpu(wgpu::Backends::PRIMARY, features)
    }

    /// Construct an engine with the given [`Backend`]. Use
    /// [`Self::new_cpu`] / [`Self::new_gpu`] for the common cases.
    pub fn new(backend: Backend) -> Result<Self> {
        match backend {
            Backend::Cpu => Ok(Self::new_cpu()),
            Backend::AutoGpu => init_gpu(wgpu::Backends::PRIMARY, wgpu::Features::empty()),
            Backend::Gpu(b) => init_gpu(b, wgpu::Features::empty()),
            Backend::External => Err(Error::Gpu(
                "external backend: construct with Engine::from_device".into(),
            )),
        }
    }

    /// Adopt a device and queue owned elsewhere, such as a renderer.
    ///
    /// The owner keeps its own device-lost callback; this engine's
    /// [`DeviceHealth`] only reports what the owner forwards through
    /// [`DeviceHealth::record`]. Every Molla resource created through this
    /// engine belongs to the adopted device and must be dropped before it.
    pub fn from_device(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        adapter_info: Option<AdapterInfo>,
    ) -> Self {
        Self {
            backend: Backend::External,
            programs: Default::default(),
            pipeline_cache: None,
            adapter_info,
            device: Some(device),
            queue: Some(queue),
            health: Some(Arc::new(DeviceHealth::default())),
        }
    }

    /// Selected backend.
    #[must_use]
    pub fn backend(&self) -> Backend {
        self.backend
    }

    /// Adapter info if a GPU backend was selected.
    #[must_use]
    pub fn adapter_info(&self) -> Option<&AdapterInfo> {
        self.adapter_info.as_ref()
    }

    /// Owned reference to the `wgpu::Device`, if a GPU backend is active.
    #[must_use]
    pub fn device(&self) -> Option<&Arc<wgpu::Device>> {
        self.device.as_ref()
    }

    /// Owned reference to the `wgpu::Queue`, if a GPU backend is active.
    #[must_use]
    pub fn queue(&self) -> Option<&Arc<wgpu::Queue>> {
        self.queue.as_ref()
    }

    /// Observed device loss; raw wgpu loss callbacks may require polling for delivery.
    pub fn device_health(&self) -> Option<&Arc<DeviceHealth>> {
        self.health.as_ref()
    }

    pub fn check_device(&self) -> Result<()> {
        self.health
            .as_ref()
            .ok_or_else(|| Error::Gpu("engine has no GPU device".into()))?
            .check()
    }

    /// Enable compiler caching on a device requested with `PIPELINE_CACHE`.
    /// Existing programs remain valid; only subsequent pipeline builds use this cache.
    /// Incompatible snapshots fall back to empty caches. No disk I/O is performed.
    pub fn enable_pipeline_cache(
        &mut self,
        snapshot: Option<&crate::PipelineCacheSnapshot>,
    ) -> Result<()> {
        self.check_device()?;
        let device = self.device.as_ref().expect("GPU health has a device");
        if !device.features().contains(wgpu::Features::PIPELINE_CACHE) {
            return Err(Error::Gpu(
                "PIPELINE_CACHE feature was not requested".into(),
            ));
        }
        let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let allocation = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let internal = device.push_error_scope(wgpu::ErrorFilter::Internal);
        // Snapshot bytes originate from get_data or the unsafe import contract.
        let cache = unsafe {
            device.create_pipeline_cache(&wgpu::PipelineCacheDescriptor {
                label: Some("molla::compiler cache"),
                data: snapshot.map(|snapshot| snapshot.as_bytes()),
                fallback: true,
            })
        };
        let errors = [
            pollster::block_on(internal.pop()),
            pollster::block_on(allocation.pop()),
            pollster::block_on(validation.pop()),
        ];
        self.check_device()?;
        if let Some(error) = errors.into_iter().flatten().next() {
            return Err(Error::Gpu(format!(
                "GPU compiler cache initialization: {error}"
            )));
        }
        self.pipeline_cache = Some(cache);
        Ok(())
    }

    /// Compiler cache for pipelines created on this engine's device.
    pub fn pipeline_cache(&self) -> Option<&wgpu::PipelineCache> {
        self.pipeline_cache.as_ref()
    }

    /// Export driver compilation data, not physical state or a cache-hit report.
    /// This explicit setup/maintenance operation may block and allocate host memory.
    pub fn pipeline_cache_snapshot(&self) -> Result<Option<crate::PipelineCacheSnapshot>> {
        self.check_device()?;
        let snapshot = self
            .pipeline_cache
            .as_ref()
            .and_then(|cache| cache.get_data())
            .map(|bytes| crate::PipelineCacheSnapshot { bytes });
        self.check_device()?;
        Ok(snapshot)
    }

    /// Stop supplying compiler data to future builds; existing programs remain valid.
    pub fn disable_pipeline_cache(&mut self) {
        self.pipeline_cache = None;
    }

    /// Retain up to 64 immutable GPU programs by type and compiler-input variant.
    /// Up to two different keys initialize concurrently; each key has one initializer.
    /// Initializers must not reenter this engine's cache.
    /// Programs contain pipelines/layouts, not model/state buffers. Failed builds are not retained.
    pub fn cached_gpu_program<T: std::any::Any + Send + Sync>(
        &self,
        variant: &str,
        create: impl FnOnce(&wgpu::Device) -> Result<T>,
    ) -> Result<Arc<T>> {
        self.check_device()?;
        let device = self.device.as_ref().expect("GPU health has a device");
        self.programs.get_or_create(variant, || {
            let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
            let allocation = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
            let internal = device.push_error_scope(wgpu::ErrorFilter::Internal);
            let program = create(device);
            let errors = [
                pollster::block_on(internal.pop()),
                pollster::block_on(allocation.pop()),
                pollster::block_on(validation.pop()),
            ];
            self.check_device()?;
            if let Some(error) = errors.into_iter().flatten().next() {
                return Err(Error::Gpu(format!("GPU program initialization: {error}")));
            }
            program
        })
    }

    /// Drop retained programs; existing kernels keep their own references.
    /// In-flight builds finish for their callers but are not retained after this clear.
    pub fn clear_gpu_program_cache(&self) {
        self.programs.clear();
    }

    /// Marks this engine's logical device lost immediately, then destroys it.
    pub fn destroy_device(&self) -> Result<()> {
        let health = self
            .health
            .as_ref()
            .ok_or_else(|| Error::Gpu("engine has no GPU device".into()))?;
        health.record(
            wgpu::DeviceLostReason::Destroyed,
            "logical device explicitly destroyed".into(),
        );
        self.clear_gpu_program_cache();
        self.device
            .as_ref()
            .expect("GPU health has a device")
            .destroy();
        Ok(())
    }
}

fn init_gpu(backends: wgpu::Backends, features: wgpu::Features) -> Result<Engine> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });

    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .map_err(|e| Error::Gpu(format!("no compatible wgpu adapter: {e}")))?;

    let adapter_info = AdapterInfo::from_wgpu(adapter.get_info());

    // Request whatever this adapter natively supports — `downlevel_defaults`
    // caps storage buffers at 4 per stage, which we exceed in the rigid-body
    // integrator (7 model + 5 state + 1 control = 13). Modern desktop GPUs
    // typically allow 16+, mobile / web much less; if we ever target the
    // browser we'll need to consolidate buffers.
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("molla::compute::Engine"),
        required_features: features,
        required_limits: adapter.limits(),
        experimental_features: wgpu::ExperimentalFeatures::default(),
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
    }))
    .map_err(|e| Error::Gpu(format!("device request failed: {e}")))?;

    let health = Arc::new(DeviceHealth::default());
    let signal = health.clone();
    device.set_device_lost_callback(move |reason, message| signal.record(reason, message));
    Ok(Engine {
        programs: Default::default(),
        pipeline_cache: None,
        backend: Backend::Gpu(backends),
        adapter_info: Some(adapter_info),
        device: Some(Arc::new(device)),
        queue: Some(Arc::new(queue)),
        health: Some(health),
    })
}

#[cfg(test)]
mod loss_tests {
    use super::*;

    #[test]
    fn external_backend_is_not_selectable() {
        assert!(Engine::new(Backend::External).is_err());
    }

    #[test]
    fn cpu_engine_has_no_device_loss_signal() {
        let engine = Engine::new_cpu();
        assert!(engine.device_health().is_none());
        assert!(engine.check_device().is_err());
        assert!(engine.destroy_device().is_err());
    }

    #[test]
    fn loss_is_sticky_and_keeps_the_first_diagnostic() {
        let health = Arc::new(DeviceHealth::default());
        assert!(health.check().is_ok());
        assert!(health.loss().is_none());
        let signal = health.clone();
        std::thread::spawn(move || {
            signal.record(wgpu::DeviceLostReason::Unknown, "first loss".into())
        })
        .join()
        .unwrap();
        health.record(
            wgpu::DeviceLostReason::Destroyed,
            "later destruction".into(),
        );
        assert!(health.is_lost());
        assert_eq!(health.loss().unwrap().message, "first loss");
        assert!(
            health
                .check()
                .unwrap_err()
                .to_string()
                .contains("first loss")
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_engine_reports_no_adapter() {
        let e = Engine::new_cpu();
        assert_eq!(e.backend(), Backend::Cpu);
        assert!(e.adapter_info().is_none());
        assert!(e.device().is_none());
        assert!(e.queue().is_none());
    }

    #[test]
    fn cpu_engine_does_not_initialize_cached_gpu_programs() {
        let e = Engine::new_cpu();
        assert!(
            e.cached_gpu_program::<u32>("cpu", |_| panic!("CPU used GPU cache"))
                .is_err()
        );
        e.clear_gpu_program_cache();
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    #[ignore = "requires GPU; run via make test-primitives-program-cache-gpu"]
    fn gpu_program_cache_retains_valid_objects_without_crossing_engines() {
        let a = Engine::new_gpu().unwrap();
        let source = "@compute @workgroup_size(1) fn main() {}";
        let first = a
            .cached_gpu_program("same", |device| {
                Ok(device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("cache identity"),
                    source: wgpu::ShaderSource::Wgsl(source.into()),
                }))
            })
            .unwrap();
        let weak = Arc::downgrade(&first);
        drop(first);
        let retained = a
            .cached_gpu_program::<wgpu::ShaderModule>("same", |_| panic!("cache recompiled"))
            .unwrap();
        assert!(Arc::ptr_eq(&retained, &weak.upgrade().unwrap()));
        let b = Engine::new_gpu().unwrap();
        let other = b
            .cached_gpu_program("same", |device| {
                Ok(device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("other cache"),
                    source: wgpu::ShaderSource::Wgsl(source.into()),
                }))
            })
            .unwrap();
        assert!(!Arc::ptr_eq(&retained, &other));
        drop(retained);
        a.destroy_device().unwrap();
        assert!(weak.upgrade().is_none());
        assert!(
            a.cached_gpu_program::<wgpu::ShaderModule>("same", |_| panic!("lost device compiled"))
                .is_err()
        );
        assert!(b.check_device().is_ok());
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    #[ignore = "requires GPU; run via make test-primitives-program-cache-gpu"]
    fn gpu_program_cache_surfaces_validation_and_retries_failed_variants() {
        let engine = Engine::new_gpu().unwrap();
        let device = engine.device().unwrap();
        let outer = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let failed = engine.cached_gpu_program("retry", |device| {
            Ok(device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("invalid cache fixture"),
                source: wgpu::ShaderSource::Wgsl("invalid WGSL".into()),
            }))
        });
        assert!(
            failed
                .err()
                .unwrap()
                .to_string()
                .contains("GPU program initialization")
        );
        assert!(pollster::block_on(outer.pop()).is_none());
        let valid = engine
            .cached_gpu_program("retry", |device| {
                Ok(device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("valid retry"),
                    source: wgpu::ShaderSource::Wgsl(
                        "@compute @workgroup_size(1) fn main() {}".into(),
                    ),
                }))
            })
            .unwrap();
        let again = engine
            .cached_gpu_program::<wgpu::ShaderModule>("retry", |_| panic!("valid variant rebuilt"))
            .unwrap();
        assert!(Arc::ptr_eq(&valid, &again));
    }

    #[test]
    #[ignore = "requires GPU; concurrent cache initializers and thread-local validation scopes"]
    fn gpu_program_cache_isolates_concurrent_validation_errors() {
        let engine = Engine::new_gpu().unwrap();
        let barrier = std::sync::Barrier::new(2);
        let build = |bad| {
            engine.cached_gpu_program(if bad { "bad" } else { "good" }, |device| {
                barrier.wait();
                Ok(device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("parallel cache validation"),
                    source: wgpu::ShaderSource::Wgsl(
                        if bad {
                            "invalid shader"
                        } else {
                            "@compute @workgroup_size(1) fn main() {}"
                        }
                        .into(),
                    ),
                }))
            })
        };
        let (bad, good) = std::thread::scope(|scope| {
            let bad = scope.spawn(|| build(true));
            let good = scope.spawn(|| build(false));
            (bad.join().unwrap(), good.join().unwrap())
        });
        assert!(
            bad.unwrap_err()
                .to_string()
                .contains("GPU program initialization")
        );
        let good = good.unwrap();
        assert!(Arc::ptr_eq(
            &good,
            &engine
                .cached_gpu_program::<wgpu::ShaderModule>("good", |_| panic!(
                    "valid concurrent build lost"
                ))
                .unwrap()
        ));
        engine
            .cached_gpu_program("bad", |device| {
                Ok(device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("parallel failure retry"),
                    source: wgpu::ShaderSource::Wgsl(
                        "@compute @workgroup_size(1) fn main() {}".into(),
                    ),
                }))
            })
            .unwrap();
        engine.check_device().unwrap();
    }
}
