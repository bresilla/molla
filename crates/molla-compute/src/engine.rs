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
}

impl AdapterInfo {
    pub fn from_wgpu(info: wgpu::AdapterInfo) -> Self {
        Self {
            vendor: info.vendor,
            name: info.name,
            backend: format!("{:?}", info.backend),
            is_software: matches!(info.device_type, wgpu::DeviceType::Cpu),
            driver: info.driver,
        }
    }
}

/// The runtime that owns the `wgpu::Device` / `Queue` and feeds them
/// to the kernel dispatch wrappers. Construct
/// via [`Engine::new_cpu`] / [`Engine::new_gpu`] / [`Engine::new`].
pub struct Engine {
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

    /// Construct an engine with the given [`Backend`]. Use
    /// [`Self::new_cpu`] / [`Self::new_gpu`] for the common cases.
    pub fn new(backend: Backend) -> Result<Self> {
        match backend {
            Backend::Cpu => Ok(Self::new_cpu()),
            Backend::AutoGpu => init_gpu(wgpu::Backends::PRIMARY),
            Backend::Gpu(b) => init_gpu(b),
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
        self.device
            .as_ref()
            .expect("GPU health has a device")
            .destroy();
        Ok(())
    }
}

fn init_gpu(backends: wgpu::Backends) -> Result<Engine> {
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
        required_features: wgpu::Features::empty(),
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
}
