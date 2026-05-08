//! `Engine` — owns the wgpu runtime.

use std::sync::Arc;

use crate::core::{Error, Result};

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
    fn from_wgpu(info: wgpu::AdapterInfo) -> Self {
        Self {
            vendor: info.vendor,
            name: info.name,
            backend: format!("{:?}", info.backend),
            is_software: matches!(info.device_type, wgpu::DeviceType::Cpu),
            driver: info.driver,
        }
    }
}

/// The runtime that owns the `wgpu::Device` / `Queue` and dispatches
/// kernels.
///
/// At stage 0, `Engine` only stands up the runtime and reports adapter
/// info. `Engine::step` and the kernel dispatch surface land in stage 2.
pub struct Engine {
    backend: Backend,
    adapter_info: Option<AdapterInfo>,
    device: Option<Arc<wgpu::Device>>,
    queue: Option<Arc<wgpu::Queue>>,
}

impl Engine {
    /// Construct a CPU-only engine. No `wgpu` resources are allocated.
    #[must_use]
    pub fn new_cpu() -> Self {
        Self {
            backend: Backend::Cpu,
            adapter_info: None,
            device: None,
            queue: None,
        }
    }

    /// Construct a GPU engine, picking whatever adapter `wgpu` finds.
    pub fn new_gpu() -> Result<Self> {
        Self::new(Backend::AutoGpu)
    }

    /// Construct an engine with the given backend.
    pub fn new(backend: Backend) -> Result<Self> {
        match backend {
            Backend::Cpu => Ok(Self::new_cpu()),
            Backend::AutoGpu => init_gpu(wgpu::Backends::PRIMARY),
            Backend::Gpu(b) => init_gpu(b),
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
}

fn init_gpu(backends: wgpu::Backends) -> Result<Engine> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends,
        ..Default::default()
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

    Ok(Engine {
        backend: Backend::Gpu(backends),
        adapter_info: Some(adapter_info),
        device: Some(Arc::new(device)),
        queue: Some(Arc::new(queue)),
    })
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
