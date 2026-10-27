//! Retained native device and queue identities for soil resources.

/// Reject unavailable or different native devices before preparing shared soil resources.
pub fn validate_soil_device(
    device: &wgpu::Device,
    source: &wgpu::Device,
) -> molla_core::Result<()> {
    let identity = device_identity(device)
        .ok_or_else(|| molla_core::Error::Gpu("soil native device identity unavailable".into()))?;
    if device_identity(source) != Some(identity) {
        return Err(molla_core::Error::Gpu(
            "soil source belongs to another device".into(),
        ));
    }
    Ok(())
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct NativeIdentity(u8, usize);

pub(crate) fn device_identity(device: &wgpu::Device) -> Option<NativeIdentity> {
    #[cfg(all(not(target_arch = "wasm32"), not(target_vendor = "apple")))]
    {
        // Reads the retained HAL object address without modifying the device.
        if let Some(guard) = unsafe { device.as_hal::<wgpu::hal::api::Vulkan>() } {
            return Some(NativeIdentity(0, std::ptr::from_ref(&*guard) as usize));
        }
    }
    #[cfg(target_vendor = "apple")]
    {
        if let Some(guard) = unsafe { device.as_hal::<wgpu::hal::api::Metal>() } {
            return Some(NativeIdentity(1, std::ptr::from_ref(&*guard) as usize));
        }
    }
    #[cfg(target_os = "windows")]
    {
        if let Some(guard) = unsafe { device.as_hal::<wgpu::hal::api::Dx12>() } {
            return Some(NativeIdentity(2, std::ptr::from_ref(&*guard) as usize));
        }
    }
    let _ = device;
    None
}

pub(crate) fn queue_identity(queue: &wgpu::Queue) -> Option<NativeIdentity> {
    #[cfg(all(not(target_arch = "wasm32"), not(target_vendor = "apple")))]
    {
        // Reads the retained HAL object address without modifying the queue.
        if let Some(guard) = unsafe { queue.as_hal::<wgpu::hal::api::Vulkan>() } {
            return Some(NativeIdentity(0, std::ptr::from_ref(&*guard) as usize));
        }
    }
    #[cfg(target_vendor = "apple")]
    {
        if let Some(guard) = unsafe { queue.as_hal::<wgpu::hal::api::Metal>() } {
            return Some(NativeIdentity(1, std::ptr::from_ref(&*guard) as usize));
        }
    }
    #[cfg(target_os = "windows")]
    {
        if let Some(guard) = unsafe { queue.as_hal::<wgpu::hal::api::Dx12>() } {
            return Some(NativeIdentity(2, std::ptr::from_ref(&*guard) as usize));
        }
    }
    let _ = queue;
    None
}
