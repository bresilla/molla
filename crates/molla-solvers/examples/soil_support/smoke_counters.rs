#[cfg(feature = "resource-counters")]
use molla_core::gpu_metrics::{DeviceScope, DispatchScope, TransferScope};

pub struct Scope {
    #[cfg(feature = "resource-counters")]
    resources: DeviceScope,
    #[cfg(feature = "resource-counters")]
    transfers: TransferScope,
    #[cfg(feature = "resource-counters")]
    dispatches: DispatchScope,
}

pub struct Counts {
    pub allocations: Option<u64>,
    pub readbacks: Option<u64>,
    pub uploads: Option<u64>,
    pub dispatches: Option<u64>,
}

pub fn verify_build() -> Result<(), Box<dyn std::error::Error>> {
    let expected = cfg!(feature = "resource-counters");
    let counters = wgpu::InternalCounters::default();
    counters.hal.buffers.add(1);
    let wgpu_enabled = counters.hal.buffers.read() == 1;
    let core_enabled = molla_core::gpu_metrics::ENABLED;
    if core_enabled != expected || wgpu_enabled != expected {
        return Err("resource-counter feature unification does not match smoke build".into());
    }
    println!("instrumentation_features core={core_enabled} wgpu={wgpu_enabled}");
    Ok(())
}

impl Scope {
    pub fn start() -> molla_core::Result<Self> {
        Ok(Self {
            #[cfg(feature = "resource-counters")]
            resources: DeviceScope::start()?,
            #[cfg(feature = "resource-counters")]
            transfers: TransferScope::start()?,
            #[cfg(feature = "resource-counters")]
            dispatches: DispatchScope::start()?,
        })
    }

    pub fn finish(self) -> molla_core::Result<Counts> {
        #[cfg(feature = "resource-counters")]
        {
            let resources = self.resources.finish();
            let transfers = self.transfers.finish();
            let dispatches = self.dispatches.finish();
            if resources.invalid || transfers.overflowed || dispatches.overflowed {
                return Err(molla_core::Error::Solver(
                    "invalid instrumentation counters".into(),
                ));
            }
            Ok(Counts {
                allocations: Some(resources.observations),
                readbacks: Some(transfers.readback_bytes),
                uploads: Some(transfers.upload_bytes),
                dispatches: Some(dispatches.encoded_dispatches),
            })
        }
        #[cfg(not(feature = "resource-counters"))]
        {
            Ok(Counts {
                allocations: None,
                readbacks: None,
                uploads: None,
                dispatches: None,
            })
        }
    }
}

impl Counts {
    pub fn failed(&self) -> bool {
        self.allocations.is_some_and(|v| v != 0)
            || self.readbacks.is_some_and(|v| v != 0)
            || self.dispatches == Some(0)
    }
}

#[test]
fn smoke_counter_build_matches_metadata() {
    verify_build().unwrap();
    let counts = Scope::start().unwrap().finish().unwrap();
    assert_eq!(
        counts.allocations.is_some(),
        cfg!(feature = "resource-counters")
    );
    assert_eq!(
        counts.readbacks.is_some(),
        cfg!(feature = "resource-counters")
    );
    assert_eq!(
        counts.uploads.is_some(),
        cfg!(feature = "resource-counters")
    );
    assert_eq!(
        counts.dispatches.is_some(),
        cfg!(feature = "resource-counters")
    );
    assert_eq!(counts.failed(), cfg!(feature = "resource-counters"));
}
