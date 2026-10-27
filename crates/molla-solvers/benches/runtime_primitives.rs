//! Live RigidWorld primitive timing; timestamps span the device timeline, including queue idle.

#[path = "primitives/allocations.rs"]
mod allocations;
#[path = "primitives/fixtures.rs"]
mod fixtures;
#[path = "primitives/process.rs"]
mod process;
#[path = "primitives/settings.rs"]
mod settings;
#[path = "primitives/timing.rs"]
mod timing;
#[path = "primitives/validation.rs"]
mod validation;

#[global_allocator]
static ALLOCATOR: allocations::CountingAllocator = allocations::CountingAllocator;

use molla_compute::Engine;
use molla_core::gpu_metrics::{DeviceScope, DispatchScope, TransferScope, record_device};
use molla_sim::runtime::JointBreakLimits;
use molla_solvers::rigid_world::{GpuContactCapacity, RigidWorld};
use settings::Settings;
use std::{error::Error, sync::Arc, time::Instant};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

fn measure(s: &Settings, kind: &str, count: usize, engine: Option<&Arc<Engine>>) -> Result<()> {
    let backend = if engine.is_some() { "gpu" } else { "cpu" };
    println!("case_start\tprimitive={kind}\tinstances={count}\tbackend={backend}");
    let pairs = u32::try_from(count)
        .map_err(|_| "instance count exceeds contact index capacity")?
        .checked_mul(2)
        .ok_or("pair capacity overflow")?
        .max(8);
    let setup = Instant::now();
    let (scene, breaks) = fixtures::build(kind, count)?;
    let articulations = scene.model().articulation_count;
    let bodies = scene.model().body_count;
    let mut world = if let Some(engine) = engine {
        RigidWorld::from_scene_gpu_with_contact_capacity(
            scene,
            engine.clone(),
            GpuContactCapacity {
                pairs,
                contacts_per_island: 8,
            },
        )?
    } else {
        RigidWorld::from_scene(scene)?
    };
    world.solver.substeps = s.substeps;
    world.set_reaction_readout(true)?;
    let setup_seconds = setup.elapsed().as_secs_f64();
    println!(
        "case_setup_complete\tprimitive={kind}\tinstances={count}\tbackend={backend}\tseconds={setup_seconds:.9}"
    );
    for step in 0..s.warmup {
        if s.trace_frames {
            println!("case_frame_begin\tphase=warmup\tstep={step}");
        }
        world
            .step(s.dt)
            .map_err(|e| format!("warmup step {step}: {e}"))?;
        validation::check_frame(&world, step + 1)?;
    }
    println!(
        "case_warmup_complete\tprimitive={kind}\tinstances={count}\tbackend={backend}\tsteps={}",
        s.warmup
    );
    let timer = engine.map(|e| timing::Timer::new(e));
    if let (Some(timer), Some(engine)) = (&timer, engine) {
        timer.begin(engine);
    }
    let mut upload = 0u64;
    let mut readback = 0u64;
    let mut boundary_buffer_bytes = 0;
    let mut boundary_allocations: Option<isize> = None;
    let mut boundary_buffers: Option<isize> = None;
    let mut peak_step_seconds: f64 = 0.;
    let dispatch_scope = DispatchScope::start()?;
    let transfer_scope = TransferScope::start()?;
    let device_scope = engine.map(|_| DeviceScope::start()).transpose()?;
    if let Some(engine) = engine {
        let counters = engine.device().unwrap().get_internal_counters().hal;
        record_device(
            counters.buffer_memory.read(),
            counters.buffers.read(),
            counters.memory_allocations.read(),
        );
    }
    let mut host = allocations::Counts::default();
    let start = Instant::now();
    for step in 0..s.steps {
        if s.trace_frames {
            println!("case_frame_begin\tphase=measured\tstep={step}");
        }
        let frame_start = Instant::now();
        let allocation_scope = allocations::Scope::start()?;
        if step == s.steps / 2 {
            for &joint in &breaks {
                world.scene.set_joint_break_limits(
                    joint,
                    JointBreakLimits {
                        force: Some(0.),
                        torque: None,
                    },
                )?;
            }
        }
        world
            .step(s.dt)
            .map_err(|e| format!("measured step {step}: {e}"))?;
        validation::check_frame(&world, s.warmup + step + 1)?;
        let counts = allocation_scope.finish();
        if counts.overflowed {
            return Err("host allocation metrics overflow".into());
        }
        host.allocations = host
            .allocations
            .checked_add(counts.allocations)
            .ok_or("host allocation count overflow")?;
        host.reallocations = host
            .reallocations
            .checked_add(counts.reallocations)
            .ok_or("host reallocation count overflow")?;
        host.requested_bytes = host
            .requested_bytes
            .checked_add(counts.requested_bytes)
            .ok_or("host requested byte count overflow")?;
        host.peak_live_bytes = host.peak_live_bytes.max(counts.peak_live_bytes);
        host.peak_extra_bytes = host.peak_extra_bytes.max(counts.peak_extra_bytes);
        host.peak_live_allocations = host.peak_live_allocations.max(counts.peak_live_allocations);
        host.peak_extra_allocations = host
            .peak_extra_allocations
            .max(counts.peak_extra_allocations);
        if step == 0 {
            host.initial_live_bytes = counts.initial_live_bytes;
        }
        peak_step_seconds = peak_step_seconds.max(frame_start.elapsed().as_secs_f64());
        if let Some(d) = world.gpu_diagnostics() {
            upload += d.last_state_upload_bytes;
            readback += d.last_state_readback_bytes;
        }
        if let Some(engine) = engine {
            let counters = engine.device().unwrap().get_internal_counters().hal;
            record_device(
                counters.buffer_memory.read(),
                counters.buffers.read(),
                counters.memory_allocations.read(),
            );
            boundary_buffer_bytes = boundary_buffer_bytes.max(counters.buffer_memory.read());
            boundary_buffers = Some(boundary_buffers.unwrap_or(0).max(counters.buffers.read()));
            let allocations = counters.memory_allocations.read();
            if allocations > 0 {
                boundary_allocations = Some(boundary_allocations.unwrap_or(0).max(allocations));
            }
        }
    }
    let wall_seconds = start.elapsed().as_secs_f64();
    let dispatches = dispatch_scope.finish();
    let transfers = transfer_scope.finish();
    let devices = device_scope.map(DeviceScope::finish);
    if devices.is_some_and(|counts| counts.invalid || counts.observations == 0) {
        return Err("invalid runtime device counter observations".into());
    }
    let observed_buffer_bytes = devices.map(|counts| counts.peak_buffer_bytes);
    let observed_buffers = devices.map(|counts| counts.peak_buffers);
    let observed_native_allocations = devices.and_then(|counts| counts.peak_native_allocations);
    let device_observations = devices.map(|counts| counts.observations);
    if transfers.overflowed
        || transfers.upload_bytes < upload
        || transfers.readback_bytes < readback
    {
        return Err("invalid runtime transfer instrumentation count".into());
    }
    let all_transfer_bytes = transfers
        .upload_bytes
        .checked_add(transfers.readback_bytes)
        .ok_or("runtime transfer byte overflow")?;
    if dispatches.overflowed || (engine.is_some() && dispatches.encoded_dispatches == 0) {
        return Err("invalid runtime dispatch instrumentation count".into());
    }
    let encoded_dispatches = dispatches.encoded_dispatches;
    let timeline_seconds = match (&timer, engine) {
        (Some(timer), Some(engine)) => Some(timer.end(engine)?),
        _ => None,
    };
    let events = world.joint_break_events().len();
    validation::check_final_state(&world)?;
    if events != breaks.len() {
        return Err(format!("break event count {events}, expected {}", breaks.len()).into());
    }
    let compilations = world.gpu_diagnostics().map(|d| d.model_compilations);
    println!(
        "case_ok\tprimitive={kind}\tinstances={count}\tbackend={backend}\tbodies={bodies}\tarticulations={articulations}\twarmup={}\tsteps={}\tdt={}\tsubsteps={}\treactions=true\tpair_capacity={pairs}\tcontact_width=8\tsetup_seconds={setup_seconds:.9}\twall_seconds={wall_seconds:.9}\tmax_step_seconds={peak_step_seconds:.9}\tdevice_timeline_seconds={timeline_seconds:?}\tstate_upload_bytes={upload}\tstate_readout_bytes={readback}\tboundary_buffer_bytes={boundary_buffer_bytes}\tboundary_buffers={boundary_buffers:?}\tboundary_allocations={boundary_allocations:?}\tmodel_compilations={compilations:?}\tbreak_events={events}\tencoded_dispatches={encoded_dispatches}\tall_transfer_bytes={all_transfer_bytes}\tall_upload_bytes={}\tall_readback_bytes={}\thost_allocations={}\thost_reallocations={}\thost_requested_bytes={}\thost_initial_live_bytes={}\thost_peak_live_bytes={}\thost_peak_extra_bytes={}\thost_peak_live_allocations={}\thost_peak_extra_allocations={}\tobserved_device_buffer_bytes={observed_buffer_bytes:?}\tobserved_device_buffers={observed_buffers:?}\tobserved_device_native_allocations={observed_native_allocations:?}\tdevice_counter_observations={device_observations:?}\tdevice_transient_peak_allocations=unavailable",
        s.warmup,
        s.steps,
        s.dt,
        s.substeps,
        transfers.upload_bytes,
        transfers.readback_bytes,
        host.allocations,
        host.reallocations,
        host.requested_bytes,
        host.initial_live_bytes,
        host.peak_live_bytes,
        host.peak_extra_bytes,
        host.peak_live_allocations,
        host.peak_extra_allocations
    );
    Ok(())
}

fn main() -> Result<()> {
    let Some(settings) = Settings::parse(std::env::args().skip(1), fixtures::CASES)? else {
        println!("{}", settings::HELP);
        return Ok(());
    };
    println!(
        "benchmark_schema=6\trun_mode={}\ttiming_scope=device_timeline_including_host_queue_idle\tdispatch_scope=instrumented_live_runtime_encodings\tallocation_scope=process_rust_allocator_layouts_in_measured_frames_and_observed_device_totals\tdevice_counter_scope=orchestrator_buffer_allocation_events_and_frame_boundaries_excluding_timestamp_harness\ttransfer_scope=orchestrator_queue_writes_and_mapped_view_footprints_excluding_timestamp_harness\tframe_trace={}\tbuild_profile={}",
        if settings.smoke { "smoke" } else { "benchmark" },
        settings.trace_frames,
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        }
    );
    if settings.worker {
        let engine = if settings.backend != "cpu" {
            let engine = Arc::new(Engine::new_gpu_with_features(timing::FEATURES)?);
            println!("adapter\t{:?}", engine.adapter_info());
            if engine.adapter_info().is_some_and(|info| info.is_software) {
                return Err("primitive GPU benchmarks require a hardware adapter".into());
            }
            Some(engine)
        } else {
            None
        };
        return measure(
            &settings,
            &settings.primitive,
            settings.counts[0],
            engine.as_ref(),
        );
    }
    let single = [settings.primitive.as_str()];
    let kinds = if settings.primitive == "all" {
        fixtures::CASES
    } else {
        &single
    };
    let mut failed = false;
    for &kind in kinds {
        for &count in &settings.counts {
            for gpu in [false, true] {
                if (gpu && settings.backend == "cpu") || (!gpu && settings.backend == "gpu") {
                    continue;
                }
                let backend = if gpu { "gpu" } else { "cpu" };
                println!(
                    "case_process_start\tprimitive={kind}\tinstances={count}\tbackend={backend}\ttimeout_seconds={}",
                    settings.case_timeout_seconds
                );
                let mut child = std::process::Command::new(std::env::current_exe()?)
                    .args([
                        "--worker",
                        "--primitive",
                        kind,
                        "--backend",
                        backend,
                        "--counts",
                        &count.to_string(),
                        "--warmup",
                        &settings.warmup.to_string(),
                        "--steps",
                        &settings.steps.to_string(),
                        "--dt",
                        &settings.dt.to_string(),
                        "--substeps",
                        &settings.substeps.to_string(),
                    ])
                    .args(if settings.trace_frames {
                        &["--trace-frames"][..]
                    } else {
                        &[][..]
                    })
                    .args(if settings.smoke {
                        &["--smoke"][..]
                    } else {
                        &[][..]
                    })
                    .spawn()?;
                let status = match process::wait_for_exit(
                    &mut child,
                    std::time::Duration::from_secs(settings.case_timeout_seconds),
                )? {
                    process::WorkerExit::Completed(status) => status,
                    process::WorkerExit::TimedOut(status) => {
                        eprintln!(
                            "case_process_timeout\tprimitive={kind}\tinstances={count}\tbackend={backend}\tstatus={status}\ttimeout_seconds={}",
                            settings.case_timeout_seconds
                        );
                        return Err("benchmark worker timed out and was reaped; remaining cases were not run".into());
                    }
                };
                if !status.success() {
                    eprintln!(
                        "case_process_error\tprimitive={kind}\tinstances={count}\tbackend={backend}\tstatus={status}"
                    );
                    failed = true;
                }
            }
        }
    }
    if failed {
        Err("one or more primitive benchmark cases failed; see case_process_error records".into())
    } else {
        Ok(())
    }
}
