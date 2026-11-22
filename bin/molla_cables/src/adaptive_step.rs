use molla_compute::Engine;
use molla_core::{Error, Result};
use molla_solvers::strands::{GpuStrandLimits, GpuStrands, StepConfig, StrandModel};

const MAX_HALVINGS: u32 = 6;

pub struct BatchResult {
    pub model: StrandModel,
    pub halvings: u32,
}

/// Replay a display batch from its checkpoint with finer steps on convergence or angular-resolution failure.
pub fn advance(
    engine: &Engine,
    gpu: &mut GpuStrands,
    checkpoint: &StrandModel,
    limits: GpuStrandLimits,
    config: StepConfig,
    count: usize,
) -> Result<BatchResult> {
    advance_with_grab_start(engine, gpu, checkpoint, limits, config, count, None)
}

pub fn advance_with_grab_start(
    engine: &Engine,
    gpu: &mut GpuStrands,
    checkpoint: &StrandModel,
    limits: GpuStrandLimits,
    config: StepConfig,
    count: usize,
    grab_start: Option<molla_math::Vec3>,
) -> Result<BatchResult> {
    config.validate()?;
    if !(1..=1024).contains(&count) {
        return Err(Error::Solver("invalid adaptive strand batch size".into()));
    }
    let mut failure = String::new();
    for halvings in 0..=MAX_HALVINGS {
        if halvings > 0 {
            if std::env::var_os("MOLLA_STRAND_TRACE").is_some() {
                eprintln!("Adaptive retry: {failure}");
            }
            *gpu = GpuStrands::with_limits(engine, checkpoint.clone(), limits)?;
        }
        let subdivisions = 1usize << halvings;
        let config = StepConfig {
            dt: config.dt / subdivisions as f64,
            ..config
        };
        for step in 0..count * subdivisions {
            let mut sampled = config;
            if let (Some(start), Some(grab)) = (grab_start, &mut sampled.grab) {
                grab.target = start.lerp(
                    grab.target,
                    (step + 1) as f64 / (count * subdivisions) as f64,
                );
            }
            gpu.step(engine, sampled)?;
        }
        let (model, status) = gpu.last_valid_snapshot(engine)?;
        if status == 0 {
            let sweep = model
                .fibers()
                .iter()
                .map(|f| f.motion.angular_velocity.length() * config.dt)
                .fold(0.0, f64::max);
            if sweep <= 0.35 {
                return Ok(BatchResult { model, halvings });
            }
            let fastest = model
                .fibers()
                .iter()
                .max_by(|a, b| {
                    a.motion
                        .angular_velocity
                        .length_squared()
                        .total_cmp(&b.motion.angular_velocity.length_squared())
                })
                .unwrap();
            failure = format!(
                "unresolved angular sweep {sweep} radians, cell {}, dt={}",
                fastest.id, config.dt
            );
            continue;
        }
        failure = format!(
            "fault {status:#x}, dt={}, {:?}",
            config.dt,
            gpu.last_coupled_diagnostics()
        );
        if status & !20 != 0 {
            break;
        }
    }
    *gpu = GpuStrands::with_limits(engine, checkpoint.clone(), limits)?;
    Err(Error::Solver(format!(
        "strand batch rejected; checkpoint restored: {failure}"
    )))
}
