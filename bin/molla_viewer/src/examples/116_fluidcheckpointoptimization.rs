// ═══ Scene 116: FluidCheckpointOptimization — reduced checkpointed fluid opt ═

const FLUID_CHECKPOINT_N: usize = 32;
const FLUID_CHECKPOINT_HORIZON: usize = 48;
const FLUID_CHECKPOINT_EVERY: usize = 8;

#[derive(Clone)]
struct FluidCheckpointSnapshot {
    step: usize,
    field: Vec<f64>,
}

struct FluidCheckpointResult {
    final_field: Vec<f64>,
    loss: f64,
    checkpoints: Vec<FluidCheckpointSnapshot>,
}

struct FluidCheckpointState {
    control: f64,
    target_amp: f64,
    lr: f64,
    iterations: usize,
    loss: f64,
    initial_loss: f64,
    field: Vec<f64>,
    target: Vec<f64>,
    checkpoint_count: usize,
    last_checkpoint_step: usize,
    checkpoint_field_len: usize,
    needs_rerun: bool,
}

impl Default for FluidCheckpointState {
    fn default() -> Self {
        let target = fluid_checkpoint_target(FLUID_CHECKPOINT_N, 0.85);
        Self {
            control: 0.10,
            target_amp: 0.85,
            lr: 0.8,
            iterations: 0,
            loss: f64::INFINITY,
            initial_loss: f64::INFINITY,
            field: vec![0.0; FLUID_CHECKPOINT_N],
            target,
            checkpoint_count: 0,
            last_checkpoint_step: 0,
            checkpoint_field_len: 0,
            needs_rerun: true,
        }
    }
}

fn fluid_checkpoint_source(n: usize) -> Vec<f64> {
    let center = 0.35 * (n as f64 - 1.0);
    (0..n)
        .map(|i| {
            let x = (i as f64 - center) / (0.12 * n as f64);
            (-0.5 * x * x).exp()
        })
        .collect()
}

fn fluid_checkpoint_target(n: usize, amp: f64) -> Vec<f64> {
    let center = 0.68 * (n as f64 - 1.0);
    (0..n)
        .map(|i| {
            let x = (i as f64 - center) / (0.15 * n as f64);
            amp * (-0.5 * x * x).exp()
        })
        .collect()
}

fn fluid_checkpoint_step(field: &[f64], source: &[f64], control: f64) -> Vec<f64> {
    let n = field.len();
    let mut next = vec![0.0; n];
    for i in 0..n {
        let l = field[(i + n - 1) % n];
        let c = field[i];
        let r = field[(i + 1) % n];
        let advected = 0.72 * c + 0.23 * l + 0.05 * r;
        let diffused = advected + 0.06 * (l - 2.0 * c + r);
        next[i] = 0.985 * diffused + 0.020 * control * source[i];
    }
    next
}

fn fluid_checkpoint_rollout(
    control: f64,
    target: &[f64],
    checkpoint_every: usize,
) -> FluidCheckpointResult {
    let n = target.len();
    let source = fluid_checkpoint_source(n);
    let mut field = vec![0.0; n];
    let mut checkpoints = vec![FluidCheckpointSnapshot {
        step: 0,
        field: field.clone(),
    }];
    for step in 1..=FLUID_CHECKPOINT_HORIZON {
        field = fluid_checkpoint_step(&field, &source, control);
        if step % checkpoint_every == 0 || step == FLUID_CHECKPOINT_HORIZON {
            checkpoints.push(FluidCheckpointSnapshot {
                step,
                field: field.clone(),
            });
        }
    }
    let loss = field
        .iter()
        .zip(target.iter())
        .map(|(a, b)| {
            let e = a - b;
            0.5 * e * e
        })
        .sum::<f64>()
        / n as f64;
    FluidCheckpointResult {
        final_field: field,
        loss,
        checkpoints,
    }
}

fn fluid_checkpoint_loss(control: f64, target: &[f64]) -> f64 {
    fluid_checkpoint_rollout(control, target, FLUID_CHECKPOINT_EVERY).loss
}

fn fluid_checkpoint_fd_gradient(control: f64, target: &[f64]) -> f64 {
    let eps = 1.0e-4;
    (fluid_checkpoint_loss(control + eps, target) - fluid_checkpoint_loss(control - eps, target))
        / (2.0 * eps)
}

fn fluid_checkpoint_optimize(control0: f64, target: &[f64], iters: usize, lr: f64) -> (f64, f64) {
    let mut control = control0;
    for _ in 0..iters {
        let g = fluid_checkpoint_fd_gradient(control, target);
        control = (control - lr * g).clamp(-4.0, 4.0);
    }
    (control, fluid_checkpoint_loss(control, target))
}

fn update_fluid_checkpoint(mut warp: ResMut<WarpExampleStates>, sim: Res<Sim>) {
    if sim.scene != Scene::FluidCheckpointOptimization {
        return;
    }
    let st = &mut warp.fluid_checkpoint;
    if !st.needs_rerun {
        return;
    }
    st.target = fluid_checkpoint_target(FLUID_CHECKPOINT_N, st.target_amp);
    st.initial_loss = fluid_checkpoint_loss(st.control, &st.target);
    let (control, loss) = fluid_checkpoint_optimize(st.control, &st.target, 32, st.lr);
    st.control = control;
    let result = fluid_checkpoint_rollout(st.control, &st.target, FLUID_CHECKPOINT_EVERY);
    st.field = result.final_field;
    st.loss = loss.min(result.loss);
    st.iterations += 32;
    st.checkpoint_count = result.checkpoints.len();
    if let Some(last) = result.checkpoints.last() {
        st.last_checkpoint_step = last.step;
        st.checkpoint_field_len = last.field.len();
    }
    st.needs_rerun = false;
}

fn draw_fluid_checkpoint_gizmos(sim: Res<Sim>, warp: Res<WarpExampleStates>, mut gizmos: Gizmos) {
    if sim.scene != Scene::FluidCheckpointOptimization {
        return;
    }
    let st = &warp.fluid_checkpoint;
    if st.field.is_empty() {
        return;
    }
    let origin = Vec3::new(-0.8, 0.08, -0.15);
    let dx = 1.6 / (st.field.len().saturating_sub(1).max(1)) as f32;
    for i in 0..st.field.len() {
        let x = origin.x + i as f32 * dx;
        let y = origin.y + st.field[i] as f32 * 0.75;
        let yt = origin.y + st.target[i] as f32 * 0.75;
        gizmos.sphere(Vec3::new(x, y, 0.0), 0.014, Color::srgb(0.25, 0.85, 1.0));
        gizmos.sphere(Vec3::new(x, yt, -0.08), 0.010, Color::srgb(1.0, 0.45, 0.25));
        if i + 1 < st.field.len() {
            let x2 = origin.x + (i + 1) as f32 * dx;
            let y2 = origin.y + st.field[i + 1] as f32 * 0.75;
            let yt2 = origin.y + st.target[i + 1] as f32 * 0.75;
            gizmos.line(
                Vec3::new(x, y, 0.0),
                Vec3::new(x2, y2, 0.0),
                Color::srgb(0.25, 0.85, 1.0),
            );
            gizmos.line(
                Vec3::new(x, yt, -0.08),
                Vec3::new(x2, yt2, -0.08),
                Color::srgb(1.0, 0.45, 0.25),
            );
        }
    }
}

fn fluid_checkpoint_pane(ui: &mut egui::Ui, st: &mut FluidCheckpointState) {
    ui.label("Fluid checkpoint optimization — reduced scalar rollout");
    ui.separator();
    readout(ui, "Control", &format!("{:.4}", st.control));
    readout(ui, "Initial loss", &format!("{:.3e}", st.initial_loss));
    readout(ui, "Loss", &format!("{:.3e}", st.loss));
    readout(ui, "Checkpoints", &format!("{}", st.checkpoint_count));
    readout(
        ui,
        "Last ckpt step",
        &format!("{}", st.last_checkpoint_step),
    );
    readout(
        ui,
        "Ckpt field len",
        &format!("{}", st.checkpoint_field_len),
    );
    readout(ui, "Iterations", &format!("{}", st.iterations));
    let changed_amp = ui
        .add(egui::Slider::new(&mut st.target_amp, 0.1..=1.5).text("target amp"))
        .changed();
    let changed_lr = ui
        .add(egui::Slider::new(&mut st.lr, 0.05..=2.0).text("learning rate"))
        .changed();
    if changed_amp || changed_lr || ui.button("Re-run optimization").clicked() {
        st.needs_rerun = true;
    }
    ui.small("Blue = optimized final field; orange = target. Checkpoints save every 8 steps.");
    ui.small(
        "CPU reduced checkpoint oracle; full GPU fluid adjoint/checkpointing remains future work.",
    );
}

