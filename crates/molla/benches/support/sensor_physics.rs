use molla_compute::Engine;
use molla_core::BodyId;
use molla_math::{GpuTransform, SpatialInertia, Transform, Vec3};
use molla_sim::{BodyParams, Contacts, Control, Model, ModelBuilder, State, eval_fk};
use molla_solvers::{FeatherstoneSolver, Solver};

pub const DT: f64 = 0.001;
pub const LINKS: usize = 4;
pub const RADIUS: f64 = 0.22;

pub struct Physics {
    pub model: Model,
    pub state: State,
    scratch: State,
    control: Control,
    contacts: Contacts,
    solver: FeatherstoneSolver,
    pose_upload: Vec<GpuTransform>,
    pub gpu: bool,
}

impl Physics {
    pub fn new(engine: &Engine, articulations: usize, gpu: bool) -> Self {
        let mut builder = ModelBuilder::new();
        builder
            .add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
            .unwrap();
        let side = (articulations as f64).sqrt().ceil() as usize;
        for art in 0..articulations {
            if art != 0 {
                builder.begin_articulation();
            }
            let mut parent = BodyId::NONE;
            for link in 0..LINKS {
                let body = builder.add_body(
                    BodyParams::new().with_spatial_inertia(SpatialInertia::sphere(1.0, RADIUS)),
                );
                builder
                    .add_shape_sphere(body, RADIUS, Transform::IDENTITY)
                    .unwrap();
                let anchor = if link == 0 {
                    Vec3::new((art % side) as f64 * 4.0, 8.0, (art / side) as f64 * 4.0)
                } else {
                    Vec3::NEG_Y * 0.5
                };
                builder
                    .add_joint_revolute(
                        parent,
                        body,
                        Transform::from_translation(anchor),
                        Transform::from_translation(Vec3::Y * 0.5),
                        Vec3::Z,
                    )
                    .unwrap();
                parent = body;
            }
        }
        let mut model = builder.build().unwrap();
        let mut state = model.state().unwrap();
        state.joint_q.host_mut().unwrap().fill(0.35);
        state.joint_qd.host_mut().unwrap().fill(0.15);
        eval_fk(&model, &mut state).unwrap();
        let mut scratch = state.clone_host().unwrap();
        let mut control = model.control();
        let (device, queue) = (engine.device().unwrap(), engine.queue().unwrap());
        model.upload_to_device(device, queue).unwrap();
        state.upload_to_device(device, queue).unwrap();
        scratch.upload_to_device(device, queue).unwrap();
        control.upload_to_device(device, queue).unwrap();
        let solver = if gpu {
            FeatherstoneSolver::new_gpu(engine, &model).unwrap()
        } else {
            FeatherstoneSolver::new_cpu()
        }
        .with_damping(0.1);
        Self {
            model,
            state,
            scratch,
            control,
            contacts: Contacts::with_capacity(0),
            solver,
            pose_upload: Vec::with_capacity(articulations * LINKS),
            gpu,
        }
    }

    pub fn step(&mut self, engine: &Engine, steps: usize) -> u64 {
        for _ in 0..steps {
            self.solver
                .step(
                    &self.model,
                    &self.state,
                    &self.control,
                    &self.contacts,
                    DT,
                    &mut self.scratch,
                )
                .unwrap();
            std::mem::swap(&mut self.state, &mut self.scratch);
        }
        if self.gpu {
            assert!(self.state.body_q.host().is_err());
            0
        } else {
            self.state
                .body_q
                .upload_with_scratch(engine.queue().unwrap(), &mut self.pose_upload)
                .unwrap();
            self.model.body_count as u64 * 32
        }
    }
}
