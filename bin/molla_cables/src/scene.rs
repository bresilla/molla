use molla_compute::Engine;
use molla_core::{BodyId, Result, WorldId};
use molla_math::{Transform, Vec3};
use molla_sim::{Contacts, Control, Model, ModelBuilder, State};
use molla_solvers::{Solver, VbdSolver};

pub const DT: f64 = 1.0 / 240.0;
pub const NAMES: [&str; 3] = [
    "Suspended cable",
    "Bending comparison",
    "Drape over a sphere",
];
pub const DESCRIPTIONS: [&str; 3] = [
    "Two fixed ends, gravity and a sideways disturbance. Pluck it to send another wave down the cable.",
    "Three equally loaded cantilevers. Cyan is flexible, amber is medium, violet is stiff. Two nodes clamp each root.",
    "Three cables fall onto a fixed sphere and the floor. Contact uses the actual VBD particle-shape penalties.",
];

pub struct Scene {
    pub model: Model,
    pub solver: VbdSolver,
    pub current: State,
    next: State,
    control: Control,
    contacts: Contacts,
    pub cables: Vec<Vec<u32>>,
    pub anchors: Vec<u32>,
    pub sphere: Option<(Vec3, f64)>,
    pub points: Vec<Vec3>,
    pub time: f64,
    pub max_stretch: f64,
    rest: Vec<Vec3>,
}

impl Scene {
    pub fn new(engine: &Engine, preset: usize) -> Result<Self> {
        let mut b = ModelBuilder::new();
        b.set_linear_damping(0.45);
        b.add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)?;
        let mut cables = Vec::new();
        let mut anchors = Vec::new();
        let count = if preset == 0 { 1 } else { 3 };
        let n = if preset == 1 { 16 } else { 32 };
        for c in 0..count {
            let mut ids = Vec::new();
            for i in 0..n {
                let t = i as f64 / (n - 1) as f64;
                let p = match preset {
                    0 => Vec3::new(
                        -1.6 + 3.2 * t,
                        2.5 - 0.65 * (std::f64::consts::PI * t).sin(),
                        0.0,
                    ),
                    1 => Vec3::new(-1.2 + 2.4 * t, 2.1, (c as f64 - 1.0) * 0.65),
                    _ => Vec3::new(
                        -1.7 + 3.4 * t,
                        2.45 + 0.06 * c as f64,
                        (c as f64 - 1.0) * 0.4,
                    ),
                };
                ids.push(b.add_particle(p, 0.025, 0.035, WorldId(0)));
            }
            let id = b.add_cable(&ids, 5.0e4, 0.05);
            b.set_cable_bending_stiffness(
                id,
                if preset == 1 {
                    [5.0, 2.0e3, 8.0e4][c]
                } else {
                    5.0
                },
            );
            if preset < 2 {
                anchors.extend([ids[0], ids[if preset == 0 { n - 1 } else { 1 }]]);
            }
            cables.push(ids);
        }
        for &id in &anchors {
            b.pin_particle(id);
        }
        let sphere = (preset == 2).then_some((Vec3::new(0.0, 0.8, 0.0), 0.7));
        if let Some((center, radius)) = sphere {
            b.add_shape_sphere(BodyId::NONE, radius, Transform::from_translation(center))?;
        }
        let mut model = b.build()?;
        let mut current = model.state()?;
        if preset == 0 {
            for (i, v) in current.particle_qd.host_mut()?.iter_mut().enumerate() {
                v.z = 1.6 * (std::f64::consts::PI * i as f64 / (n - 1) as f64).sin();
            }
            for &id in &anchors {
                current.particle_qd.host_mut()?[id as usize] = Vec3::ZERO;
            }
        }
        let rest = current.particle_q.host()?.to_vec();
        let mut next = model.state()?;
        let mut control = model.control();
        let device = engine.device().unwrap();
        let queue = engine.queue().unwrap();
        model.upload_to_device(device, queue)?;
        current.upload_to_device(device, queue)?;
        next.upload_to_device(device, queue)?;
        control.upload_to_device(device, queue)?;
        let solver = VbdSolver::new_gpu(engine, &model)?
            .with_iterations(if preset == 1 { 64 } else { 24 })
            .with_contact_stiffness(2.0e5);
        Ok(Self {
            model,
            solver,
            current,
            next,
            control,
            contacts: Contacts::with_capacity(0),
            cables,
            anchors,
            sphere,
            points: rest.clone(),
            time: 0.0,
            max_stretch: 1.0,
            rest,
        })
    }

    pub fn advance(&mut self, engine: &Engine, steps: usize) -> Result<()> {
        for _ in 0..steps {
            self.solver.step(
                &self.model,
                &self.current,
                &self.control,
                &self.contacts,
                DT,
                &mut self.next,
            )?;
            std::mem::swap(&mut self.current, &mut self.next);
            self.time += DT;
        }
        self.current
            .particle_q
            .download(engine.device().unwrap(), engine.queue().unwrap())?;
        self.points.copy_from_slice(self.current.particle_q.host()?);
        self.max_stretch = self
            .cables
            .iter()
            .flat_map(|c| c.windows(2))
            .map(|e| {
                self.points[e[0] as usize].distance(self.points[e[1] as usize])
                    / self.rest[e[0] as usize].distance(self.rest[e[1] as usize])
            })
            .fold(0.0, f64::max);
        self.validate()
    }

    pub fn pluck(&mut self, engine: &Engine) -> Result<()> {
        let device = engine.device().unwrap();
        let queue = engine.queue().unwrap();
        self.current.particle_qd.download(device, queue)?;
        let velocities = self.current.particle_qd.host_mut()?;
        for ids in &self.cables {
            for (i, &id) in ids.iter().enumerate() {
                if !self.anchors.contains(&id) {
                    velocities[id as usize].z +=
                        1.5 * (std::f64::consts::PI * i as f64 / (ids.len() - 1) as f64).sin();
                }
            }
        }
        self.current.particle_qd.upload(queue)
    }

    pub fn validate(&self) -> Result<()> {
        use molla_core::Error;
        if self.points.iter().any(|p| !p.is_finite())
            || !self.max_stretch.is_finite()
            || self.max_stretch > 1.05
        {
            return Err(Error::Solver(format!(
                "invalid cable state: maximum stretch {}",
                self.max_stretch
            )));
        }
        for &id in &self.anchors {
            if self.points[id as usize].distance(self.rest[id as usize]) > 1.0e-5 {
                return Err(Error::Solver("cable anchor moved".into()));
            }
        }
        Ok(())
    }

    pub fn clearance(&self) -> f64 {
        self.points
            .iter()
            .map(|p| {
                let floor = p.y - 0.035;
                self.sphere
                    .map_or(floor, |(c, r)| floor.min(p.distance(c) - r - 0.035))
            })
            .fold(f64::INFINITY, f64::min)
    }

    pub fn tube_clearance(&self) -> f64 {
        let mut minimum = self.clearance();
        if let Some((center, radius)) = self.sphere {
            for edge in self.cables.iter().flat_map(|ids| ids.windows(2)) {
                let a = self.points[edge[0] as usize];
                let d = self.points[edge[1] as usize] - a;
                let t = ((center - a).dot(d) / d.length_squared().max(1.0e-20)).clamp(0.0, 1.0);
                minimum = minimum.min((a + t * d).distance(center) - radius - 0.035);
            }
        }
        minimum
    }
}
