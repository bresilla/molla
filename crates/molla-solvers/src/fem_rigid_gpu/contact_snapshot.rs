use molla_kernels::soft_rigid_solve::{SoftRigidSolveBindings, SoftRigidSolveSettings};

const NAMES: [&str; 10] = [
    "positions",
    "velocities",
    "inverse_mass",
    "contacts",
    "body_poses",
    "body_com",
    "jacobians",
    "mass_matrix",
    "rigid_velocities",
    "body_velocities",
];

fn sources(bindings: SoftRigidSolveBindings<'_>) -> [&wgpu::Buffer; 10] {
    [
        bindings.positions,
        bindings.velocities,
        bindings.inverse_mass,
        bindings.contacts,
        bindings.body_poses,
        bindings.body_com,
        bindings.jacobians,
        bindings.mass_matrix,
        bindings.rigid_velocities,
        bindings.body_velocities,
    ]
}

/// Device-resident first-contact inputs and final pre-drift solved velocities.
pub struct ContactSolveSnapshot {
    buffers: [wgpu::Buffer; 10],
    solved: [wgpu::Buffer; 2],
    pub(super) settings: Option<SoftRigidSolveSettings>,
}

impl ContactSolveSnapshot {
    pub(super) fn new(device: &wgpu::Device, bindings: SoftRigidSolveBindings<'_>) -> Self {
        let source = sources(bindings);
        Self {
            buffers: std::array::from_fn(|i| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(NAMES[i]),
                    size: source[i].size(),
                    usage: wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                })
            }),
            solved: [source[1], source[8]].map(|buffer| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("FEM pre-drift solved velocity snapshot"),
                    size: buffer.size(),
                    usage: wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                })
            }),
            settings: None,
        }
    }

    pub(super) fn capture(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        bindings: SoftRigidSolveBindings<'_>,
        settings: SoftRigidSolveSettings,
    ) {
        for (source, destination) in sources(bindings).into_iter().zip(&self.buffers) {
            encoder.copy_buffer_to_buffer(source, 0, destination, 0, source.size());
        }
        self.settings = Some(settings);
    }

    /// Named input buffers; the owning system's submission must complete before reading.
    pub fn buffers(&self) -> impl Iterator<Item = (&'static str, &wgpu::Buffer)> {
        NAMES.into_iter().zip(&self.buffers)
    }

    pub(super) fn capture_solved(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        soft_velocity: &wgpu::Buffer,
        rigid_velocity: &wgpu::Buffer,
    ) {
        for (source, destination) in [soft_velocity, rigid_velocity].into_iter().zip(&self.solved) {
            encoder.copy_buffer_to_buffer(source, 0, destination, 0, source.size());
        }
    }

    /// Post-contact velocities before drift, including rejected candidates; wait for completion.
    pub fn solved_buffers(&self) -> impl Iterator<Item = (&'static str, &wgpu::Buffer)> {
        ["solved_velocities", "solved_rigid_velocities"].into_iter().zip(&self.solved)
    }

    /// Settings of the captured first solve, or None before capture.
    pub fn settings(&self) -> Option<SoftRigidSolveSettings> {
        self.settings
    }
}
