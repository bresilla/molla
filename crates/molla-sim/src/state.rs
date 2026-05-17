//! `State` — time-varying positions, velocities, forces.

use molla_math::{Mat3, Real, SpatialVector, Transform, Vec3};
use molla_storage::{Buffer, BufferUsage};

/// Mutable state advanced by a [`crate::solvers::Solver`] each step.
#[derive(Debug)]
pub struct State {
    /// Particle positions (m).
    pub particle_q: Buffer<Vec3>,
    /// Particle velocities (m/s).
    pub particle_qd: Buffer<Vec3>,
    /// Particle forces (N).
    pub particle_f: Buffer<Vec3>,

    /// Rigid body poses. Translation is the body-frame origin in world,
    /// not the COM (see `Model.body_com` for the COM offset).
    pub body_q: Buffer<Transform>,
    /// Rigid body twists (m/s, rad/s) — linear is COM velocity in world,
    /// angular is body angular velocity in world frame.
    pub body_qd: Buffer<SpatialVector>,
    /// Rigid body applied wrenches (N, N·m) at COM, world frame.
    pub body_f: Buffer<SpatialVector>,

    /// Generalised joint coordinates (m or rad).
    pub joint_q: Buffer<Real>,
    /// Generalised joint velocities (m/s or rad/s).
    pub joint_qd: Buffer<Real>,

    // ── Stage 20.3 — VBD per-tet plasticity state ───────────────────
    // Multiplicative split `F = F_e · F_p`. Initialised from
    // `Model.tet_f_plastic` / `Model.tet_plastic_strain` at
    // `State::for_model`. The VBD return-mapping pass mutates these
    // each substep; the elastic Neo-Hookean energy is evaluated on
    // `F_e = F · F_p⁻¹` so the per-vertex block forces decay back to
    // the rest state of the *plastically-deformed* configuration
    // (i.e. permanent set after release).
    /// Per-tet plastic deformation gradient `F_p` (3×3 column-major).
    /// Length = `model.tet_count`. Identity at allocation.
    pub tet_f_plastic: Buffer<Mat3>,
    /// Per-tet equivalent plastic strain (scalar accumulator).
    /// Length = `model.tet_count`. Zero at allocation.
    pub tet_plastic_strain: Buffer<Real>,
    // ── Stage 20.4 — kinematic-hardening back-stress ───────────────
    /// Per-tet symmetric back-stress tensor `α` (Pa, 3×3 column-major
    /// stored as `Mat3`). Mutated by the return-mapping pass per the
    /// Prager / Ziegler evolution rule when
    /// `Model.tet_kinematic_hardening > 0`. Length = `model.tet_count`.
    /// Zero at allocation. Only the symmetric part is physically
    /// meaningful — the kernel symmetrises on read.
    pub tet_back_stress: Buffer<Mat3>,
}

impl State {
    /// An empty state — zero of everything. Useful as a sentinel and for
    /// tests that don't need a real `Model`.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            particle_q: Buffer::from_vec(Vec::new(), BufferUsage::StorageReadWrite),
            particle_qd: Buffer::from_vec(Vec::new(), BufferUsage::StorageReadWrite),
            particle_f: Buffer::from_vec(Vec::new(), BufferUsage::StorageReadWrite),
            body_q: Buffer::from_vec(Vec::new(), BufferUsage::StorageReadWrite),
            body_qd: Buffer::from_vec(Vec::new(), BufferUsage::StorageReadWrite),
            body_f: Buffer::from_vec(Vec::new(), BufferUsage::StorageReadWrite),
            joint_q: Buffer::from_vec(Vec::new(), BufferUsage::StorageReadWrite),
            joint_qd: Buffer::from_vec(Vec::new(), BufferUsage::StorageReadWrite),
            tet_f_plastic: Buffer::from_vec(Vec::new(), BufferUsage::StorageReadWrite),
            tet_plastic_strain: Buffer::from_vec(Vec::new(), BufferUsage::StorageReadWrite),
            tet_back_stress: Buffer::from_vec(Vec::new(), BufferUsage::StorageReadWrite),
        }
    }

    /// Allocate a state sized for `model`. Initial body poses come from
    /// `model.initial_body_q`; everything else is zero.
    pub fn for_model(model: &super::Model) -> molla_core::Result<Self> {
        let initial = model.initial_body_q.host()?.to_vec();
        let initial_particles = if model.particle_count > 0 {
            model.initial_particle_q.host()?.to_vec()
        } else {
            Vec::new()
        };
        // Stage 20.3 — seed the per-tet plasticity state from the
        // model's build-time defaults (identity F_p, zero ε_p when the
        // user uses the back-compat `add_tet_mesh`; user-supplied
        // values when they call `add_tet_mesh_per_material` with a
        // pre-deformed initial configuration).
        let initial_fp = if model.tet_count > 0 {
            model.tet_f_plastic.host()?.to_vec()
        } else {
            Vec::new()
        };
        let initial_eps = if model.tet_count > 0 {
            model.tet_plastic_strain.host()?.to_vec()
        } else {
            Vec::new()
        };
        // Stage 20.4 — kinematic-hardening back-stress, zero at start.
        let initial_alpha = if model.tet_count > 0 {
            vec![Mat3::ZERO; model.tet_count]
        } else {
            Vec::new()
        };
        Ok(Self {
            particle_q: Buffer::from_vec(initial_particles, BufferUsage::StorageReadWrite),
            particle_qd: Buffer::from_vec(
                vec![Vec3::ZERO; model.particle_count],
                BufferUsage::StorageReadWrite,
            ),
            particle_f: Buffer::from_vec(
                vec![Vec3::ZERO; model.particle_count],
                BufferUsage::StorageReadWrite,
            ),
            body_q: Buffer::from_vec(initial, BufferUsage::StorageReadWrite),
            body_qd: Buffer::from_vec(
                vec![SpatialVector::ZERO; model.body_count],
                BufferUsage::StorageReadWrite,
            ),
            body_f: Buffer::from_vec(
                vec![SpatialVector::ZERO; model.body_count],
                BufferUsage::StorageReadWrite,
            ),
            joint_q: Buffer::from_vec(
                model.initial_joint_q.host()?.to_vec(),
                BufferUsage::StorageReadWrite,
            ),
            joint_qd: Buffer::from_vec(
                vec![0.0; model.joint_dof_total],
                BufferUsage::StorageReadWrite,
            ),
            tet_f_plastic: Buffer::from_vec(initial_fp, BufferUsage::StorageReadWrite),
            tet_plastic_strain: Buffer::from_vec(initial_eps, BufferUsage::StorageReadWrite),
            tet_back_stress: Buffer::from_vec(initial_alpha, BufferUsage::StorageReadWrite),
        })
    }

    /// Allocate a fresh `State` with the same shape as `self` and copy
    /// host-side data. Used by sub-stepping wrappers to ping-pong
    /// without re-touching device buffers.
    pub fn clone_host(&self) -> molla_core::Result<Self> {
        use molla_storage::BufferUsage::StorageReadWrite;
        Ok(Self {
            particle_q: Buffer::from_vec(self.particle_q.host()?.to_vec(), StorageReadWrite),
            particle_qd: Buffer::from_vec(self.particle_qd.host()?.to_vec(), StorageReadWrite),
            particle_f: Buffer::from_vec(self.particle_f.host()?.to_vec(), StorageReadWrite),
            body_q: Buffer::from_vec(self.body_q.host()?.to_vec(), StorageReadWrite),
            body_qd: Buffer::from_vec(self.body_qd.host()?.to_vec(), StorageReadWrite),
            body_f: Buffer::from_vec(self.body_f.host()?.to_vec(), StorageReadWrite),
            joint_q: Buffer::from_vec(self.joint_q.host()?.to_vec(), StorageReadWrite),
            joint_qd: Buffer::from_vec(self.joint_qd.host()?.to_vec(), StorageReadWrite),
            tet_f_plastic: Buffer::from_vec(self.tet_f_plastic.host()?.to_vec(), StorageReadWrite),
            tet_plastic_strain: Buffer::from_vec(
                self.tet_plastic_strain.host()?.to_vec(),
                StorageReadWrite,
            ),
            tet_back_stress: Buffer::from_vec(
                self.tet_back_stress.host()?.to_vec(),
                StorageReadWrite,
            ),
        })
    }

    /// Copy host-side data from `other` into `self`. Sizes must match.
    pub fn copy_host_from(&mut self, other: &Self) -> molla_core::Result<()> {
        self.particle_q
            .host_mut()?
            .copy_from_slice(other.particle_q.host()?);
        self.particle_qd
            .host_mut()?
            .copy_from_slice(other.particle_qd.host()?);
        self.particle_f
            .host_mut()?
            .copy_from_slice(other.particle_f.host()?);
        self.body_q
            .host_mut()?
            .copy_from_slice(other.body_q.host()?);
        self.body_qd
            .host_mut()?
            .copy_from_slice(other.body_qd.host()?);
        self.body_f
            .host_mut()?
            .copy_from_slice(other.body_f.host()?);
        self.joint_q
            .host_mut()?
            .copy_from_slice(other.joint_q.host()?);
        self.joint_qd
            .host_mut()?
            .copy_from_slice(other.joint_qd.host()?);
        if !self.tet_f_plastic.host()?.is_empty() {
            self.tet_f_plastic
                .host_mut()?
                .copy_from_slice(other.tet_f_plastic.host()?);
            self.tet_plastic_strain
                .host_mut()?
                .copy_from_slice(other.tet_plastic_strain.host()?);
            self.tet_back_stress
                .host_mut()?
                .copy_from_slice(other.tet_back_stress.host()?);
        }
        Ok(())
    }

    /// Zero out all force arrays. Solvers call this at the top of each step.
    pub fn clear_forces(&mut self) -> molla_core::Result<()> {
        for f in self.body_f.host_mut()? {
            *f = SpatialVector::ZERO;
        }
        for f in self.particle_f.host_mut()? {
            *f = Vec3::ZERO;
        }
        Ok(())
    }

    /// Allocate device buffers for every state array and upload host data.
    /// Idempotent: subsequent calls re-upload from host.
    #[cfg(feature = "gpu")]
    pub fn upload_to_device(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> molla_core::Result<()> {
        macro_rules! up {
            ($field:ident) => {{
                self.$field.ensure_device(device)?;
                self.$field.upload(queue)?;
            }};
        }
        up!(particle_q);
        up!(particle_qd);
        up!(particle_f);
        up!(body_q);
        up!(body_qd);
        up!(body_f);
        up!(joint_q);
        up!(joint_qd);
        // Stage 20.3 — plasticity state lives device-side too so the
        // GPU return-mapping kernel can mutate it in place.
        if !self.tet_f_plastic.host()?.is_empty() {
            up!(tet_f_plastic);
            up!(tet_plastic_strain);
            up!(tet_back_stress);
        }
        Ok(())
    }

    /// Read every device buffer back into the host mirror. Blocks until
    /// the transfers complete.
    #[cfg(feature = "gpu")]
    pub fn download_from_device(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> molla_core::Result<()> {
        self.body_q.download(device, queue)?;
        self.body_qd.download(device, queue)?;
        Ok(())
    }
}
