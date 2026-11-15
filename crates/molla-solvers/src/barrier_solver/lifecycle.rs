use molla_core::{Error, Result};
use molla_math::Mat3;
use molla_sim::{ElasticElementSet, Model, State};
use molla_storage::BufferRevision;

#[derive(Clone)]
pub(super) struct ModelKey {
    buffers: [BufferRevision; 23],
    counts: [usize; 8],
    elements: ElasticElementSet,
    rods: bool,
}

impl ModelKey {
    pub(super) fn new(m: &Model, rods: bool) -> Self {
        Self {
            buffers: [
                m.initial_particle_q.revision(),
                m.particle_mass.revision(),
                m.particle_inv_mass.revision(),
                m.spring_a.revision(),
                m.spring_b.revision(),
                m.spring_rest_length.revision(),
                m.spring_compliance.revision(),
                m.spring_tension_only.revision(),
                m.shape_body.revision(),
                m.shape_transform.revision(),
                m.body_mass.revision(),
                m.body_inv_mass.revision(),
                m.tet_indices.revision(),
                m.tet_dm_inv.revision(),
                m.tet_rest_volume.revision(),
                m.tet_mu.revision(),
                m.tet_lambda.revision(),
                m.tet_viscosity.revision(),
                m.tet_f_plastic.revision(),
                m.tet_plastic_strain.revision(),
                m.tet_yield_stress.revision(),
                m.tet_hardening.revision(),
                m.tet_kinematic_hardening.revision(),
            ],
            counts: [
                m.particle_count,
                m.body_count,
                m.spring_count,
                m.tet_count,
                m.shape_count,
                m.soft_surface_tris.len(),
                m.soft_surfaces.len(),
                m.shape_geometry.len(),
            ],
            elements: m.elastic_elements().clone(),
            rods,
        }
    }

    pub(super) fn matches(&self, other: &Self) -> bool {
        self.buffers == other.buffers
            && self.counts == other.counts
            && self.rods == other.rods
            && self.elements.same_revision(&other.elements)
    }

    pub(super) fn same_affine_definition(&self, other: &Self) -> bool {
        self.buffers[8..12] == other.buffers[8..12]
            && self.counts[0..2] == other.counts[0..2]
            && self.counts[4] == other.counts[4]
            && self.counts[7] == other.counts[7]
    }
}

#[cfg(feature = "gpu")]
pub(super) fn motion_key(state: &State) -> [BufferRevision; 2] {
    [state.particle_q.revision(), state.particle_qd.revision()]
}

#[derive(Default)]
pub(super) struct TetStates {
    validated: Vec<[BufferRevision; 3]>,
}

impl TetStates {
    pub(super) fn validate(&mut self, count: usize, state: &State) -> Result<()> {
        if state.tet_f_plastic.len() != count
            || state.tet_plastic_strain.len() != count
            || state.tet_back_stress.len() != count
        {
            return Err(Error::Solver(
                "barrier solver: invalid tet state lengths".into(),
            ));
        }
        if count == 0 {
            return Ok(());
        }
        let key = [
            state.tet_f_plastic.revision(),
            state.tet_plastic_strain.revision(),
            state.tet_back_stress.revision(),
        ];
        if self.validated.contains(&key) {
            return Ok(());
        }
        if state
            .tet_f_plastic
            .host()?
            .iter()
            .any(|m| *m != Mat3::IDENTITY)
            || state.tet_plastic_strain.host()?.iter().any(|v| *v != 0.0)
            || state
                .tet_back_stress
                .host()?
                .iter()
                .any(|m| *m != Mat3::ZERO)
        {
            return Err(Error::Solver(
                "barrier solver: legacy plastic tet state is not integrated".into(),
            ));
        }
        if self.validated.len() == 4 {
            self.validated.remove(0);
        }
        self.validated.push(key);
        Ok(())
    }
}
