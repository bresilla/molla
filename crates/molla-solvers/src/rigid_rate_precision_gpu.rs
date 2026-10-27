use super::{Layout, copy_retained, encode};
use crate::FeatherstoneSolver;
use molla_compute::Engine;
use molla_core::{Error, Result};
use molla_math::{Real, SpatialVector};
use molla_sim::State;
use std::collections::BTreeMap;

#[cfg(test)]
#[path = "rigid_rate_precision_tests.rs"]
mod tests;

fn components(v: SpatialVector) -> [Real; 8] {
    [
        v.linear.x,
        v.linear.y,
        v.linear.z,
        0.,
        v.angular.x,
        v.angular.y,
        v.angular.z,
        0.,
    ]
}

pub(super) struct RateTransfers {
    rates: Vec<[u32; 4]>,
    velocities: Vec<[u32; 4]>,
}

impl RateTransfers {
    fn rounded(source: &State) -> Result<Self> {
        let split = |value: Real| -> Result<[f32; 2]> {
            let high = value as f32;
            let low = (value - high as Real) as f32;
            if !high.is_finite() || !low.is_finite() {
                return Err(Error::Solver("nonfinite runtime paired velocity".into()));
            }
            Ok([high, low])
        };
        let mut rates = Vec::with_capacity(source.joint_qd.len() * 2);
        for &value in source.joint_qd.host()? {
            rates.extend(split(value)?.map(|v| [0, 0, v.to_bits(), 0]));
        }
        let mut velocities = Vec::with_capacity(source.body_qd.len() * 8);
        for value in source.body_qd.host()? {
            for v in value
                .linear
                .to_array()
                .into_iter()
                .chain([0.])
                .chain(value.angular.to_array())
                .chain([0.])
            {
                velocities.push([0, 0, split(v)?[1].to_bits(), 0]);
            }
        }
        Ok(Self { rates, velocities })
    }

    fn remap(
        layout: &Layout,
        old: &Layout,
        source: &State,
        old_rates: &[Real],
        old_velocities: &[SpatialVector],
    ) -> Result<Self> {
        if source.body_qd.len() != layout.bodies.len()
            || source.joint_qd.len() != layout.dofs.iter().map(|r| r.end).max().unwrap_or(0)
        {
            return Err(Error::Solver(
                "runtime paired velocity layout mismatch".into(),
            ));
        }
        let mut transfers = Self::rounded(source)?;
        let source_rates = source.joint_qd.host()?;
        let indices: BTreeMap<_, _> = old
            .bodies
            .iter()
            .enumerate()
            .map(|(i, &id)| (id, i))
            .collect();
        for (body, id) in layout.bodies.iter().enumerate() {
            let Some(&previous) = indices.get(id) else {
                continue;
            };
            if layout.identities[body] != old.identities[previous] {
                continue;
            }
            if let Some(old_velocity) = old_velocities.get(previous) {
                let velocity = source.body_qd.host()?[body];
                for (axis, (a, b)) in components(velocity)
                    .into_iter()
                    .zip(components(*old_velocity))
                    .enumerate()
                {
                    if a.to_bits() == b.to_bits() {
                        transfers.velocities[body * 8 + axis] =
                            [(previous * 8 + axis) as u32, 2, 0, 0];
                    }
                }
            }
            let range = &layout.dofs[body];
            let old_range = &old.dofs[previous];
            if layout.joints[body].0 == old.joints[previous].0
                && layout.joints[body].1 == old.joints[previous].1
                && layout.frames[body] == old.frames[previous]
                && range.len() == old_range.len()
            {
                for (to, from) in range.clone().zip(old_range.clone()) {
                    if old_rates
                        .get(from)
                        .is_none_or(|v| v.to_bits() != source_rates[to].to_bits())
                    {
                        continue;
                    }
                    for part in 0..2 {
                        transfers.rates[to * 2 + part] = [(from * 2 + part) as u32, 1, 0, 0];
                    }
                }
            }
        }
        Ok(transfers)
    }
}

pub(super) struct RuntimeRates {
    states: [[wgpu::Buffer; 2]; 2],
    scalar_rates: [wgpu::Buffer; 2],
    delta: wgpu::Buffer,
    checkpoints: [wgpu::Buffer; 3],
    published_rates: Vec<Real>,
    published_velocities: Vec<SpatialVector>,
}

impl RuntimeRates {
    pub(super) fn new(
        engine: &Engine,
        states: &[State; 2],
        solver: &FeatherstoneSolver,
    ) -> Result<Self> {
        let buffers = |state| {
            Ok([
                solver.gpu_paired_rates_buffer(state)?.clone(),
                solver.gpu_body_velocity_compensation_buffer(state)?.clone(),
            ])
        };
        let scalar_rates = [
            states[0].joint_qd.device_buffer_current()?.clone(),
            states[1].joint_qd.device_buffer_current()?.clone(),
        ];
        let states = [buffers(&states[0])?, buffers(&states[1])?];
        let delta = solver
            .gpu_paired_velocity_delta_buffer()
            .ok_or_else(|| Error::Gpu("missing runtime paired delta".into()))?
            .clone();
        Self::from_buffers(engine, states, delta, scalar_rates)
    }

    fn from_buffers(
        engine: &Engine,
        states: [[wgpu::Buffer; 2]; 2],
        delta: wgpu::Buffer,
        scalar_rates: [wgpu::Buffer; 2],
    ) -> Result<Self> {
        let buffers = [
            &states[0][0],
            &states[0][1],
            &states[1][0],
            &states[1][1],
            &delta,
            &scalar_rates[0],
            &scalar_rates[1],
        ];
        let required = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        if states[0][0].size() != states[1][0].size()
            || states[0][1].size() != states[1][1].size()
            || delta.size() != states[0][0].size()
            || scalar_rates
                .iter()
                .any(|b| b.size() * 2 != states[0][0].size())
            || buffers
                .iter()
                .enumerate()
                .any(|(i, b)| !b.usage().contains(required) || buffers[..i].contains(b))
        {
            return Err(Error::Gpu(
                "invalid runtime paired checkpoint buffers".into(),
            ));
        }
        let checkpoints = [&states[0][0], &states[0][1], &delta].map(|b| {
            molla_core::gpu_create_buffer!(
                engine.device().unwrap(),
                &wgpu::BufferDescriptor {
                    label: Some("runtime paired velocity checkpoint"),
                    size: b.size(),
                    usage: required,
                    mapped_at_creation: false,
                }
            )
        });
        Ok(Self {
            states,
            scalar_rates,
            delta,
            checkpoints,
            published_rates: Vec::new(),
            published_velocities: Vec::new(),
        })
    }

    pub(super) fn plan(&self, layout: &Layout, source: &State) -> Result<Option<RateTransfers>> {
        if source.joint_qd.len() == self.published_rates.len()
            && source.body_qd.len() == self.published_velocities.len()
            && source
                .joint_qd
                .host()?
                .iter()
                .zip(&self.published_rates)
                .all(|(a, b)| a.to_bits() == b.to_bits())
            && source
                .body_qd
                .host()?
                .iter()
                .zip(&self.published_velocities)
                .all(|(a, b)| {
                    components(*a)
                        .into_iter()
                        .zip(components(*b))
                        .all(|(a, b)| a.to_bits() == b.to_bits())
                })
        {
            return Ok(None);
        }
        RateTransfers::remap(
            layout,
            layout,
            source,
            &self.published_rates,
            &self.published_velocities,
        )
        .map(Some)
    }

    pub(super) fn prepare_frame(
        &self,
        engine: &Engine,
        encoder: &mut wgpu::CommandEncoder,
        transfers: Option<RateTransfers>,
        side: usize,
    ) -> Result<()> {
        if let Some(transfers) = transfers {
            for (from, to) in self.states[side].iter().zip(&self.checkpoints) {
                encoder.copy_buffer_to_buffer(from, 0, to, 0, to.size());
            }
            for (lane, table) in [&transfers.rates, &transfers.velocities]
                .into_iter()
                .enumerate()
            {
                copy_retained(
                    engine.queue().unwrap(),
                    encoder,
                    table,
                    [&self.checkpoints[0], &self.checkpoints[1]],
                    [&self.states[0][lane], &self.states[1][lane]],
                )?;
            }
        } else if side != 0 {
            for (from, to) in self.states[side].iter().zip(&self.states[0]) {
                encoder.copy_buffer_to_buffer(from, 0, to, 0, from.size());
            }
        }
        for (from, to) in self.states[0]
            .iter()
            .chain([&self.delta])
            .zip(&self.checkpoints)
        {
            encoder.copy_buffer_to_buffer(from, 0, to, 0, to.size());
        }
        Ok(())
    }

    pub(super) fn rollback(&self, encoder: &mut wgpu::CommandEncoder) {
        for state in &self.states {
            for (from, to) in self.checkpoints.iter().zip(state) {
                encoder.copy_buffer_to_buffer(from, 0, to, 0, to.size());
            }
        }
        encoder.copy_buffer_to_buffer(&self.checkpoints[2], 0, &self.delta, 0, self.delta.size());
        for scalar in &self.scalar_rates {
            for offset in (0..scalar.size()).step_by(4) {
                encoder.copy_buffer_to_buffer(&self.checkpoints[0], offset * 2, scalar, offset, 4);
            }
        }
    }

    pub(super) fn committed(&mut self, source: &State) -> Result<()> {
        let rates = source.joint_qd.host()?.to_vec();
        let velocities = source.body_qd.host()?.to_vec();
        self.published_rates = rates;
        self.published_velocities = velocities;
        Ok(())
    }

    pub(super) fn inherit_plan(
        &self,
        layout: &Layout,
        old: &Layout,
        source: &State,
        previous: Option<&Self>,
    ) -> Result<RateTransfers> {
        RateTransfers::remap(
            layout,
            old,
            source,
            previous.map_or(&[], |p| p.published_rates.as_slice()),
            previous.map_or(&[], |p| p.published_velocities.as_slice()),
        )
    }

    pub(super) fn inherit(
        &self,
        engine: &Engine,
        encoder: &mut wgpu::CommandEncoder,
        transfers: &RateTransfers,
        previous: Option<&Self>,
        side: usize,
    ) {
        let old = previous.map(|p| [&p.states[side][0], &p.states[side][1]]);
        for (lane, table) in [&transfers.rates, &transfers.velocities]
            .into_iter()
            .enumerate()
        {
            encode(
                engine,
                encoder,
                table,
                old,
                [&self.states[0][lane], &self.states[1][lane]],
            );
        }
    }
}
