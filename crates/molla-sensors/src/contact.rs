//! Public contact sensor — orchestrates a [`ContactAggregateKernel`]
//! dispatch and returns one [`BodyContactReading`] per body.
//!
//! GPU-first: the per-body aggregation runs in
//! `shaders/contact_aggregate.wgsl` (one thread per body, walks the
//! contact CSR). This module owns the four output GPU buffers
//! and downloads them into a host-side `Vec<BodyContactReading>`.
//!
//! # What it measures
//!
//! For each body, the sensor reports
//! - `count` — number of contacts the body participates in.
//! - `max_depth` — largest penetration depth across the body's
//!   contacts.
//! - `normal_sum` — Σ (sign × normal) summed over the body's
//!   contacts. molla's contact normal points a→b, so the sign is
//!   flipped when the body is the a-side of a contact; the result
//!   then points along the net contact-force direction acting on
//!   *this* body.
//! - `centroid` — mean contact point in world frame.
//!
//! This GPU sensor reports geometry, not forces. CPU Featherstone exposes
//! penalty forces through `FeatherstoneSolver::contact_forces`; `RigidWorld`
//! groups those readings by stable collider pair.

use molla_compute::Engine;
use molla_core::{Error, Result};
use molla_kernels::ContactAggregateKernel;
use molla_math::{Real, Vec3};
use molla_sim::{Contacts, Model};
use molla_storage::{Buffer, BufferUsage};

/// Per-body contact summary returned by [`ContactSensor::update`].
#[derive(Debug, Clone, Copy, Default)]
pub struct BodyContactReading {
    /// Number of contacts this body participates in this frame.
    pub count: u32,
    /// Largest penetration depth across the body's contacts (m).
    /// Zero when the body has no contacts.
    pub max_depth: Real,
    /// Sum of contact normals, sign-flipped on the body's a-side of
    /// each contact (the normal points a→b), so it points along the
    /// net push direction. Zero-vector when the body has no contacts.
    pub normal_sum: Vec3,
    /// Mean contact point in world frame. Zero when no contacts.
    pub centroid: Vec3,
}

/// GPU contact sensor. Runs a single dispatch per frame; reusable
/// across many `update` calls — only the output buffers need
/// rebuilding when `body_count` changes.
pub struct ContactSensor {
    kernel: ContactAggregateKernel,
    body_count: u32,
    out_count: Buffer<u32>,
    out_max_depth: Buffer<Real>,
    out_normal_sum: Buffer<Vec3>,
    out_centroid: Buffer<Vec3>,
}

impl ContactSensor {
    pub fn new(engine: &Engine, model: &Model) -> Result<Self> {
        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("contact sensor: engine has no GPU device".into()))?
            .clone();
        let queue = engine
            .queue()
            .ok_or_else(|| Error::Gpu("contact sensor: engine has no GPU queue".into()))?
            .clone();

        let n = model.body_count;
        let mut out_count = Buffer::from_vec(vec![0u32; n], BufferUsage::StorageReadWrite);
        let mut out_max_depth = Buffer::from_vec(vec![0.0_f64; n], BufferUsage::StorageReadWrite);
        let mut out_normal_sum =
            Buffer::from_vec(vec![Vec3::ZERO; n], BufferUsage::StorageReadWrite);
        let mut out_centroid = Buffer::from_vec(vec![Vec3::ZERO; n], BufferUsage::StorageReadWrite);

        out_count.ensure_device(&device)?;
        out_max_depth.ensure_device(&device)?;
        out_normal_sum.ensure_device(&device)?;
        out_centroid.ensure_device(&device)?;
        out_count.upload(&queue)?;
        out_max_depth.upload(&queue)?;
        out_normal_sum.upload(&queue)?;
        out_centroid.upload(&queue)?;

        Ok(Self {
            kernel: ContactAggregateKernel::new(&device),
            body_count: n as u32,
            out_count,
            out_max_depth,
            out_normal_sum,
            out_centroid,
        })
    }

    /// Number of bodies covered by the sensor (= `model.body_count`
    /// at construction time).
    pub fn body_count(&self) -> u32 {
        self.body_count
    }

    /// Run the kernel against the current `contacts` and return one
    /// reading per body (length = `body_count`).
    pub fn update(
        &mut self,
        engine: &Engine,
        contacts: &Contacts,
    ) -> Result<Vec<BodyContactReading>> {
        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("contact sensor: engine has no GPU device".into()))?
            .clone();
        let queue = engine
            .queue()
            .ok_or_else(|| Error::Gpu("contact sensor: engine has no GPU queue".into()))?
            .clone();

        self.kernel.dispatch(
            &device,
            &queue,
            contacts,
            self.body_count,
            &mut self.out_count,
            &mut self.out_max_depth,
            &mut self.out_normal_sum,
            &mut self.out_centroid,
        )?;

        self.out_count.download(&device, &queue)?;
        self.out_max_depth.download(&device, &queue)?;
        self.out_normal_sum.download(&device, &queue)?;
        self.out_centroid.download(&device, &queue)?;

        let count = self.out_count.host()?;
        let depth = self.out_max_depth.host()?;
        let normal = self.out_normal_sum.host()?;
        let centroid = self.out_centroid.host()?;

        let mut out = Vec::with_capacity(self.body_count as usize);
        for i in 0..self.body_count as usize {
            out.push(BodyContactReading {
                count: count[i],
                max_depth: depth[i],
                normal_sum: normal[i],
                centroid: centroid[i],
            });
        }
        Ok(out)
    }
}
