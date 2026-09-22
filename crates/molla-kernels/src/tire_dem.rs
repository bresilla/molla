//! GPU tire-contact impulse transfer for DEM/granular particles.
//!
//! This is the first GPU counterpart to `GranularBed::apply_tire_contact`:
//! tire contact cells are converted into world-space localized force patches,
//! and each grain gathers the impulses from nearby patches without atomics.
//! The module also contains a correctness-first O(N²) GPU DEM step and a
//! cell-grid broadphase GPU DEM step that add grain-grain and floor contacts
//! before tire impulses. The grid paths are the first large-bed scaling steps:
//! host-built sorted lists and an unbounded GPU-radix sorted list that stays
//! resident through the neighbor step. The older GPU-built fixed-capacity API
//! is retained as a compatibility wrapper but routes through the unbounded
//! path. Sparse LBVH beds now also have CPU/GPU traversal lanes plus a
//! surface-velocity-map export helper for FEM terrain feedback.

use bytemuck::{Pod, Zeroable};
use molla_core::{Error, Result};
use std::sync::Arc;

const WORKGROUP_SIZE: u32 = 64;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable, Default)]
pub struct TireDemContactCell {
    /// `(world_x, surface_y, world_z, influence_radius_m)`.
    pub center_radius: [f32; 4],
    /// World force on grains in newtons `(fx, fy, fz, unused)`.
    pub force: [f32; 4],
}

#[derive(Clone, Debug)]
pub struct TireDemGpuInput {
    pub positions: Vec<[f32; 4]>,
    pub velocities: Vec<[f32; 4]>,
    pub contacts: Vec<TireDemContactCell>,
    pub grain_mass: f32,
    pub grain_radius: f32,
    pub dt: f32,
    /// Position-level downward push scale for immediate visible rutting. This
    /// mirrors the CPU DEM coupling's capped normal push and is zero-safe.
    pub position_push_scale: f32,
}

#[derive(Clone, Debug)]
pub struct TireDemGpuOutput {
    pub positions: Vec<[f32; 4]>,
    pub velocities: Vec<[f32; 4]>,
}

/// GPU-resident DEM step output buffers.
///
/// This is the no-readback counterpart to [`TireDemGpuOutput`]. Callers that
/// are chaining DEM into GPU terrain/FEM contact stages can keep these buffers
/// resident; viewer/tests can explicitly cross the readback seam with
/// [`TireDemGpuKernel::readback_resident_output`].
pub struct TireDemResidentOutput {
    pub positions: wgpu::Buffer,
    pub velocities: wgpu::Buffer,
    pub grain_count: usize,
    pub dispatch_plan: TireDemLbvhDispatchPlan,
}

/// Persistent GPU-resident DEM grain state.
///
/// Positions and velocities can be ping-ponged through resident DEM kernels
/// without rewriting them from the CPU every frame. Sparse-LBVH topology can
/// now be built directly from these resident position buffers by
/// `build_resident_sparse_lbvh_tree_rows_gpu_radix`; CPU mirrors remain only
/// for bounds/debug/readback seams in the higher-level terrain owner.
pub struct TireDemResidentState {
    pub positions: wgpu::Buffer,
    pub velocities: wgpu::Buffer,
    pub grain_count: usize,
}

/// GPU-resident sparse-LBVH topology buffers for DEM neighbor traversal.
///
/// Callers can either upload CPU/reference topology rows for debug parity or
/// build the Morton order, sorted indices, leaf rows, and complete-binary tree
/// rows directly on GPU. The resident DEM step consumes the same buffer
/// contract in both cases, so topology construction can be moved off the hot
/// CPU path without changing the traversal kernel.
/// Outcome of [`TireDemGpuKernel::update_resident_lbvh`] — the two
/// non-update cases need DIFFERENT remedies.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ResidentLbvhUpdate {
    /// Topology uploaded; revision bumped.
    Updated,
    /// The resident state was built for a different grain count —
    /// rebuild the resident state, not just the LBVH buffers.
    GrainCountMismatch,
    /// Node/index rows exceed the resident capacities — reallocate
    /// (larger `TireDemResidentLbvh`) and retry.
    CapacityExceeded,
}

pub struct TireDemResidentLbvh {
    pub node_rows: wgpu::Buffer,
    pub sorted_indices: wgpu::Buffer,
    pub grain_count: usize,
    pub node_count: usize,
    pub node_row_count: usize,
    pub sorted_index_count: usize,
    /// Allocated `[f32; 4]` row capacity for `node_rows`.
    ///
    /// `node_row_count` is the active row count consumed by the shader; this
    /// capacity lets callers refresh a CPU-built LBVH into existing resident
    /// buffers when the topology still fits.
    pub node_row_capacity: usize,
    /// Allocated `u32` capacity for `sorted_indices`.
    pub sorted_index_capacity: usize,
    pub leaf_size: u32,
    /// Monotonic count of in-place topology refreshes after the initial upload.
    pub topology_revision: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TireDemResidentTerrainParams {
    pub origin: [f32; 2],
    pub spacing: f32,
    pub nx: u32,
    pub nz: u32,
    pub floor_y: f32,
    pub grain_radius: f32,
}

/// GPU-resident DEM-to-FEM terrain raster.
///
/// The buffers use the same row-major terrain contract as
/// `TireFemContactTerrain`: `heights` and `sinkage` are `f32` rows, while
/// `surface_velocity_rows` are `(vx, vy, vz, contributing_grain_count)`.
/// Downstream GPU FEM feedback can consume these buffers directly; tests and
/// viewer telemetry should cross the explicit readback seam.
pub struct TireDemResidentTerrainOutput {
    pub heights: wgpu::Buffer,
    pub sinkage: wgpu::Buffer,
    pub surface_velocity_rows: wgpu::Buffer,
    pub params: TireDemResidentTerrainParams,
}

#[derive(Clone, Debug)]
pub struct TireDemResidentTerrainReadback {
    pub heights: Vec<f32>,
    pub sinkage: Vec<f32>,
    pub surface_velocity_map: TireDemSurfaceVelocityMap,
}

/// Reusable GPU-resident DEM terrain owner for FEM feedback.
///
/// This bundles the resident grain buffers with the resident DEM-to-FEM
/// terrain raster. It still keeps an explicit CPU mirror for viewer/debug
/// telemetry, but the Morton quantization bounds are cached at construction so
/// repeated topology builds do not depend on per-step CPU grain positions.
/// Topology construction, the hot DEM step, and height/sinkage/surface-velocity
/// terrain export stay on resident GPU buffers.
pub struct TireDemResidentFemTerrainOwner {
    terrain_params: TireDemResidentTerrainParams,
    reference_heights: Vec<f32>,
    lbvh_scene_min: [f32; 3],
    lbvh_scene_extent: [f32; 3],
    state: TireDemResidentState,
    lbvh: Option<TireDemResidentLbvh>,
    terrain: TireDemResidentTerrainOutput,
    mirror_positions: Vec<[f32; 4]>,
    mirror_velocities: Vec<[f32; 4]>,
    reseed_count: u64,
}

impl TireDemResidentTerrainOutput {
    pub fn fem_contact_feedback_gpu_terrain(
        &self,
        friction: f32,
        soil_rolling_resistance_scale: f32,
    ) -> Result<crate::tire_fem::TireFemContactFeedbackGpuTerrain<'_>> {
        if !friction.is_finite() || friction < 0.0 {
            return Err(Error::Solver(
                "tire_dem_resident_terrain: friction must be finite and non-negative".into(),
            ));
        }
        if !soil_rolling_resistance_scale.is_finite() || soil_rolling_resistance_scale < 0.0 {
            return Err(Error::Solver(
                "tire_dem_resident_terrain: rolling-resistance scale must be finite and non-negative"
                    .into(),
            ));
        }
        Ok(crate::tire_fem::TireFemContactFeedbackGpuTerrain {
            origin: self.params.origin,
            spacing: self.params.spacing,
            nx: self.params.nx,
            nz: self.params.nz,
            heights: &self.heights,
            friction,
            friction_map: None,
            water_film_map: None,
            sinkage_map: Some(&self.sinkage),
            surface_velocity_map: Some(&self.surface_velocity_rows),
            soil_rolling_resistance_scale,
            surface_velocity: [0.0; 3],
        })
    }
}

impl TireDemResidentFemTerrainOwner {
    pub fn from_grain_state(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        kernel: &TireDemGpuKernel,
        terrain_params: TireDemResidentTerrainParams,
        positions: &[[f32; 4]],
        velocities: &[[f32; 4]],
    ) -> Result<Self> {
        if positions.len() != velocities.len() {
            return Err(Error::Solver(
                "tire_dem_resident_fem_owner: positions/velocities length mismatch".into(),
            ));
        }
        let reference_heights = tire_dem_raster_surface_height_map(
            positions,
            terrain_params.origin,
            terrain_params.spacing,
            terrain_params.nx,
            terrain_params.nz,
            terrain_params.floor_y,
            terrain_params.grain_radius,
        )?;
        validate_dem_resident_terrain_params(positions.len(), &reference_heights, terrain_params)?;
        let (lbvh_scene_min, lbvh_scene_extent) =
            tire_dem_resident_owner_lbvh_bounds(positions, terrain_params)?;
        let state = kernel.create_resident_state(device, queue, positions, velocities)?;
        let terrain = kernel.dispatch_resident_terrain_raster(
            device,
            queue,
            &state,
            &reference_heights,
            terrain_params,
        )?;
        Ok(Self {
            terrain_params,
            reference_heights,
            lbvh_scene_min,
            lbvh_scene_extent,
            state,
            lbvh: None,
            terrain,
            mirror_positions: positions.to_vec(),
            mirror_velocities: velocities.to_vec(),
            reseed_count: 0,
        })
    }

    /// Reseed the resident DEM/FEM terrain owner from CPU grain rows.
    ///
    /// This is the explicit reset/editor bridge: when the grain count is
    /// unchanged the existing resident position/velocity buffers are rewritten
    /// in place; when the count changes a new resident state is allocated. In
    /// both cases the reference terrain, cached Morton bounds, resident terrain
    /// raster, and CPU mirror are refreshed together, and stale LBVH topology is
    /// dropped so the next step rebuilds it on GPU from the reseeded state.
    pub fn reseed_grain_state(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        kernel: &TireDemGpuKernel,
        positions: &[[f32; 4]],
        velocities: &[[f32; 4]],
    ) -> Result<()> {
        if positions.len() != velocities.len() {
            return Err(Error::Solver(
                "tire_dem_resident_fem_owner_reseed: positions/velocities length mismatch".into(),
            ));
        }
        let reference_heights = tire_dem_raster_surface_height_map(
            positions,
            self.terrain_params.origin,
            self.terrain_params.spacing,
            self.terrain_params.nx,
            self.terrain_params.nz,
            self.terrain_params.floor_y,
            self.terrain_params.grain_radius,
        )?;
        validate_dem_resident_terrain_params(
            positions.len(),
            &reference_heights,
            self.terrain_params,
        )?;
        let (lbvh_scene_min, lbvh_scene_extent) =
            tire_dem_resident_owner_lbvh_bounds(positions, self.terrain_params)?;
        if self.state.grain_count == positions.len() {
            kernel.update_resident_state(queue, &self.state, positions, velocities)?;
        } else {
            self.state = kernel.create_resident_state(device, queue, positions, velocities)?;
        }
        let terrain = kernel.dispatch_resident_terrain_raster(
            device,
            queue,
            &self.state,
            &reference_heights,
            self.terrain_params,
        )?;
        self.reference_heights = reference_heights;
        self.lbvh_scene_min = lbvh_scene_min;
        self.lbvh_scene_extent = lbvh_scene_extent;
        self.lbvh = None;
        self.terrain = terrain;
        self.mirror_positions = positions.to_vec();
        self.mirror_velocities = velocities.to_vec();
        self.reseed_count = self.reseed_count.saturating_add(1);
        Ok(())
    }

    pub fn step_sparse_lbvh_gpu_mirrored(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        kernel: &TireDemGpuKernel,
        contacts: &[TireDemContactCell],
        params: TireDemStepParams,
        leaf_size: usize,
    ) -> Result<TireDemFemTerrainStepReport> {
        let before_positions = self.mirror_positions.clone();
        let leaf_size = leaf_size.clamp(1, 32);
        self.lbvh = Some(kernel.build_resident_sparse_lbvh_tree_rows_gpu_radix(
            device,
            queue,
            &self.state,
            self.lbvh_scene_min,
            self.lbvh_scene_extent,
            leaf_size,
        )?);
        let resident_lbvh = self
            .lbvh
            .as_ref()
            .expect("resident LBVH must exist after GPU topology build");
        let lbvh_node_count = resident_lbvh.node_count;
        let lbvh_leaf_count = self.state.grain_count.div_ceil(leaf_size);
        let lbvh_max_leaf_grains = self.state.grain_count.min(leaf_size) as u32;
        let next = kernel
            .dispatch_lbvh_step_resident_state_with_resident_lbvh(
                device,
                queue,
                &self.state,
                contacts,
                params,
                resident_lbvh,
            )?
            .into_state();
        let terrain = kernel.dispatch_resident_terrain_raster(
            device,
            queue,
            &next,
            &self.reference_heights,
            self.terrain_params,
        )?;
        let terrain_readback = kernel.readback_resident_terrain(device, queue, &terrain)?;
        let mirror = kernel.readback_resident_state(device, queue, &next)?;
        self.mirror_positions = mirror.positions;
        self.mirror_velocities = mirror.velocities;
        self.state = next;
        self.terrain = terrain;

        let max_position_delta_y_m = self
            .mirror_positions
            .iter()
            .zip(before_positions.iter())
            .map(|(after, before)| (after[1] - before[1]).abs())
            .fold(0.0_f32, f32::max);
        Ok(TireDemFemTerrainStepReport {
            grain_count: self.mirror_positions.len(),
            contact_count: contacts.len(),
            lbvh_node_count,
            lbvh_leaf_count,
            lbvh_max_leaf_grains,
            max_position_delta_y_m,
            active_velocity_cells: terrain_readback.surface_velocity_map.active_cells,
            max_surface_speed_m_per_s: terrain_readback.surface_velocity_map.max_speed_m_per_s,
            max_sinkage_m: terrain_readback
                .sinkage
                .iter()
                .copied()
                .fold(0.0_f32, f32::max),
        })
    }

    pub fn resident_state(&self) -> &TireDemResidentState {
        &self.state
    }

    pub fn resident_lbvh(&self) -> Option<&TireDemResidentLbvh> {
        self.lbvh.as_ref()
    }

    pub fn resident_terrain(&self) -> &TireDemResidentTerrainOutput {
        &self.terrain
    }

    pub fn reference_heights(&self) -> &[f32] {
        &self.reference_heights
    }

    pub fn lbvh_scene_bounds(&self) -> ([f32; 3], [f32; 3]) {
        (self.lbvh_scene_min, self.lbvh_scene_extent)
    }

    pub fn reseed_count(&self) -> u64 {
        self.reseed_count
    }

    pub fn positions(&self) -> &[[f32; 4]] {
        &self.mirror_positions
    }

    pub fn velocities(&self) -> &[[f32; 4]] {
        &self.mirror_velocities
    }
}

impl TireDemResidentOutput {
    pub fn into_state(self) -> TireDemResidentState {
        TireDemResidentState {
            positions: self.positions,
            velocities: self.velocities,
            grain_count: self.grain_count,
        }
    }
}

/// Row-major DEM surface-velocity map compatible with FEM tire terrain input.
///
/// Rows are `(vx, vy, vz, contributing_grain_count)` per X/Z terrain cell.
/// Only grains close to the top visible layer of each cell contribute, so
/// buried grains do not dominate the surface velocity seen by a tire contact
/// patch. This is a CPU/export helper for sparse DEM owners; GPU FEM contact
/// can consume `rows` as `TireFemContactTerrain::surface_velocity_map`.
#[derive(Clone, Debug, Default)]
pub struct TireDemSurfaceVelocityMap {
    pub origin: [f32; 2],
    pub spacing: f32,
    pub nx: u32,
    pub nz: u32,
    pub rows: Vec<[f32; 4]>,
    pub active_cells: u32,
    pub max_speed_m_per_s: f32,
    pub mean_speed_m_per_s: f32,
}

/// DEM-to-FEM terrain bridge for sparse granular beds.
///
/// This owner keeps a DEM grain cloud as the source of truth, rasterizes the
/// visible top layer into FEM contact terrain height/sinkage rows, and attaches
/// the top-layer velocity raster through
/// `tire_fem_contact_terrain_with_dem_surface_velocity`. It is intentionally a
/// small bridge object: DEM stepping can still come from the sparse-LBVH,
/// grid, or GPU paths, while FEM tire contact receives one consistent terrain
/// shape instead of viewer-local slip or rut math.
#[derive(Clone, Debug)]
pub struct TireDemFemTerrainOwner {
    origin: [f32; 2],
    spacing: f32,
    nx: u32,
    nz: u32,
    floor_y: f32,
    friction: f32,
    grain_radius: f32,
    reference_heights: Vec<f32>,
    heights: Vec<f32>,
    sinkage: Vec<f32>,
    surface_velocity_map: TireDemSurfaceVelocityMap,
    positions: Vec<[f32; 4]>,
    velocities: Vec<[f32; 4]>,
}

/// Summary for one owned DEM terrain step.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TireDemFemTerrainStepReport {
    pub grain_count: usize,
    pub contact_count: usize,
    pub lbvh_node_count: usize,
    pub lbvh_leaf_count: usize,
    pub lbvh_max_leaf_grains: u32,
    pub max_position_delta_y_m: f32,
    pub active_velocity_cells: u32,
    pub max_surface_speed_m_per_s: f32,
    pub max_sinkage_m: f32,
}

impl TireDemFemTerrainOwner {
    pub fn from_grain_state(
        origin: [f32; 2],
        spacing: f32,
        nx: u32,
        nz: u32,
        floor_y: f32,
        friction: f32,
        grain_radius: f32,
        positions: &[[f32; 4]],
        velocities: &[[f32; 4]],
    ) -> Result<Self> {
        validate_dem_fem_terrain_grid(origin, spacing, nx, nz, floor_y, friction, grain_radius)?;
        let heights = tire_dem_raster_surface_height_map(
            positions,
            origin,
            spacing,
            nx,
            nz,
            floor_y,
            grain_radius,
        )?;
        let surface_velocity_map = tire_dem_raster_surface_velocity_map(
            positions,
            velocities,
            origin,
            spacing,
            nx,
            nz,
            grain_radius,
        )?;
        Ok(Self {
            origin,
            spacing,
            nx,
            nz,
            floor_y,
            friction,
            grain_radius,
            reference_heights: heights.clone(),
            heights,
            sinkage: vec![0.0; nx as usize * nz as usize],
            surface_velocity_map,
            positions: positions.to_vec(),
            velocities: velocities.to_vec(),
        })
    }

    pub fn update_grain_state(
        &mut self,
        positions: &[[f32; 4]],
        velocities: &[[f32; 4]],
    ) -> Result<()> {
        let heights = tire_dem_raster_surface_height_map(
            positions,
            self.origin,
            self.spacing,
            self.nx,
            self.nz,
            self.floor_y,
            self.grain_radius,
        )?;
        let surface_velocity_map = tire_dem_raster_surface_velocity_map(
            positions,
            velocities,
            self.origin,
            self.spacing,
            self.nx,
            self.nz,
            self.grain_radius,
        )?;
        self.sinkage = self
            .reference_heights
            .iter()
            .zip(heights.iter())
            .map(|(&reference, &height)| (reference - height).max(0.0))
            .collect();
        self.heights = heights;
        self.surface_velocity_map = surface_velocity_map;
        self.positions = positions.to_vec();
        self.velocities = velocities.to_vec();
        Ok(())
    }

    pub fn step_sparse_lbvh_reference(
        &mut self,
        contacts: &[TireDemContactCell],
        params: TireDemStepParams,
    ) -> Result<TireDemFemTerrainStepReport> {
        let before = self.positions.clone();
        let input = TireDemStepGpuInput {
            positions: self.positions.clone(),
            velocities: self.velocities.clone(),
            contacts: contacts.to_vec(),
            params,
        };
        let (out, lbvh) = tire_dem_reference_sparse_lbvh_step(&input)?;
        self.update_grain_state(&out.positions, &out.velocities)?;
        Ok(self.step_report(&before, contacts.len(), &lbvh))
    }

    #[cfg(feature = "gpu")]
    pub fn step_sparse_lbvh_gpu_blocking(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        kernel: &TireDemGpuKernel,
        contacts: &[TireDemContactCell],
        params: TireDemStepParams,
        leaf_size: usize,
    ) -> Result<TireDemFemTerrainStepReport> {
        let before = self.positions.clone();
        let input = tire_dem_build_sparse_lbvh_step_input(
            self.positions.clone(),
            self.velocities.clone(),
            contacts.to_vec(),
            params,
            leaf_size,
        )?;
        let lbvh = input.lbvh.clone();
        let out = kernel.dispatch_lbvh_step_blocking(device, queue, &input)?;
        self.update_grain_state(&out.positions, &out.velocities)?;
        Ok(self.step_report(&before, contacts.len(), &lbvh))
    }

    fn step_report(
        &self,
        before_positions: &[[f32; 4]],
        contact_count: usize,
        lbvh: &TireDemSparseLbvh,
    ) -> TireDemFemTerrainStepReport {
        let lbvh_leaf_count = lbvh.nodes.iter().filter(|node| node.is_leaf()).count();
        let lbvh_max_leaf_grains = lbvh
            .nodes
            .iter()
            .filter(|node| node.is_leaf())
            .map(|node| node.count)
            .max()
            .unwrap_or(0);
        let max_position_delta_y_m = self
            .positions
            .iter()
            .zip(before_positions.iter())
            .map(|(after, before)| (after[1] - before[1]).abs())
            .fold(0.0_f32, f32::max);
        TireDemFemTerrainStepReport {
            grain_count: self.positions.len(),
            contact_count,
            lbvh_node_count: lbvh.nodes.len(),
            lbvh_leaf_count,
            lbvh_max_leaf_grains,
            max_position_delta_y_m,
            active_velocity_cells: self.surface_velocity_map.active_cells,
            max_surface_speed_m_per_s: self.surface_velocity_map.max_speed_m_per_s,
            max_sinkage_m: self.sinkage.iter().copied().fold(0.0_f32, f32::max),
        }
    }

    pub fn fem_contact_terrain(&self) -> Result<crate::tire_fem::TireFemContactTerrain> {
        let terrain = crate::tire_fem::TireFemContactTerrain {
            origin: self.origin,
            spacing: self.spacing,
            nx: self.nx,
            nz: self.nz,
            heights: self.heights.clone(),
            friction: self.friction.max(0.0),
            friction_map: None,
            water_film_map: None,
            sinkage_map: Some(self.sinkage.clone()),
            soil_rolling_resistance_scale: 1.0,
            surface_velocity: [0.0; 3],
            surface_velocity_map: None,
            sdf_primitives: Vec::new(),
            triangles: Vec::new(),
        };
        crate::tire_fem::tire_fem_contact_terrain_with_dem_surface_velocity(
            terrain,
            &self.surface_velocity_map,
        )
    }

    pub fn heights(&self) -> &[f32] {
        &self.heights
    }

    pub fn sinkage(&self) -> &[f32] {
        &self.sinkage
    }

    pub fn surface_velocity_map(&self) -> &TireDemSurfaceVelocityMap {
        &self.surface_velocity_map
    }

    pub fn positions(&self) -> &[[f32; 4]] {
        &self.positions
    }

    pub fn velocities(&self) -> &[[f32; 4]] {
        &self.velocities
    }

    /// Heightfield origin in the XZ plane (matches `tire_dem_raster_surface_height_map`).
    pub fn origin(&self) -> [f32; 2] {
        self.origin
    }

    /// Heightfield cell spacing in metres.
    pub fn spacing(&self) -> f32 {
        self.spacing
    }

    /// Heightfield extent along X (cell count).
    pub fn nx(&self) -> u32 {
        self.nx
    }

    /// Heightfield extent along Z (cell count).
    pub fn nz(&self) -> u32 {
        self.nz
    }
}

#[derive(Clone, Copy, Debug)]
pub struct TireDemStepParams {
    pub grain_mass: f32,
    pub grain_radius: f32,
    pub normal_stiffness: f32,
    pub damping: f32,
    pub friction: f32,
    pub floor_y: f32,
    pub gravity: [f32; 3],
    pub dt: f32,
    pub tire_position_push_scale: f32,
}

impl Default for TireDemStepParams {
    fn default() -> Self {
        Self {
            grain_mass: 1.5,
            grain_radius: 0.05,
            normal_stiffness: 1.0e4,
            damping: 80.0,
            friction: 0.65,
            floor_y: 0.0,
            gravity: [0.0, -9.81, 0.0],
            dt: 5.0e-4,
            tire_position_push_scale: 0.20,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TireDemStepGpuInput {
    pub positions: Vec<[f32; 4]>,
    pub velocities: Vec<[f32; 4]>,
    pub contacts: Vec<TireDemContactCell>,
    pub params: TireDemStepParams,
}

#[derive(Clone, Debug)]
pub struct TireDemGridStepGpuInput {
    pub positions: Vec<[f32; 4]>,
    pub velocities: Vec<[f32; 4]>,
    pub contacts: Vec<TireDemContactCell>,
    pub params: TireDemStepParams,
    pub grid_origin: [f32; 3],
    pub cell_size: f32,
    pub grid_shape: [u32; 3],
    /// Grain ids sorted by cell. May be built on the host or by the unbounded
    /// GPU-radix staging path.
    pub sorted_indices: Vec<u32>,
    /// Per-cell `[start, end, count, unused]` ranges into `sorted_indices`.
    pub cell_ranges: Vec<[u32; 4]>,
}

#[derive(Clone, Debug)]
pub struct TireDemGpuBuiltGridStepInput {
    pub positions: Vec<[f32; 4]>,
    pub velocities: Vec<[f32; 4]>,
    pub contacts: Vec<TireDemContactCell>,
    pub params: TireDemStepParams,
    pub grid_origin: [f32; 3],
    pub cell_size: f32,
    pub grid_shape: [u32; 3],
    /// Historical fixed capacity per grid cell. Kept for API compatibility;
    /// dispatch now routes through the unbounded GPU-radix grid step so dense
    /// cells are not truncated by this cap.
    pub max_grains_per_cell: u32,
}

#[derive(Clone, Debug)]
pub struct TireDemGpuRadixGridStepInput {
    pub positions: Vec<[f32; 4]>,
    pub velocities: Vec<[f32; 4]>,
    pub contacts: Vec<TireDemContactCell>,
    pub params: TireDemStepParams,
    pub grid_origin: [f32; 3],
    pub cell_size: f32,
    pub grid_shape: [u32; 3],
}

#[derive(Clone, Debug)]
pub struct TireDemLbvhStepGpuInput {
    pub positions: Vec<[f32; 4]>,
    pub velocities: Vec<[f32; 4]>,
    pub contacts: Vec<TireDemContactCell>,
    pub params: TireDemStepParams,
    pub lbvh: TireDemSparseLbvh,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TireDemLbvhDispatchPlan {
    pub grain_count: usize,
    pub contact_count: usize,
    pub node_count: usize,
    pub node_row_count: usize,
    pub workgroup_count: u32,
    pub storage_bytes: u64,
}

impl TireDemLbvhDispatchPlan {
    pub fn is_nonempty(self) -> bool {
        self.grain_count > 0 && self.node_count > 0 && self.node_row_count == self.node_count * 3
    }
}

#[derive(Clone, Debug)]
pub struct TireDemUnboundedSortedGrid {
    /// Sorted `(cell_id, grain_id)` keys. This is the CPU oracle shape for the
    /// future GPU radix-sort builder: sort by `cell_id`, scan equal-key runs,
    /// and feed the resulting ranges into the existing grid-neighbor step.
    pub sorted_cell_keys: Vec<u32>,
    pub sorted_indices: Vec<u32>,
    pub cell_ranges: Vec<[u32; 4]>,
    pub max_cell_occupancy: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TireDemGpuMortonOrder {
    /// Morton keys generated on GPU and sorted by the reusable GPU radix sorter.
    pub sorted_morton_codes: Vec<u32>,
    /// Grain ids ordered by `sorted_morton_codes`.
    pub sorted_indices: Vec<u32>,
    /// Number of adjacent equal-key runs after sorting. Non-zero means the
    /// sorted code order is still valid, but equal-key grain-id tie-breaking
    /// remains host-defined until the full GPU LBVH builder owns stable ties.
    pub duplicate_morton_runs: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TireDemGpuLbvhLeafRows {
    pub sorted_morton_codes: Vec<u32>,
    pub sorted_indices: Vec<u32>,
    /// Three `[f32; 4]` rows per leaf: min, max, and `(start, count, is_leaf, 0)`.
    pub leaf_node_rows: Vec<[f32; 4]>,
    pub leaf_count: usize,
    pub leaf_size: u32,
    pub duplicate_morton_runs: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TireDemGpuLbvhTreeRows {
    pub sorted_morton_codes: Vec<u32>,
    pub sorted_indices: Vec<u32>,
    /// Full three-vec4 sparse-LBVH row atlas rooted at node 0.
    pub node_rows: Vec<[f32; 4]>,
    pub node_count: usize,
    pub leaf_count: usize,
    pub padded_leaf_count: usize,
    pub leaf_node_offset: usize,
    pub leaf_size: u32,
    pub duplicate_morton_runs: usize,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct TireDemLbvhNode {
    /// Center-position bounds for grains contained by this node.
    pub min: [f32; 3],
    pub max: [f32; 3],
    /// Child node indices for internal nodes. Leaves use `u32::MAX`.
    pub left: u32,
    pub right: u32,
    /// Range into [`TireDemSparseLbvh::sorted_indices`] for leaves.
    pub start: u32,
    pub count: u32,
}

impl TireDemLbvhNode {
    pub fn is_leaf(&self) -> bool {
        self.count > 0
    }
}

#[derive(Clone, Debug, Default)]
pub struct TireDemSparseLbvh {
    /// Morton-sorted grain ids.
    pub sorted_indices: Vec<u32>,
    /// Morton code for each sorted grain id.
    pub sorted_morton_codes: Vec<u32>,
    /// Binary LBVH-style tree over sorted grain centers.
    pub nodes: Vec<TireDemLbvhNode>,
    pub leaf_size: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct TireDemStepUniforms {
    dims: [u32; 4],
    params0: [f32; 4],
    params1: [f32; 4],
    gravity: [f32; 4],
    grid_origin_cell_size: [f32; 4],
    grid_shape: [u32; 4],
}

pub struct TireDemGpuKernel {
    pipeline: wgpu::ComputePipeline,
    step_pipeline: wgpu::ComputePipeline,
    grid_step_pipeline: wgpu::ComputePipeline,
    lbvh_step_pipeline: wgpu::ComputePipeline,
    terrain_raster_pipeline: wgpu::ComputePipeline,
    unbounded_key_pipeline: wgpu::ComputePipeline,
    unbounded_range_pipeline: wgpu::ComputePipeline,
    sparse_lbvh_morton_pipeline: wgpu::ComputePipeline,
    sparse_lbvh_leaf_pipeline: wgpu::ComputePipeline,
    sparse_lbvh_internal_pipeline: wgpu::ComputePipeline,
    layout_io: wgpu::BindGroupLayout,
    layout_grid_io: wgpu::BindGroupLayout,
    layout_lbvh_io: wgpu::BindGroupLayout,
    layout_terrain_io: wgpu::BindGroupLayout,
    layout_unbounded_grid_io: wgpu::BindGroupLayout,
    layout_sparse_lbvh_build_io: wgpu::BindGroupLayout,
    layout_uniforms: wgpu::BindGroupLayout,
    layout_step_uniforms: wgpu::BindGroupLayout,
}

impl TireDemGpuInput {
    fn validate(&self) -> Result<usize> {
        if self.positions.len() != self.velocities.len() {
            return Err(Error::Solver(
                "tire_dem: positions/velocities length mismatch".into(),
            ));
        }
        if self.positions.is_empty() {
            return Err(Error::Solver(
                "tire_dem: at least one grain required".into(),
            ));
        }
        if !self.grain_mass.is_finite() || self.grain_mass <= 0.0 {
            return Err(Error::Solver(
                "tire_dem: grain_mass must be positive".into(),
            ));
        }
        if !self.grain_radius.is_finite() || self.grain_radius <= 0.0 {
            return Err(Error::Solver(
                "tire_dem: grain_radius must be positive".into(),
            ));
        }
        if !self.dt.is_finite() || self.dt < 0.0 {
            return Err(Error::Solver(
                "tire_dem: dt must be finite and nonnegative".into(),
            ));
        }
        Ok(self.positions.len())
    }
}

impl TireDemStepGpuInput {
    fn validate(&self) -> Result<usize> {
        if self.positions.len() != self.velocities.len() {
            return Err(Error::Solver(
                "tire_dem_step: positions/velocities length mismatch".into(),
            ));
        }
        if self.positions.is_empty() {
            return Err(Error::Solver(
                "tire_dem_step: at least one grain required".into(),
            ));
        }
        if !self.params.grain_mass.is_finite() || self.params.grain_mass <= 0.0 {
            return Err(Error::Solver(
                "tire_dem_step: grain_mass must be positive".into(),
            ));
        }
        if !self.params.grain_radius.is_finite() || self.params.grain_radius <= 0.0 {
            return Err(Error::Solver(
                "tire_dem_step: grain_radius must be positive".into(),
            ));
        }
        if !self.params.dt.is_finite() || self.params.dt < 0.0 {
            return Err(Error::Solver(
                "tire_dem_step: dt must be finite and nonnegative".into(),
            ));
        }
        Ok(self.positions.len())
    }
}

impl TireDemGridStepGpuInput {
    fn validate(&self) -> Result<usize> {
        if self.positions.len() != self.velocities.len() {
            return Err(Error::Solver(
                "tire_dem_grid_step: positions/velocities length mismatch".into(),
            ));
        }
        if self.positions.is_empty() {
            return Err(Error::Solver(
                "tire_dem_grid_step: at least one grain required".into(),
            ));
        }
        if !self.params.grain_mass.is_finite() || self.params.grain_mass <= 0.0 {
            return Err(Error::Solver(
                "tire_dem_grid_step: grain_mass must be positive".into(),
            ));
        }
        if !self.params.grain_radius.is_finite() || self.params.grain_radius <= 0.0 {
            return Err(Error::Solver(
                "tire_dem_grid_step: grain_radius must be positive".into(),
            ));
        }
        if !self.params.dt.is_finite() || self.params.dt < 0.0 {
            return Err(Error::Solver(
                "tire_dem_grid_step: dt must be finite and nonnegative".into(),
            ));
        }
        if !self.cell_size.is_finite() || self.cell_size <= 0.0 {
            return Err(Error::Solver(
                "tire_dem_grid_step: cell_size must be positive".into(),
            ));
        }
        if self.cell_size < self.params.grain_radius * 2.0 {
            return Err(Error::Solver(
                "tire_dem_grid_step: cell_size must be at least one grain diameter".into(),
            ));
        }
        if self.grid_shape.contains(&0) {
            return Err(Error::Solver(
                "tire_dem_grid_step: grid_shape axes must be nonzero".into(),
            ));
        }
        let cells = grid_cell_count(self.grid_shape)?;
        if self.cell_ranges.len() != cells {
            return Err(Error::Solver(format!(
                "tire_dem_grid_step: expected {cells} cell ranges, got {}",
                self.cell_ranges.len()
            )));
        }
        if self.sorted_indices.len() != self.positions.len() {
            return Err(Error::Solver(format!(
                "tire_dem_grid_step: expected {} sorted grain ids, got {}",
                self.positions.len(),
                self.sorted_indices.len()
            )));
        }
        for (cell, range) in self.cell_ranges.iter().enumerate() {
            if range[0] > range[1] || range[1] as usize > self.sorted_indices.len() {
                return Err(Error::Solver(format!(
                    "tire_dem_grid_step: invalid range for cell {cell}: {range:?}"
                )));
            }
        }
        for &idx in &self.sorted_indices {
            if idx as usize >= self.positions.len() {
                return Err(Error::Solver(format!(
                    "tire_dem_grid_step: sorted index {idx} out of range {}",
                    self.positions.len()
                )));
            }
        }
        Ok(self.positions.len())
    }
}

impl TireDemGpuBuiltGridStepInput {
    fn validate(&self) -> Result<usize> {
        if self.positions.len() != self.velocities.len() {
            return Err(Error::Solver(
                "tire_dem_gpu_grid_step: positions/velocities length mismatch".into(),
            ));
        }
        if self.positions.is_empty() {
            return Err(Error::Solver(
                "tire_dem_gpu_grid_step: at least one grain required".into(),
            ));
        }
        if !self.params.grain_mass.is_finite() || self.params.grain_mass <= 0.0 {
            return Err(Error::Solver(
                "tire_dem_gpu_grid_step: grain_mass must be positive".into(),
            ));
        }
        if !self.params.grain_radius.is_finite() || self.params.grain_radius <= 0.0 {
            return Err(Error::Solver(
                "tire_dem_gpu_grid_step: grain_radius must be positive".into(),
            ));
        }
        if !self.params.dt.is_finite() || self.params.dt < 0.0 {
            return Err(Error::Solver(
                "tire_dem_gpu_grid_step: dt must be finite and nonnegative".into(),
            ));
        }
        if !self.cell_size.is_finite() || self.cell_size <= 0.0 {
            return Err(Error::Solver(
                "tire_dem_gpu_grid_step: cell_size must be positive".into(),
            ));
        }
        if self.cell_size < self.params.grain_radius * 2.0 {
            return Err(Error::Solver(
                "tire_dem_gpu_grid_step: cell_size must be at least one grain diameter".into(),
            ));
        }
        if self.grid_shape.contains(&0) {
            return Err(Error::Solver(
                "tire_dem_gpu_grid_step: grid_shape axes must be nonzero".into(),
            ));
        }
        if self.max_grains_per_cell == 0 {
            return Err(Error::Solver(
                "tire_dem_gpu_grid_step: max_grains_per_cell must be nonzero".into(),
            ));
        }
        let cells = grid_cell_count(self.grid_shape)?;
        let slots = cells
            .checked_mul(self.max_grains_per_cell as usize)
            .ok_or_else(|| {
                Error::Solver("tire_dem_gpu_grid_step: cell slot count overflow".into())
            })?;
        if slots == 0 {
            return Err(Error::Solver(
                "tire_dem_gpu_grid_step: at least one grid slot required".into(),
            ));
        }
        Ok(self.positions.len())
    }
}

impl TireDemGpuRadixGridStepInput {
    fn validate(&self) -> Result<usize> {
        if self.positions.len() != self.velocities.len() {
            return Err(Error::Solver(
                "tire_dem_gpu_radix_grid_step: positions/velocities length mismatch".into(),
            ));
        }
        if self.positions.is_empty() {
            return Err(Error::Solver(
                "tire_dem_gpu_radix_grid_step: at least one grain required".into(),
            ));
        }
        if !self.params.grain_mass.is_finite() || self.params.grain_mass <= 0.0 {
            return Err(Error::Solver(
                "tire_dem_gpu_radix_grid_step: grain_mass must be positive".into(),
            ));
        }
        if !self.params.grain_radius.is_finite() || self.params.grain_radius <= 0.0 {
            return Err(Error::Solver(
                "tire_dem_gpu_radix_grid_step: grain_radius must be positive".into(),
            ));
        }
        if !self.params.dt.is_finite() || self.params.dt < 0.0 {
            return Err(Error::Solver(
                "tire_dem_gpu_radix_grid_step: dt must be finite and nonnegative".into(),
            ));
        }
        if !self.cell_size.is_finite() || self.cell_size <= 0.0 {
            return Err(Error::Solver(
                "tire_dem_gpu_radix_grid_step: cell_size must be positive".into(),
            ));
        }
        if self.cell_size < self.params.grain_radius * 2.0 {
            return Err(Error::Solver(
                "tire_dem_gpu_radix_grid_step: cell_size must be at least one grain diameter"
                    .into(),
            ));
        }
        if self.grid_shape.contains(&0) {
            return Err(Error::Solver(
                "tire_dem_gpu_radix_grid_step: grid_shape axes must be nonzero".into(),
            ));
        }
        grid_cell_count(self.grid_shape)?;
        Ok(self.positions.len())
    }
}

impl TireDemLbvhStepGpuInput {
    fn validate(&self) -> Result<usize> {
        if self.positions.len() != self.velocities.len() {
            return Err(Error::Solver(
                "tire_dem_lbvh_step: positions/velocities length mismatch".into(),
            ));
        }
        if self.positions.is_empty() {
            return Err(Error::Solver(
                "tire_dem_lbvh_step: at least one grain required".into(),
            ));
        }
        if !self.params.grain_mass.is_finite() || self.params.grain_mass <= 0.0 {
            return Err(Error::Solver(
                "tire_dem_lbvh_step: grain_mass must be positive".into(),
            ));
        }
        if !self.params.grain_radius.is_finite() || self.params.grain_radius <= 0.0 {
            return Err(Error::Solver(
                "tire_dem_lbvh_step: grain_radius must be positive".into(),
            ));
        }
        if !self.params.dt.is_finite() || self.params.dt < 0.0 {
            return Err(Error::Solver(
                "tire_dem_lbvh_step: dt must be finite and nonnegative".into(),
            ));
        }
        if self.lbvh.nodes.is_empty() || self.lbvh.sorted_indices.len() != self.positions.len() {
            return Err(Error::Solver(
                "tire_dem_lbvh_step: lbvh must contain one sorted index per grain".into(),
            ));
        }
        for &idx in &self.lbvh.sorted_indices {
            if idx as usize >= self.positions.len() {
                return Err(Error::Solver(format!(
                    "tire_dem_lbvh_step: lbvh sorted index {idx} out of range {}",
                    self.positions.len()
                )));
            }
        }
        Ok(self.positions.len())
    }
}

fn grid_cell_count(shape: [u32; 3]) -> Result<usize> {
    let xy = shape[0]
        .checked_mul(shape[1])
        .ok_or_else(|| Error::Solver("tire_dem_grid: grid_shape x*y overflow".into()))?;
    let xyz = xy
        .checked_mul(shape[2])
        .ok_or_else(|| Error::Solver("tire_dem_grid: grid_shape x*y*z overflow".into()))?;
    Ok(xyz as usize)
}

fn grid_cell_coords(
    position: [f32; 4],
    origin: [f32; 3],
    cell_size: f32,
    shape: [u32; 3],
) -> [u32; 3] {
    let coord = |axis: usize| {
        (((position[axis] - origin[axis]) / cell_size).floor() as i32)
            .clamp(0, shape[axis] as i32 - 1) as u32
    };
    [coord(0), coord(1), coord(2)]
}

fn grid_cell_id(coords: [u32; 3], shape: [u32; 3]) -> usize {
    (coords[0] + coords[1] * shape[0] + coords[2] * shape[0] * shape[1]) as usize
}

fn expand_bits_10(mut v: u32) -> u32 {
    v &= 0x3ff;
    v = (v | (v << 16)) & 0x0300_00ff;
    v = (v | (v << 8)) & 0x0300_f00f;
    v = (v | (v << 4)) & 0x030c_30c3;
    v = (v | (v << 2)) & 0x0924_9249;
    v
}

fn morton3_10(x: u32, y: u32, z: u32) -> u32 {
    expand_bits_10(x) | (expand_bits_10(y) << 1) | (expand_bits_10(z) << 2)
}

fn dem_center_bounds(positions: &[[f32; 4]], indices: &[u32]) -> ([f32; 3], [f32; 3]) {
    let first = positions[indices[0] as usize];
    let mut min = [first[0], first[1], first[2]];
    let mut max = min;
    for &idx in &indices[1..] {
        let p = positions[idx as usize];
        for axis in 0..3 {
            min[axis] = min[axis].min(p[axis]);
            max[axis] = max[axis].max(p[axis]);
        }
    }
    (min, max)
}

fn tire_dem_scene_bounds_extent(
    positions: &[[f32; 4]],
    label: &str,
) -> Result<([f32; 3], [f32; 3])> {
    if positions.is_empty() {
        return Err(Error::Solver(format!(
            "{label}: at least one grain required for sparse-LBVH bounds"
        )));
    }
    if positions.len() > u32::MAX as usize {
        return Err(Error::Solver(format!(
            "{label}: grain count exceeds u32 indexing for sparse-LBVH bounds"
        )));
    }
    if positions.iter().flatten().any(|v| !v.is_finite()) {
        return Err(Error::Solver(format!(
            "{label}: grain positions must be finite for sparse-LBVH bounds"
        )));
    }
    let all_indices = (0..positions.len() as u32).collect::<Vec<_>>();
    let (scene_min, scene_max) = dem_center_bounds(positions, &all_indices);
    let scene_extent = [
        (scene_max[0] - scene_min[0]).max(1.0e-9),
        (scene_max[1] - scene_min[1]).max(1.0e-9),
        (scene_max[2] - scene_min[2]).max(1.0e-9),
    ];
    Ok((scene_min, scene_extent))
}

fn tire_dem_resident_owner_lbvh_bounds(
    positions: &[[f32; 4]],
    params: TireDemResidentTerrainParams,
) -> Result<([f32; 3], [f32; 3])> {
    let (grain_min, grain_extent) =
        tire_dem_scene_bounds_extent(positions, "tire_dem_resident_fem_owner")?;
    let grain_max = [
        grain_min[0] + grain_extent[0],
        grain_min[1] + grain_extent[1],
        grain_min[2] + grain_extent[2],
    ];
    let margin = (params.spacing.max(params.grain_radius * 8.0)).max(1.0e-3);
    let grid_min_x = params.origin[0] - margin;
    let grid_max_x = params.origin[0] + params.spacing * params.nx as f32 + margin;
    let grid_min_z = params.origin[1] - margin;
    let grid_max_z = params.origin[1] + params.spacing * params.nz as f32 + margin;
    let min = [
        grain_min[0].min(grid_min_x),
        grain_min[1].min(params.floor_y) - margin,
        grain_min[2].min(grid_min_z),
    ];
    let max = [
        grain_max[0].max(grid_max_x),
        grain_max[1].max(params.floor_y + margin) + margin,
        grain_max[2].max(grid_max_z),
    ];
    if min.iter().any(|v| !v.is_finite()) || max.iter().any(|v| !v.is_finite()) {
        return Err(Error::Solver(
            "tire_dem_resident_fem_owner: cached LBVH bounds must be finite".into(),
        ));
    }
    Ok((
        min,
        [
            (max[0] - min[0]).max(1.0e-9),
            (max[1] - min[1]).max(1.0e-9),
            (max[2] - min[2]).max(1.0e-9),
        ],
    ))
}

fn aabb_distance2_to_point(min: [f32; 3], max: [f32; 3], p: [f32; 3]) -> f32 {
    let mut d2 = 0.0;
    for axis in 0..3 {
        let d = if p[axis] < min[axis] {
            min[axis] - p[axis]
        } else if p[axis] > max[axis] {
            p[axis] - max[axis]
        } else {
            0.0
        };
        d2 += d * d;
    }
    d2
}

/// Build a deterministic Morton-sorted sparse LBVH over DEM grain centres.
///
/// This is the sparse-bed counterpart to the regular grid builders. It does
/// not quantize the bed into a dense world grid; instead it builds a compact
/// binary hierarchy over occupied grain positions and stores only one sorted
/// index per grain plus tree nodes. The current CPU reference uses it to prune
/// far grain pairs, and the node/index layout is intentionally simple to mirror
/// into a future GPU sparse-bed traversal.
pub fn tire_dem_build_sparse_lbvh(
    positions: &[[f32; 4]],
    leaf_size: usize,
) -> Result<TireDemSparseLbvh> {
    if positions.is_empty() {
        return Err(Error::Solver(
            "tire_dem_sparse_lbvh_build: at least one grain required".into(),
        ));
    }
    let leaf_size = leaf_size.clamp(1, 32);
    let all_indices = (0..positions.len() as u32).collect::<Vec<_>>();
    let (scene_min, scene_max) = dem_center_bounds(positions, &all_indices);
    let extent = [
        (scene_max[0] - scene_min[0]).max(1.0e-9),
        (scene_max[1] - scene_min[1]).max(1.0e-9),
        (scene_max[2] - scene_min[2]).max(1.0e-9),
    ];
    let mut pairs = positions
        .iter()
        .enumerate()
        .map(|(idx, p)| {
            let q = |axis: usize| {
                // round_ties_even matches WGSL's round() (half-to-even);
                // Rust's round() is half-AWAY-from-zero — a grain
                // landing exactly on x.5 would quantize differently on
                // CPU vs GPU and break the bitwise sorted-Morton parity
                // contracts.
                (((p[axis] - scene_min[axis]) / extent[axis]).clamp(0.0, 1.0) * 1023.0)
                    .round_ties_even() as u32
            };
            (morton3_10(q(0), q(1), q(2)), idx as u32)
        })
        .collect::<Vec<_>>();
    pairs.sort_by_key(|&(code, idx)| (code, idx));
    let (sorted_morton_codes, sorted_indices): (Vec<_>, Vec<_>) = pairs.into_iter().unzip();
    let mut nodes = Vec::new();
    build_dem_lbvh_node(
        positions,
        &sorted_indices,
        0,
        sorted_indices.len(),
        leaf_size,
        &mut nodes,
    );
    Ok(TireDemSparseLbvh {
        sorted_indices,
        sorted_morton_codes,
        nodes,
        leaf_size: leaf_size as u32,
    })
}

/// Build the sparse-LBVH Morton ordering stage on GPU.
///
/// This is not the full GPU LBVH builder yet: node bounds and tree rows are
/// still produced by [`tire_dem_build_sparse_lbvh`]. It moves the first
/// topology-construction stage onto GPU, using the same 10-bit-per-axis Morton
/// quantization as the CPU builder plus the reusable GPU radix sorter. The
/// returned order is read back for now so tests can pin CPU/GPU equivalence
/// before the next pass keeps the sorted order resident and builds node ranges
/// without host involvement.
pub fn tire_dem_build_sparse_lbvh_morton_order_gpu_radix(
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    positions: Vec<[f32; 4]>,
) -> Result<TireDemGpuMortonOrder> {
    if positions.is_empty() {
        return Err(Error::Solver(
            "tire_dem_gpu_sparse_lbvh_morton_build: at least one grain required".into(),
        ));
    }
    if positions.iter().flatten().any(|v| !v.is_finite()) {
        return Err(Error::Solver(
            "tire_dem_gpu_sparse_lbvh_morton_build: positions must be finite".into(),
        ));
    }
    let n = positions.len();
    if n > u32::MAX as usize {
        return Err(Error::Solver(
            "tire_dem_gpu_sparse_lbvh_morton_build: grain count exceeds u32 indexing".into(),
        ));
    }

    let all_indices = (0..n as u32).collect::<Vec<_>>();
    let (scene_min, scene_max) = dem_center_bounds(&positions, &all_indices);
    let extent = [
        (scene_max[0] - scene_min[0]).max(1.0e-9),
        (scene_max[1] - scene_min[1]).max(1.0e-9),
        (scene_max[2] - scene_min[2]).max(1.0e-9),
    ];

    let u32_bytes = |len: usize| (len.max(1) * core::mem::size_of::<u32>()) as u64;
    let vec4_bytes = |len: usize| (len.max(1) * core::mem::size_of::<[f32; 4]>()) as u64;
    let storage =
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
    let pos_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tire_dem gpu sparse lbvh morton positions"),
        size: vec4_bytes(n),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let keys_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tire_dem gpu sparse lbvh morton keys"),
        size: u32_bytes(n),
        usage: storage,
        mapped_at_creation: false,
    });
    let values_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tire_dem gpu sparse lbvh morton grain ids"),
        size: u32_bytes(n),
        usage: storage,
        mapped_at_creation: false,
    });
    let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tire_dem gpu sparse lbvh morton uniforms"),
        size: core::mem::size_of::<TireDemStepUniforms>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let stage_keys = stage_u32(&device, "tire_dem gpu sparse lbvh stage morton keys", n);
    let stage_values = stage_u32(&device, "tire_dem gpu sparse lbvh stage grain ids", n);

    queue.write_buffer(&pos_buf, 0, bytemuck::cast_slice(&positions));
    queue.write_buffer(
        &uniforms,
        0,
        bytemuck::bytes_of(&TireDemStepUniforms {
            dims: [n as u32, 0, 0, 0],
            params0: [0.0; 4],
            params1: [extent[0], extent[1], extent[2], 0.0],
            gravity: [0.0; 4],
            grid_origin_cell_size: [scene_min[0], scene_min[1], scene_min[2], 0.0],
            grid_shape: [0; 4],
        }),
    );

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("molla::tire_dem gpu sparse lbvh morton"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/tire_dem.wgsl").into()),
    });
    let layout_data = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("molla::tire_dem gpu sparse lbvh morton data"),
        entries: &[
            storage_entry(0, true),
            storage_entry(5, false),
            storage_entry(9, false),
        ],
    });
    let layout_uniforms = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("molla::tire_dem gpu sparse lbvh morton uniforms"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("molla::tire_dem gpu sparse lbvh morton layout"),
        bind_group_layouts: &[Some(&layout_data), Some(&layout_uniforms)],
        immediate_size: 0,
    });
    let keygen_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("molla::tire_dem::build_sparse_lbvh_morton_keys"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("build_sparse_lbvh_morton_keys"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    let bg_data = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("molla::tire_dem gpu sparse lbvh morton data bg"),
        layout: &layout_data,
        entries: &[
            entry(0, &pos_buf),
            entry(5, &values_buf),
            entry(9, &keys_buf),
        ],
    });
    let bg_uniforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("molla::tire_dem gpu sparse lbvh morton uniforms bg"),
        layout: &layout_uniforms,
        entries: &[entry(0, &uniforms)],
    });

    let mut key_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("molla::tire_dem gpu sparse lbvh morton key encoder"),
    });
    {
        let mut pass = key_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("molla::tire_dem gpu sparse lbvh morton key pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&keygen_pipeline);
        pass.set_bind_group(0, &bg_data, &[]);
        pass.set_bind_group(1, &bg_uniforms, &[]);
        pass.dispatch_workgroups((n as u32).div_ceil(WORKGROUP_SIZE), 1, 1);
    }
    queue.submit(std::iter::once(key_encoder.finish()));

    molla_compute_primitives::radix_sort_pairs(
        Arc::clone(&device),
        Arc::clone(&queue),
        &keys_buf,
        &values_buf,
        n,
    )?;

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("molla::tire_dem gpu sparse lbvh morton readback encoder"),
    });
    encoder.copy_buffer_to_buffer(&keys_buf, 0, &stage_keys, 0, u32_bytes(n));
    encoder.copy_buffer_to_buffer(&values_buf, 0, &stage_values, 0, u32_bytes(n));
    queue.submit(std::iter::once(encoder.finish()));

    let sorted_morton_codes =
        read_u32(&device, &stage_keys, "tire_dem gpu sparse lbvh morton keys")?;
    let sorted_indices = read_u32(&device, &stage_values, "tire_dem gpu sparse lbvh grain ids")?;
    if !sorted_morton_codes.windows(2).all(|w| w[0] <= w[1]) {
        return Err(Error::Solver(
            "tire_dem_gpu_sparse_lbvh_morton_build: GPU radix keys are not sorted".into(),
        ));
    }
    let duplicate_morton_runs = sorted_morton_codes
        .windows(2)
        .filter(|w| w[0] == w[1])
        .count();
    Ok(TireDemGpuMortonOrder {
        sorted_morton_codes,
        sorted_indices,
        duplicate_morton_runs,
    })
}

/// Build GPU sparse-LBVH Morton order plus fixed-size leaf AABB rows.
///
/// This is the second GPU topology-construction stage after Morton sorting.
/// It still does not build the full internal binary tree on GPU, but it does
/// produce the leaf ranges/bounds the traversal tree is built from. The rows
/// use the same three-vec4 node contract as [`tire_dem_sparse_lbvh_gpu_rows`],
/// with one leaf per consecutive `leaf_size` run in sorted Morton order.
pub fn tire_dem_build_sparse_lbvh_leaf_rows_gpu_radix(
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    positions: Vec<[f32; 4]>,
    leaf_size: usize,
) -> Result<TireDemGpuLbvhLeafRows> {
    if positions.is_empty() {
        return Err(Error::Solver(
            "tire_dem_gpu_sparse_lbvh_leaf_build: at least one grain required".into(),
        ));
    }
    if positions.iter().flatten().any(|v| !v.is_finite()) {
        return Err(Error::Solver(
            "tire_dem_gpu_sparse_lbvh_leaf_build: positions must be finite".into(),
        ));
    }
    let n = positions.len();
    if n > u32::MAX as usize {
        return Err(Error::Solver(
            "tire_dem_gpu_sparse_lbvh_leaf_build: grain count exceeds u32 indexing".into(),
        ));
    }
    let leaf_size = leaf_size.clamp(1, 32);
    let leaf_count = n.div_ceil(leaf_size);

    let all_indices = (0..n as u32).collect::<Vec<_>>();
    let (scene_min, scene_max) = dem_center_bounds(&positions, &all_indices);
    let extent = [
        (scene_max[0] - scene_min[0]).max(1.0e-9),
        (scene_max[1] - scene_min[1]).max(1.0e-9),
        (scene_max[2] - scene_min[2]).max(1.0e-9),
    ];

    let u32_bytes = |len: usize| (len.max(1) * core::mem::size_of::<u32>()) as u64;
    let vec4_bytes = |len: usize| (len.max(1) * core::mem::size_of::<[f32; 4]>()) as u64;
    let storage =
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
    let pos_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tire_dem gpu sparse lbvh leaf positions"),
        size: vec4_bytes(n),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let keys_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tire_dem gpu sparse lbvh leaf morton keys"),
        size: u32_bytes(n),
        usage: storage,
        mapped_at_creation: false,
    });
    let values_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tire_dem gpu sparse lbvh leaf grain ids"),
        size: u32_bytes(n),
        usage: storage,
        mapped_at_creation: false,
    });
    let leaf_rows_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tire_dem gpu sparse lbvh leaf rows"),
        size: vec4_bytes(leaf_count * 3),
        usage: storage,
        mapped_at_creation: false,
    });
    let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tire_dem gpu sparse lbvh leaf uniforms"),
        size: core::mem::size_of::<TireDemStepUniforms>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let stage_keys = stage_u32(
        &device,
        "tire_dem gpu sparse lbvh leaf stage morton keys",
        n,
    );
    let stage_values = stage_u32(&device, "tire_dem gpu sparse lbvh leaf stage grain ids", n);
    let stage_rows = stage_vec4(
        &device,
        "tire_dem gpu sparse lbvh leaf stage rows",
        leaf_count * 3,
    );

    queue.write_buffer(&pos_buf, 0, bytemuck::cast_slice(&positions));
    queue.write_buffer(
        &uniforms,
        0,
        bytemuck::bytes_of(&TireDemStepUniforms {
            dims: [n as u32, leaf_count as u32, leaf_size as u32, 0],
            params0: [0.0; 4],
            params1: [extent[0], extent[1], extent[2], 0.0],
            gravity: [0.0; 4],
            grid_origin_cell_size: [scene_min[0], scene_min[1], scene_min[2], 0.0],
            grid_shape: [0; 4],
        }),
    );

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("molla::tire_dem gpu sparse lbvh leaf"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/tire_dem.wgsl").into()),
    });
    let layout_data = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("molla::tire_dem gpu sparse lbvh leaf data"),
        entries: &[
            storage_entry(0, true),
            storage_entry(3, false),
            storage_entry(5, false),
            storage_entry(9, false),
        ],
    });
    let layout_uniforms = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("molla::tire_dem gpu sparse lbvh leaf uniforms"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("molla::tire_dem gpu sparse lbvh leaf layout"),
        bind_group_layouts: &[Some(&layout_data), Some(&layout_uniforms)],
        immediate_size: 0,
    });
    let keygen_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("molla::tire_dem::build_sparse_lbvh_leaf_morton_keys"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("build_sparse_lbvh_morton_keys"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    let leaf_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("molla::tire_dem::build_sparse_lbvh_leaf_rows"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("build_sparse_lbvh_leaf_rows"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    let bg_data = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("molla::tire_dem gpu sparse lbvh leaf data bg"),
        layout: &layout_data,
        entries: &[
            entry(0, &pos_buf),
            entry(3, &leaf_rows_buf),
            entry(5, &values_buf),
            entry(9, &keys_buf),
        ],
    });
    let bg_uniforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("molla::tire_dem gpu sparse lbvh leaf uniforms bg"),
        layout: &layout_uniforms,
        entries: &[entry(0, &uniforms)],
    });

    let mut key_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("molla::tire_dem gpu sparse lbvh leaf key encoder"),
    });
    {
        let mut pass = key_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("molla::tire_dem gpu sparse lbvh leaf key pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&keygen_pipeline);
        pass.set_bind_group(0, &bg_data, &[]);
        pass.set_bind_group(1, &bg_uniforms, &[]);
        pass.dispatch_workgroups((n as u32).div_ceil(WORKGROUP_SIZE), 1, 1);
    }
    queue.submit(std::iter::once(key_encoder.finish()));

    molla_compute_primitives::radix_sort_pairs(
        Arc::clone(&device),
        Arc::clone(&queue),
        &keys_buf,
        &values_buf,
        n,
    )?;

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("molla::tire_dem gpu sparse lbvh leaf build/readback encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("molla::tire_dem gpu sparse lbvh leaf rows pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&leaf_pipeline);
        pass.set_bind_group(0, &bg_data, &[]);
        pass.set_bind_group(1, &bg_uniforms, &[]);
        pass.dispatch_workgroups((leaf_count as u32).div_ceil(WORKGROUP_SIZE), 1, 1);
    }
    encoder.copy_buffer_to_buffer(&keys_buf, 0, &stage_keys, 0, u32_bytes(n));
    encoder.copy_buffer_to_buffer(&values_buf, 0, &stage_values, 0, u32_bytes(n));
    encoder.copy_buffer_to_buffer(
        &leaf_rows_buf,
        0,
        &stage_rows,
        0,
        vec4_bytes(leaf_count * 3),
    );
    queue.submit(std::iter::once(encoder.finish()));

    let sorted_morton_codes = read_u32(
        &device,
        &stage_keys,
        "tire_dem gpu sparse lbvh leaf morton keys",
    )?;
    let sorted_indices = read_u32(&device, &stage_values, "tire_dem gpu sparse lbvh leaf ids")?;
    let leaf_node_rows = read_vec4(&device, &stage_rows, "tire_dem gpu sparse lbvh leaf rows")?;
    if !sorted_morton_codes.windows(2).all(|w| w[0] <= w[1]) {
        return Err(Error::Solver(
            "tire_dem_gpu_sparse_lbvh_leaf_build: GPU radix keys are not sorted".into(),
        ));
    }
    let duplicate_morton_runs = sorted_morton_codes
        .windows(2)
        .filter(|w| w[0] == w[1])
        .count();
    Ok(TireDemGpuLbvhLeafRows {
        sorted_morton_codes,
        sorted_indices,
        leaf_node_rows,
        leaf_count,
        leaf_size: leaf_size as u32,
        duplicate_morton_runs,
    })
}

/// Build a complete GPU sparse-LBVH row atlas.
///
/// This extends the GPU Morton and leaf-row stages with bottom-up internal
/// node construction. Non-power-of-two leaf counts are padded to the next
/// power of two with empty neutral leaves, so the traversal contract remains a
/// complete binary tree while real leaves keep their original sorted ranges.
pub fn tire_dem_build_sparse_lbvh_tree_rows_gpu_radix(
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    positions: Vec<[f32; 4]>,
    leaf_size: usize,
) -> Result<TireDemGpuLbvhTreeRows> {
    if positions.is_empty() {
        return Err(Error::Solver(
            "tire_dem_gpu_sparse_lbvh_tree_build: at least one grain required".into(),
        ));
    }
    if positions.iter().flatten().any(|v| !v.is_finite()) {
        return Err(Error::Solver(
            "tire_dem_gpu_sparse_lbvh_tree_build: positions must be finite".into(),
        ));
    }
    let n = positions.len();
    if n > u32::MAX as usize {
        return Err(Error::Solver(
            "tire_dem_gpu_sparse_lbvh_tree_build: grain count exceeds u32 indexing".into(),
        ));
    }
    let leaf_size = leaf_size.clamp(1, 32);
    let leaf_count = n.div_ceil(leaf_size);
    let padded_leaf_count = leaf_count.next_power_of_two();
    let leaf_node_offset = padded_leaf_count - 1;
    let node_count = padded_leaf_count * 2 - 1;

    let all_indices = (0..n as u32).collect::<Vec<_>>();
    let (scene_min, scene_max) = dem_center_bounds(&positions, &all_indices);
    let extent = [
        (scene_max[0] - scene_min[0]).max(1.0e-9),
        (scene_max[1] - scene_min[1]).max(1.0e-9),
        (scene_max[2] - scene_min[2]).max(1.0e-9),
    ];

    let u32_bytes = |len: usize| (len.max(1) * core::mem::size_of::<u32>()) as u64;
    let vec4_bytes = |len: usize| (len.max(1) * core::mem::size_of::<[f32; 4]>()) as u64;
    let storage =
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
    let pos_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tire_dem gpu sparse lbvh tree positions"),
        size: vec4_bytes(n),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let keys_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tire_dem gpu sparse lbvh tree morton keys"),
        size: u32_bytes(n),
        usage: storage,
        mapped_at_creation: false,
    });
    let values_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tire_dem gpu sparse lbvh tree grain ids"),
        size: u32_bytes(n),
        usage: storage,
        mapped_at_creation: false,
    });
    let node_rows_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tire_dem gpu sparse lbvh tree rows"),
        size: vec4_bytes(node_count * 3),
        usage: storage,
        mapped_at_creation: false,
    });
    let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tire_dem gpu sparse lbvh tree uniforms"),
        size: core::mem::size_of::<TireDemStepUniforms>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let stage_keys = stage_u32(
        &device,
        "tire_dem gpu sparse lbvh tree stage morton keys",
        n,
    );
    let stage_values = stage_u32(&device, "tire_dem gpu sparse lbvh tree stage grain ids", n);
    let stage_rows = stage_vec4(
        &device,
        "tire_dem gpu sparse lbvh tree stage rows",
        node_count * 3,
    );

    queue.write_buffer(&pos_buf, 0, bytemuck::cast_slice(&positions));
    queue.write_buffer(
        &uniforms,
        0,
        bytemuck::bytes_of(&TireDemStepUniforms {
            dims: [n as u32, padded_leaf_count as u32, leaf_size as u32, 0],
            params0: [0.0; 4],
            params1: [extent[0], extent[1], extent[2], 0.0],
            gravity: [0.0; 4],
            grid_origin_cell_size: [scene_min[0], scene_min[1], scene_min[2], 0.0],
            grid_shape: [
                leaf_node_offset as u32,
                0,
                node_count as u32,
                leaf_count as u32,
            ],
        }),
    );

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("molla::tire_dem gpu sparse lbvh tree"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/tire_dem.wgsl").into()),
    });
    let layout_data = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("molla::tire_dem gpu sparse lbvh tree data"),
        entries: &[
            storage_entry(0, true),
            storage_entry(3, false),
            storage_entry(5, false),
            storage_entry(9, false),
        ],
    });
    let layout_uniforms = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("molla::tire_dem gpu sparse lbvh tree uniforms"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("molla::tire_dem gpu sparse lbvh tree layout"),
        bind_group_layouts: &[Some(&layout_data), Some(&layout_uniforms)],
        immediate_size: 0,
    });
    let keygen_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("molla::tire_dem::build_sparse_lbvh_tree_morton_keys"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("build_sparse_lbvh_morton_keys"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    let leaf_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("molla::tire_dem::build_sparse_lbvh_tree_leaf_rows"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("build_sparse_lbvh_leaf_rows"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    let internal_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("molla::tire_dem::build_sparse_lbvh_tree_internal_level"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("build_sparse_lbvh_internal_level"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    let bg_data = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("molla::tire_dem gpu sparse lbvh tree data bg"),
        layout: &layout_data,
        entries: &[
            entry(0, &pos_buf),
            entry(3, &node_rows_buf),
            entry(5, &values_buf),
            entry(9, &keys_buf),
        ],
    });
    let bg_uniforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("molla::tire_dem gpu sparse lbvh tree uniforms bg"),
        layout: &layout_uniforms,
        entries: &[entry(0, &uniforms)],
    });

    let mut key_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("molla::tire_dem gpu sparse lbvh tree key encoder"),
    });
    {
        let mut pass = key_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("molla::tire_dem gpu sparse lbvh tree key pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&keygen_pipeline);
        pass.set_bind_group(0, &bg_data, &[]);
        pass.set_bind_group(1, &bg_uniforms, &[]);
        pass.dispatch_workgroups((n as u32).div_ceil(WORKGROUP_SIZE), 1, 1);
    }
    queue.submit(std::iter::once(key_encoder.finish()));

    molla_compute_primitives::radix_sort_pairs(
        Arc::clone(&device),
        Arc::clone(&queue),
        &keys_buf,
        &values_buf,
        n,
    )?;

    queue.write_buffer(
        &uniforms,
        0,
        bytemuck::bytes_of(&TireDemStepUniforms {
            dims: [n as u32, padded_leaf_count as u32, leaf_size as u32, 0],
            params0: [0.0; 4],
            params1: [extent[0], extent[1], extent[2], 0.0],
            gravity: [0.0; 4],
            grid_origin_cell_size: [scene_min[0], scene_min[1], scene_min[2], 0.0],
            grid_shape: [
                leaf_node_offset as u32,
                0,
                node_count as u32,
                leaf_count as u32,
            ],
        }),
    );
    let mut leaf_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("molla::tire_dem gpu sparse lbvh tree leaf encoder"),
    });
    {
        let mut pass = leaf_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("molla::tire_dem gpu sparse lbvh tree leaf pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&leaf_pipeline);
        pass.set_bind_group(0, &bg_data, &[]);
        pass.set_bind_group(1, &bg_uniforms, &[]);
        pass.dispatch_workgroups((padded_leaf_count as u32).div_ceil(WORKGROUP_SIZE), 1, 1);
    }
    queue.submit(std::iter::once(leaf_encoder.finish()));

    let mut child_start = leaf_node_offset;
    let mut child_count = padded_leaf_count;
    while child_count > 1 {
        let parent_count = child_count / 2;
        let parent_start = child_start - parent_count;
        queue.write_buffer(
            &uniforms,
            0,
            bytemuck::bytes_of(&TireDemStepUniforms {
                dims: [n as u32, parent_count as u32, 0, 0],
                params0: [0.0; 4],
                params1: [extent[0], extent[1], extent[2], 0.0],
                gravity: [0.0; 4],
                grid_origin_cell_size: [scene_min[0], scene_min[1], scene_min[2], 0.0],
                grid_shape: [
                    parent_start as u32,
                    child_start as u32,
                    node_count as u32,
                    0,
                ],
            }),
        );
        let mut level_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::tire_dem gpu sparse lbvh tree internal encoder"),
        });
        {
            let mut pass = level_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::tire_dem gpu sparse lbvh tree internal pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&internal_pipeline);
            pass.set_bind_group(0, &bg_data, &[]);
            pass.set_bind_group(1, &bg_uniforms, &[]);
            pass.dispatch_workgroups((parent_count as u32).div_ceil(WORKGROUP_SIZE), 1, 1);
        }
        queue.submit(std::iter::once(level_encoder.finish()));
        child_start = parent_start;
        child_count = parent_count;
    }

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("molla::tire_dem gpu sparse lbvh tree readback encoder"),
    });
    encoder.copy_buffer_to_buffer(&keys_buf, 0, &stage_keys, 0, u32_bytes(n));
    encoder.copy_buffer_to_buffer(&values_buf, 0, &stage_values, 0, u32_bytes(n));
    encoder.copy_buffer_to_buffer(
        &node_rows_buf,
        0,
        &stage_rows,
        0,
        vec4_bytes(node_count * 3),
    );
    queue.submit(std::iter::once(encoder.finish()));

    let sorted_morton_codes = read_u32(
        &device,
        &stage_keys,
        "tire_dem gpu sparse lbvh tree morton keys",
    )?;
    let sorted_indices = read_u32(&device, &stage_values, "tire_dem gpu sparse lbvh tree ids")?;
    let node_rows = read_vec4(&device, &stage_rows, "tire_dem gpu sparse lbvh tree rows")?;
    if !sorted_morton_codes.windows(2).all(|w| w[0] <= w[1]) {
        return Err(Error::Solver(
            "tire_dem_gpu_sparse_lbvh_tree_build: GPU radix keys are not sorted".into(),
        ));
    }
    let duplicate_morton_runs = sorted_morton_codes
        .windows(2)
        .filter(|w| w[0] == w[1])
        .count();
    Ok(TireDemGpuLbvhTreeRows {
        sorted_morton_codes,
        sorted_indices,
        node_rows,
        node_count,
        leaf_count,
        padded_leaf_count,
        leaf_node_offset,
        leaf_size: leaf_size as u32,
        duplicate_morton_runs,
    })
}

pub fn tire_dem_build_sparse_lbvh_step_input(
    positions: Vec<[f32; 4]>,
    velocities: Vec<[f32; 4]>,
    contacts: Vec<TireDemContactCell>,
    params: TireDemStepParams,
    leaf_size: usize,
) -> Result<TireDemLbvhStepGpuInput> {
    if positions.len() != velocities.len() {
        return Err(Error::Solver(
            "tire_dem_lbvh_step_build: positions/velocities length mismatch".into(),
        ));
    }
    let lbvh = tire_dem_build_sparse_lbvh(&positions, leaf_size)?;
    let input = TireDemLbvhStepGpuInput {
        positions,
        velocities,
        contacts,
        params,
        lbvh,
    };
    input.validate()?;
    Ok(input)
}

pub fn tire_dem_sparse_lbvh_gpu_rows(lbvh: &TireDemSparseLbvh) -> Vec<[f32; 4]> {
    let mut rows = Vec::with_capacity(lbvh.nodes.len() * 3);
    for node in &lbvh.nodes {
        rows.push([node.min[0], node.min[1], node.min[2], node.left as f32]);
        rows.push([node.max[0], node.max[1], node.max[2], node.right as f32]);
        rows.push([
            node.start as f32,
            node.count as f32,
            node.is_leaf() as u32 as f32,
            0.0,
        ]);
    }
    rows
}

pub fn tire_dem_sparse_lbvh_dispatch_plan(
    input: &TireDemLbvhStepGpuInput,
) -> Result<TireDemLbvhDispatchPlan> {
    let grain_count = input.validate()?;
    tire_dem_sparse_lbvh_dispatch_plan_from_parts(grain_count, input.contacts.len(), &input.lbvh)
}

fn tire_dem_sparse_lbvh_dispatch_plan_from_parts(
    grain_count: usize,
    contact_count: usize,
    lbvh: &TireDemSparseLbvh,
) -> Result<TireDemLbvhDispatchPlan> {
    if grain_count == 0 {
        return Err(Error::Solver(
            "tire_dem_lbvh_plan: at least one grain required".into(),
        ));
    }
    let node_count = lbvh.nodes.len();
    if node_count == 0 {
        return Err(Error::Solver(
            "tire_dem_lbvh_plan: at least one LBVH node required".into(),
        ));
    }
    if lbvh
        .sorted_indices
        .iter()
        .any(|&idx| idx as usize >= grain_count)
    {
        return Err(Error::Solver(
            "tire_dem_lbvh_plan: sorted index out of grain range".into(),
        ));
    }
    let node_row_count = node_count
        .checked_mul(3)
        .ok_or_else(|| Error::Solver("tire_dem_lbvh_plan: node row count overflow".into()))?;
    let workgroup_count = (grain_count as u32).div_ceil(WORKGROUP_SIZE);
    let vec4 = core::mem::size_of::<[f32; 4]>() as u64;
    let u32_bytes = core::mem::size_of::<u32>() as u64;
    let contact_bytes = core::mem::size_of::<TireDemContactCell>() as u64;
    let storage_bytes = (grain_count as u64)
        .saturating_mul(vec4)
        .saturating_mul(4)
        .saturating_add((node_row_count as u64).saturating_mul(vec4))
        .saturating_add((lbvh.sorted_indices.len() as u64).saturating_mul(u32_bytes))
        .saturating_add((contact_count.max(1) as u64).saturating_mul(contact_bytes));
    Ok(TireDemLbvhDispatchPlan {
        grain_count,
        contact_count,
        node_count,
        node_row_count,
        workgroup_count,
        storage_bytes,
    })
}

fn validate_dem_fem_terrain_grid(
    origin: [f32; 2],
    spacing: f32,
    nx: u32,
    nz: u32,
    floor_y: f32,
    friction: f32,
    grain_radius: f32,
) -> Result<()> {
    if nx == 0 || nz == 0 {
        return Err(Error::Solver(
            "tire_dem_fem_terrain: nx/nz must be non-zero".into(),
        ));
    }
    if origin.iter().any(|v| !v.is_finite()) {
        return Err(Error::Solver(
            "tire_dem_fem_terrain: origin must be finite".into(),
        ));
    }
    if !spacing.is_finite() || spacing <= 0.0 {
        return Err(Error::Solver(
            "tire_dem_fem_terrain: spacing must be positive".into(),
        ));
    }
    if !floor_y.is_finite() {
        return Err(Error::Solver(
            "tire_dem_fem_terrain: floor_y must be finite".into(),
        ));
    }
    if !friction.is_finite() || friction < 0.0 {
        return Err(Error::Solver(
            "tire_dem_fem_terrain: friction must be finite and non-negative".into(),
        ));
    }
    if !grain_radius.is_finite() || grain_radius <= 0.0 {
        return Err(Error::Solver(
            "tire_dem_fem_terrain: grain_radius must be positive".into(),
        ));
    }
    Ok(())
}

fn validate_dem_resident_terrain_params(
    grain_count: usize,
    reference_heights: &[f32],
    params: TireDemResidentTerrainParams,
) -> Result<usize> {
    if grain_count == 0 {
        return Err(Error::Solver(
            "tire_dem_resident_terrain: at least one grain required".into(),
        ));
    }
    validate_dem_fem_terrain_grid(
        params.origin,
        params.spacing,
        params.nx,
        params.nz,
        params.floor_y,
        0.0,
        params.grain_radius,
    )?;
    let row_count = (params.nx as usize)
        .checked_mul(params.nz as usize)
        .ok_or_else(|| Error::Solver("tire_dem_resident_terrain: row count overflow".into()))?;
    if reference_heights.len() != row_count {
        return Err(Error::Solver(format!(
            "tire_dem_resident_terrain: reference height row count mismatch: got {}, expected {row_count}",
            reference_heights.len()
        )));
    }
    if reference_heights.iter().any(|v| !v.is_finite()) {
        return Err(Error::Solver(
            "tire_dem_resident_terrain: reference heights must be finite".into(),
        ));
    }
    Ok(row_count)
}

pub fn tire_dem_raster_surface_height_map(
    positions: &[[f32; 4]],
    origin: [f32; 2],
    spacing: f32,
    nx: u32,
    nz: u32,
    floor_y: f32,
    grain_radius: f32,
) -> Result<Vec<f32>> {
    validate_dem_fem_terrain_grid(origin, spacing, nx, nz, floor_y, 0.0, grain_radius)?;
    let n = nx as usize * nz as usize;
    let mut heights = vec![floor_y; n];
    for &p in positions {
        if !p[0].is_finite() || !p[1].is_finite() || !p[2].is_finite() {
            return Err(Error::Solver(
                "tire_dem_surface_height_map: positions must be finite".into(),
            ));
        }
        let ix = ((p[0] - origin[0]) / spacing).floor() as i32;
        let iz = ((p[2] - origin[1]) / spacing).floor() as i32;
        if ix >= 0 && iz >= 0 && ix < nx as i32 && iz < nz as i32 {
            let idx = iz as usize * nx as usize + ix as usize;
            heights[idx] = heights[idx].max(p[1] + grain_radius);
        }
    }
    Ok(heights)
}

pub fn tire_dem_raster_surface_velocity_map(
    positions: &[[f32; 4]],
    velocities: &[[f32; 4]],
    origin: [f32; 2],
    spacing: f32,
    nx: u32,
    nz: u32,
    grain_radius: f32,
) -> Result<TireDemSurfaceVelocityMap> {
    if positions.len() != velocities.len() {
        return Err(Error::Solver(
            "tire_dem_surface_velocity_map: positions/velocities length mismatch".into(),
        ));
    }
    if nx == 0 || nz == 0 {
        return Err(Error::Solver(
            "tire_dem_surface_velocity_map: nx/nz must be non-zero".into(),
        ));
    }
    if !spacing.is_finite() || spacing <= 0.0 {
        return Err(Error::Solver(
            "tire_dem_surface_velocity_map: spacing must be positive".into(),
        ));
    }
    let n = nx as usize * nz as usize;
    let cell_index = |p: [f32; 4]| -> Option<usize> {
        if !p[0].is_finite() || !p[1].is_finite() || !p[2].is_finite() {
            return None;
        }
        let ix = ((p[0] - origin[0]) / spacing).floor() as i32;
        let iz = ((p[2] - origin[1]) / spacing).floor() as i32;
        // `then_some` would evaluate its argument eagerly even when the
        // predicate is false, so a negative `iz` would underflow through the
        // `as usize` cast and overflow the subsequent multiply in debug
        // builds before the bounds check ever runs. Use the lazy `then`
        // form so the index math only happens for in-bounds samples.
        (ix >= 0 && iz >= 0 && ix < nx as i32 && iz < nz as i32)
            .then(|| iz as usize * nx as usize + ix as usize)
    };

    let mut top_y = vec![f32::NEG_INFINITY; n];
    for &p in positions {
        if let Some(idx) = cell_index(p) {
            top_y[idx] = top_y[idx].max(p[1]);
        }
    }

    let surface_band_m = (grain_radius.max(0.0) * 2.25).max(spacing * 0.25);
    let mut sums = vec![[0.0_f32; 4]; n];
    for (&p, &v) in positions.iter().zip(velocities.iter()) {
        if let Some(idx) = cell_index(p)
            && top_y[idx].is_finite()
            && p[1] >= top_y[idx] - surface_band_m
        {
            sums[idx][0] += if v[0].is_finite() { v[0] } else { 0.0 };
            sums[idx][1] += if v[1].is_finite() { v[1] } else { 0.0 };
            sums[idx][2] += if v[2].is_finite() { v[2] } else { 0.0 };
            sums[idx][3] += 1.0;
        }
    }

    let mut active_cells = 0_u32;
    let mut max_speed = 0.0_f32;
    let mut speed_sum = 0.0_f32;
    let rows = sums
        .into_iter()
        .map(|mut row| {
            if row[3] > 0.0 {
                let inv = 1.0 / row[3];
                row[0] *= inv;
                row[1] *= inv;
                row[2] *= inv;
                let speed = (row[0] * row[0] + row[1] * row[1] + row[2] * row[2]).sqrt();
                active_cells += 1;
                max_speed = max_speed.max(speed);
                speed_sum += speed;
            }
            row
        })
        .collect::<Vec<_>>();

    Ok(TireDemSurfaceVelocityMap {
        origin,
        spacing,
        nx,
        nz,
        rows,
        active_cells,
        max_speed_m_per_s: max_speed,
        mean_speed_m_per_s: if active_cells > 0 {
            speed_sum / active_cells as f32
        } else {
            0.0
        },
    })
}

fn build_dem_lbvh_node(
    positions: &[[f32; 4]],
    sorted_indices: &[u32],
    start: usize,
    end: usize,
    leaf_size: usize,
    nodes: &mut Vec<TireDemLbvhNode>,
) -> usize {
    let (min, max) = dem_center_bounds(positions, &sorted_indices[start..end]);
    let node_id = nodes.len();
    nodes.push(TireDemLbvhNode {
        min,
        max,
        left: u32::MAX,
        right: u32::MAX,
        start: start as u32,
        count: (end - start) as u32,
    });
    if end - start <= leaf_size {
        return node_id;
    }
    let extents = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
    if extents.iter().all(|&e| e <= 1.0e-9) {
        return node_id;
    }
    let mid = start + (end - start) / 2;
    let left = build_dem_lbvh_node(positions, sorted_indices, start, mid, leaf_size, nodes);
    let right = build_dem_lbvh_node(positions, sorted_indices, mid, end, leaf_size, nodes);
    nodes[node_id].left = left as u32;
    nodes[node_id].right = right as u32;
    nodes[node_id].start = 0;
    nodes[node_id].count = 0;
    node_id
}

fn tire_dem_lbvh_visit_candidates<F>(lbvh: &TireDemSparseLbvh, p: [f32; 3], radius: f32, mut f: F)
where
    F: FnMut(u32),
{
    if lbvh.nodes.is_empty() {
        return;
    }
    let radius2 = radius.max(0.0) * radius.max(0.0);
    let mut stack = vec![0usize];
    while let Some(node_id) = stack.pop() {
        let node = lbvh.nodes[node_id];
        if aabb_distance2_to_point(node.min, node.max, p) > radius2 {
            continue;
        }
        if node.is_leaf() {
            let end = node.start + node.count;
            for &idx in &lbvh.sorted_indices[node.start as usize..end as usize] {
                f(idx);
            }
        } else {
            if node.right != u32::MAX {
                stack.push(node.right as usize);
            }
            if node.left != u32::MAX {
                stack.push(node.left as usize);
            }
        }
    }
}

/// Build a host-side cell list for [`TireDemGridStepGpuInput`].
///
/// This keeps the current GPU broadphase deterministic and easy to validate.
/// A later scaling pass can replace this host build with a GPU radix-sort /
/// prefix-sum cell-list builder without changing the DEM step semantics.
pub fn tire_dem_build_grid_step_input(
    positions: Vec<[f32; 4]>,
    velocities: Vec<[f32; 4]>,
    contacts: Vec<TireDemContactCell>,
    params: TireDemStepParams,
    grid_origin: [f32; 3],
    cell_size: f32,
    grid_shape: [u32; 3],
) -> Result<TireDemGridStepGpuInput> {
    if positions.len() != velocities.len() {
        return Err(Error::Solver(
            "tire_dem_grid_build: positions/velocities length mismatch".into(),
        ));
    }
    if positions.is_empty() {
        return Err(Error::Solver(
            "tire_dem_grid_build: at least one grain required".into(),
        ));
    }
    if !cell_size.is_finite() || cell_size <= 0.0 {
        return Err(Error::Solver(
            "tire_dem_grid_build: cell_size must be positive".into(),
        ));
    }
    if grid_shape.contains(&0) {
        return Err(Error::Solver(
            "tire_dem_grid_build: grid_shape axes must be nonzero".into(),
        ));
    }
    let cell_count = grid_cell_count(grid_shape)?;
    let mut buckets: Vec<Vec<u32>> = vec![Vec::new(); cell_count];
    for (idx, &position) in positions.iter().enumerate() {
        let coords = grid_cell_coords(position, grid_origin, cell_size, grid_shape);
        buckets[grid_cell_id(coords, grid_shape)].push(idx as u32);
    }

    let mut sorted_indices = Vec::with_capacity(positions.len());
    let mut cell_ranges = vec![[0_u32; 4]; cell_count];
    for (cell, bucket) in buckets.into_iter().enumerate() {
        let start = sorted_indices.len() as u32;
        sorted_indices.extend(bucket.iter().copied());
        let end = sorted_indices.len() as u32;
        cell_ranges[cell] = [start, end, end - start, 0];
    }

    let input = TireDemGridStepGpuInput {
        positions,
        velocities,
        contacts,
        params,
        grid_origin,
        cell_size,
        grid_shape,
        sorted_indices,
        cell_ranges,
    };
    input.validate()?;
    Ok(input)
}

/// Build an unbounded sorted cell list for [`TireDemGridStepGpuInput`].
///
/// Unlike [`TireDemGpuBuiltGridStepInput`], this path has no per-cell capacity:
/// every grain contributes one `(cell_id, grain_id)` pair, pairs are sorted by
/// cell key, and ranges are produced by scanning equal-key runs. This is the
/// reference/host staging shape for replacing the fixed-capacity atomic GPU
/// builder with a radix-sort/prefix-sum builder.
pub fn tire_dem_build_unbounded_sorted_grid_step_input(
    positions: Vec<[f32; 4]>,
    velocities: Vec<[f32; 4]>,
    contacts: Vec<TireDemContactCell>,
    params: TireDemStepParams,
    grid_origin: [f32; 3],
    cell_size: f32,
    grid_shape: [u32; 3],
) -> Result<(TireDemGridStepGpuInput, TireDemUnboundedSortedGrid)> {
    if positions.len() != velocities.len() {
        return Err(Error::Solver(
            "tire_dem_unbounded_grid_build: positions/velocities length mismatch".into(),
        ));
    }
    if positions.is_empty() {
        return Err(Error::Solver(
            "tire_dem_unbounded_grid_build: at least one grain required".into(),
        ));
    }
    if !cell_size.is_finite() || cell_size <= 0.0 {
        return Err(Error::Solver(
            "tire_dem_unbounded_grid_build: cell_size must be positive".into(),
        ));
    }
    if grid_shape.contains(&0) {
        return Err(Error::Solver(
            "tire_dem_unbounded_grid_build: grid_shape axes must be nonzero".into(),
        ));
    }
    let cell_count = grid_cell_count(grid_shape)?;
    let mut pairs = positions
        .iter()
        .enumerate()
        .map(|(idx, &position)| {
            let coords = grid_cell_coords(position, grid_origin, cell_size, grid_shape);
            (grid_cell_id(coords, grid_shape) as u32, idx as u32)
        })
        .collect::<Vec<_>>();
    pairs.sort_by_key(|&(cell, grain)| (cell, grain));

    let (sorted_cell_keys, sorted_indices): (Vec<_>, Vec<_>) = pairs.into_iter().unzip();
    let build = tire_dem_scan_unbounded_sorted_grid(
        sorted_cell_keys,
        sorted_indices,
        cell_count,
        "tire_dem_unbounded_grid_build",
    )?;
    let input = TireDemGridStepGpuInput {
        positions,
        velocities,
        contacts,
        params,
        grid_origin,
        cell_size,
        grid_shape,
        sorted_indices: build.sorted_indices.clone(),
        cell_ranges: build.cell_ranges.clone(),
    };
    input.validate()?;
    Ok((input, build))
}

fn tire_dem_scan_unbounded_sorted_grid(
    sorted_cell_keys: Vec<u32>,
    sorted_indices: Vec<u32>,
    cell_count: usize,
    label: &str,
) -> Result<TireDemUnboundedSortedGrid> {
    if sorted_cell_keys.len() != sorted_indices.len() {
        return Err(Error::Solver(format!(
            "{label}: sorted key/index length mismatch"
        )));
    }
    let mut cell_ranges = vec![[0_u32; 4]; cell_count];
    let mut cursor = 0usize;
    let mut max_cell_occupancy = 0u32;
    while cursor < sorted_cell_keys.len() {
        let cell = sorted_cell_keys[cursor] as usize;
        if cell >= cell_count {
            return Err(Error::Solver(format!(
                "{label}: sorted cell id {cell} out of range {cell_count}"
            )));
        }
        let start = cursor as u32;
        while cursor < sorted_cell_keys.len() && sorted_cell_keys[cursor] as usize == cell {
            cursor += 1;
        }
        let end = cursor as u32;
        let count = end - start;
        cell_ranges[cell] = [start, end, count, 0];
        max_cell_occupancy = max_cell_occupancy.max(count);
        if cursor < sorted_cell_keys.len() && sorted_cell_keys[cursor] as usize <= cell {
            return Err(Error::Solver(format!(
                "{label}: cell keys must be sorted ascending"
            )));
        }
    }

    Ok(TireDemUnboundedSortedGrid {
        sorted_cell_keys,
        sorted_indices,
        cell_ranges,
        max_cell_occupancy,
    })
}

/// Build an unbounded cell list using GPU key generation, GPU radix sort, and
/// GPU range scanning.
///
/// This is the first unbounded GPU construction path for sparse DEM grids:
/// every grain gets a cell key on GPU, the reusable GPU radix sorter orders the
/// `(cell_id, grain_id)` pairs, and a GPU pass scans those sorted keys into
/// per-cell ranges. The host still reads the finished list back to feed the
/// existing blocking grid-step API; the next scaling step is to keep these
/// buffers resident through the DEM neighbor pass.
pub fn tire_dem_build_unbounded_sorted_grid_step_input_gpu_radix(
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    positions: Vec<[f32; 4]>,
    velocities: Vec<[f32; 4]>,
    contacts: Vec<TireDemContactCell>,
    params: TireDemStepParams,
    grid_origin: [f32; 3],
    cell_size: f32,
    grid_shape: [u32; 3],
) -> Result<(TireDemGridStepGpuInput, TireDemUnboundedSortedGrid)> {
    if positions.len() != velocities.len() {
        return Err(Error::Solver(
            "tire_dem_gpu_radix_grid_build: positions/velocities length mismatch".into(),
        ));
    }
    if positions.is_empty() {
        return Err(Error::Solver(
            "tire_dem_gpu_radix_grid_build: at least one grain required".into(),
        ));
    }
    if !cell_size.is_finite() || cell_size <= 0.0 {
        return Err(Error::Solver(
            "tire_dem_gpu_radix_grid_build: cell_size must be positive".into(),
        ));
    }
    if grid_shape.contains(&0) {
        return Err(Error::Solver(
            "tire_dem_gpu_radix_grid_build: grid_shape axes must be nonzero".into(),
        ));
    }

    let cell_count = grid_cell_count(grid_shape)?;
    let n = positions.len();

    let u32_bytes = |len: usize| (len.max(1) * core::mem::size_of::<u32>()) as u64;
    let uvec4_bytes = |len: usize| (len.max(1) * core::mem::size_of::<[u32; 4]>()) as u64;
    let vec4_bytes = |len: usize| (len.max(1) * core::mem::size_of::<[f32; 4]>()) as u64;
    let storage =
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
    let pos_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tire_dem gpu radix pos in"),
        size: vec4_bytes(n),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let keys_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tire_dem gpu radix cell keys"),
        size: u32_bytes(n),
        usage: storage,
        mapped_at_creation: false,
    });
    let values_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tire_dem gpu radix grain ids"),
        size: u32_bytes(n),
        usage: storage,
        mapped_at_creation: false,
    });
    let ranges_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tire_dem gpu radix cell ranges"),
        size: uvec4_bytes(cell_count),
        usage: storage,
        mapped_at_creation: false,
    });
    let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tire_dem gpu radix uniforms"),
        size: core::mem::size_of::<TireDemStepUniforms>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let stage_keys = stage_u32(&device, "tire_dem gpu radix stage keys", n);
    let stage_values = stage_u32(&device, "tire_dem gpu radix stage grain ids", n);
    let stage_ranges = stage_uvec4(&device, "tire_dem gpu radix stage ranges", cell_count);

    queue.write_buffer(&pos_buf, 0, bytemuck::cast_slice(&positions));
    queue.write_buffer(
        &uniforms,
        0,
        bytemuck::bytes_of(&TireDemStepUniforms {
            dims: [n as u32, contacts.len() as u32, cell_count as u32, 0],
            params0: [
                params.grain_mass,
                params.grain_radius,
                params.dt,
                params.tire_position_push_scale,
            ],
            params1: [
                params.normal_stiffness,
                params.damping,
                params.friction,
                params.floor_y,
            ],
            gravity: [params.gravity[0], params.gravity[1], params.gravity[2], 0.0],
            grid_origin_cell_size: [grid_origin[0], grid_origin[1], grid_origin[2], cell_size],
            grid_shape: [grid_shape[0], grid_shape[1], grid_shape[2], 0],
        }),
    );

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("molla::tire_dem gpu radix grid build"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/tire_dem.wgsl").into()),
    });
    let layout_data = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("molla::tire_dem gpu radix grid data"),
        entries: &[
            storage_entry(0, true),
            storage_entry(5, false),
            storage_entry(6, false),
            storage_entry(9, false),
        ],
    });
    let layout_uniforms = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("molla::tire_dem gpu radix grid uniforms"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("molla::tire_dem gpu radix grid layout"),
        bind_group_layouts: &[Some(&layout_data), Some(&layout_uniforms)],
        immediate_size: 0,
    });
    let keygen_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("molla::tire_dem::build_unbounded_cell_keys"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("build_unbounded_cell_keys"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    let range_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("molla::tire_dem::scan_unbounded_cell_ranges"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("scan_unbounded_cell_ranges"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    let bg_data = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("molla::tire_dem gpu radix grid data bg"),
        layout: &layout_data,
        entries: &[
            entry(0, &pos_buf),
            entry(5, &values_buf),
            entry(6, &ranges_buf),
            entry(9, &keys_buf),
        ],
    });
    let bg_uniforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("molla::tire_dem gpu radix grid uniforms bg"),
        layout: &layout_uniforms,
        entries: &[entry(0, &uniforms)],
    });

    let mut key_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("molla::tire_dem gpu radix keygen encoder"),
    });
    {
        let mut pass = key_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("molla::tire_dem gpu radix keygen pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&keygen_pipeline);
        pass.set_bind_group(0, &bg_data, &[]);
        pass.set_bind_group(1, &bg_uniforms, &[]);
        pass.dispatch_workgroups((n as u32).div_ceil(WORKGROUP_SIZE), 1, 1);
    }
    queue.submit(std::iter::once(key_encoder.finish()));

    molla_compute_primitives::radix_sort_pairs(
        Arc::clone(&device),
        Arc::clone(&queue),
        &keys_buf,
        &values_buf,
        n,
    )?;

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("molla::tire_dem gpu radix range/readback encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("molla::tire_dem gpu radix range pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&range_pipeline);
        pass.set_bind_group(0, &bg_data, &[]);
        pass.set_bind_group(1, &bg_uniforms, &[]);
        pass.dispatch_workgroups((cell_count as u32).div_ceil(WORKGROUP_SIZE), 1, 1);
    }
    let bytes = u32_bytes(n);
    encoder.copy_buffer_to_buffer(&keys_buf, 0, &stage_keys, 0, bytes);
    encoder.copy_buffer_to_buffer(&values_buf, 0, &stage_values, 0, bytes);
    encoder.copy_buffer_to_buffer(&ranges_buf, 0, &stage_ranges, 0, uvec4_bytes(cell_count));
    queue.submit(std::iter::once(encoder.finish()));

    let sorted_cell_keys = read_u32(&device, &stage_keys, "tire_dem gpu radix cell keys")?;
    let sorted_indices = read_u32(&device, &stage_values, "tire_dem gpu radix grain ids")?;
    let gpu_cell_ranges = read_uvec4(&device, &stage_ranges, "tire_dem gpu radix cell ranges")?;
    let build = tire_dem_scan_unbounded_sorted_grid(
        sorted_cell_keys,
        sorted_indices,
        cell_count,
        "tire_dem_gpu_radix_grid_build",
    )?;
    if build.cell_ranges != gpu_cell_ranges {
        return Err(Error::Solver(format!(
            "tire_dem_gpu_radix_grid_build: GPU range scan mismatch: gpu={gpu_cell_ranges:?} cpu={:?}",
            build.cell_ranges
        )));
    }
    let input = TireDemGridStepGpuInput {
        positions,
        velocities,
        contacts,
        params,
        grid_origin,
        cell_size,
        grid_shape,
        sorted_indices: build.sorted_indices.clone(),
        cell_ranges: build.cell_ranges.clone(),
    };
    input.validate()?;
    Ok((input, build))
}

fn tire_contact_delta(
    position: [f32; 4],
    contacts: &[TireDemContactCell],
    mass: f32,
    radius: f32,
    dt: f32,
    push_scale: f32,
) -> ([f32; 3], f32) {
    let mut dv = [0.0_f32; 3];
    let mut push_y = 0.0_f32;
    for contact in contacts {
        let [cx, surface_y, cz, influence] = contact.center_radius;
        let influence = influence.max(radius * 1.25);
        let dx = position[0] - cx;
        let dz = position[2] - cz;
        let horizontal = (dx * dx + dz * dz).sqrt();
        if horizontal > influence + radius {
            continue;
        }
        let below_surface = (surface_y + radius * 1.5) - position[1];
        if below_surface < -radius || below_surface > radius * 5.0 {
            continue;
        }
        let radial_w = (1.0 - horizontal / (influence + radius)).clamp(0.0, 1.0);
        let depth_w = (1.0 - below_surface.max(0.0) / (radius * 5.0)).clamp(0.15, 1.0);
        let w = radial_w * radial_w * depth_w;
        dv[0] += contact.force[0] * w * dt / mass;
        dv[1] += contact.force[1] * w * dt / mass;
        dv[2] += contact.force[2] * w * dt / mass;
        if contact.force[1] < 0.0 {
            push_y += (-contact.force[1]) * w * dt * dt / mass * push_scale;
        }
    }
    (dv, push_y.min(radius * 0.35))
}

/// CPU reference for [`TireDemGpuKernel`].
pub fn tire_dem_reference_impulses(input: &TireDemGpuInput) -> Result<TireDemGpuOutput> {
    let n = input.validate()?;
    let mut positions = input.positions.clone();
    let mut velocities = input.velocities.clone();
    let mass = input.grain_mass.max(1.0e-9);
    let radius = input.grain_radius.max(1.0e-9);
    let dt = input.dt.max(0.0);
    let push_scale = input.position_push_scale.max(0.0);

    for i in 0..n {
        let (dv, push_y) =
            tire_contact_delta(positions[i], &input.contacts, mass, radius, dt, push_scale);
        velocities[i][0] += dv[0];
        velocities[i][1] += dv[1];
        velocities[i][2] += dv[2];
        positions[i][0] += velocities[i][0] * dt;
        positions[i][1] += velocities[i][1] * dt;
        positions[i][2] += velocities[i][2] * dt;
        positions[i][1] -= push_y;
    }
    Ok(TireDemGpuOutput {
        positions,
        velocities,
    })
}

fn dem_pair_force(
    pa: [f32; 3],
    va: [f32; 3],
    pb: [f32; 3],
    vb: [f32; 3],
    sum_radii: f32,
    normal_stiffness: f32,
    damping: f32,
    friction: f32,
) -> [f32; 3] {
    let d = [pa[0] - pb[0], pa[1] - pb[1], pa[2] - pb[2]];
    let dist = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    let overlap = sum_radii - dist;
    if overlap <= 0.0 || dist < 1.0e-9 {
        return [0.0; 3];
    }
    let n = [d[0] / dist, d[1] / dist, d[2] / dist];
    let v_rel = [va[0] - vb[0], va[1] - vb[1], va[2] - vb[2]];
    let v_n = v_rel[0] * n[0] + v_rel[1] * n[1] + v_rel[2] * n[2];
    let fn_mag = (normal_stiffness * overlap - damping * v_n).max(0.0);
    let v_t = [
        v_rel[0] - n[0] * v_n,
        v_rel[1] - n[1] * v_n,
        v_rel[2] - n[2] * v_n,
    ];
    let vt = (v_t[0] * v_t[0] + v_t[1] * v_t[1] + v_t[2] * v_t[2]).sqrt();
    let mut f = [n[0] * fn_mag, n[1] * fn_mag, n[2] * fn_mag];
    if vt > 1.0e-9 {
        let ft = friction.max(0.0) * fn_mag * (vt / (vt + 0.05));
        f[0] -= v_t[0] * ft / vt;
        f[1] -= v_t[1] * ft / vt;
        f[2] -= v_t[2] * ft / vt;
    }
    f
}

/// CPU reference for the correctness-first GPU DEM neighbor/floor/tire step.
pub fn tire_dem_reference_step(input: &TireDemStepGpuInput) -> Result<TireDemGpuOutput> {
    let n = input.validate()?;
    let p = input.params;
    let mut positions = input.positions.clone();
    let mut velocities = input.velocities.clone();
    let mass = p.grain_mass.max(1.0e-9);
    let radius = p.grain_radius.max(1.0e-9);
    let dt = p.dt.max(0.0);
    let mut forces = vec![
        [
            p.gravity[0] * mass,
            p.gravity[1] * mass,
            p.gravity[2] * mass
        ];
        n
    ];
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let f = dem_pair_force(
                [positions[i][0], positions[i][1], positions[i][2]],
                [velocities[i][0], velocities[i][1], velocities[i][2]],
                [positions[j][0], positions[j][1], positions[j][2]],
                [velocities[j][0], velocities[j][1], velocities[j][2]],
                2.0 * radius,
                p.normal_stiffness.max(0.0),
                p.damping.max(0.0),
                p.friction.max(0.0),
            );
            forces[i][0] += f[0];
            forces[i][1] += f[1];
            forces[i][2] += f[2];
        }
        let pen = (p.floor_y + radius) - positions[i][1];
        if pen > 0.0 {
            let fn_mag = (p.normal_stiffness.max(0.0) * pen
                - p.damping.max(0.0) * velocities[i][1])
                .max(0.0);
            forces[i][1] += fn_mag;
            let vt =
                (velocities[i][0] * velocities[i][0] + velocities[i][2] * velocities[i][2]).sqrt();
            if vt > 1.0e-9 {
                let ft = p.friction.max(0.0) * fn_mag * (vt / (vt + 0.05));
                forces[i][0] -= velocities[i][0] * ft / vt;
                forces[i][2] -= velocities[i][2] * ft / vt;
            }
        }
    }

    for i in 0..n {
        let (tire_dv, tire_push_y) = tire_contact_delta(
            positions[i],
            &input.contacts,
            mass,
            radius,
            dt,
            p.tire_position_push_scale.max(0.0),
        );
        velocities[i][0] += forces[i][0] * dt / mass;
        velocities[i][1] += forces[i][1] * dt / mass;
        velocities[i][2] += forces[i][2] * dt / mass;
        velocities[i][0] += tire_dv[0];
        velocities[i][1] += tire_dv[1];
        velocities[i][2] += tire_dv[2];
        positions[i][0] += velocities[i][0] * dt;
        positions[i][1] += velocities[i][1] * dt;
        positions[i][2] += velocities[i][2] * dt;
        positions[i][1] -= tire_push_y;
    }
    Ok(TireDemGpuOutput {
        positions,
        velocities,
    })
}

/// CPU reference for the cell-grid broadphase DEM step.
pub fn tire_dem_reference_grid_step(input: &TireDemGridStepGpuInput) -> Result<TireDemGpuOutput> {
    let n = input.validate()?;
    let p = input.params;
    let mut positions = input.positions.clone();
    let mut velocities = input.velocities.clone();
    let mass = p.grain_mass.max(1.0e-9);
    let radius = p.grain_radius.max(1.0e-9);
    let dt = p.dt.max(0.0);
    let mut forces = vec![
        [
            p.gravity[0] * mass,
            p.gravity[1] * mass,
            p.gravity[2] * mass
        ];
        n
    ];
    for i in 0..n {
        let coords = grid_cell_coords(
            positions[i],
            input.grid_origin,
            input.cell_size,
            input.grid_shape,
        );
        let min_x = coords[0].saturating_sub(1);
        let min_y = coords[1].saturating_sub(1);
        let min_z = coords[2].saturating_sub(1);
        let max_x = (coords[0] + 1).min(input.grid_shape[0] - 1);
        let max_y = (coords[1] + 1).min(input.grid_shape[1] - 1);
        let max_z = (coords[2] + 1).min(input.grid_shape[2] - 1);
        for z in min_z..=max_z {
            for y in min_y..=max_y {
                for x in min_x..=max_x {
                    let cell = grid_cell_id([x, y, z], input.grid_shape);
                    let range = input.cell_ranges[cell];
                    for cursor in range[0]..range[1] {
                        let j = input.sorted_indices[cursor as usize] as usize;
                        if i == j {
                            continue;
                        }
                        let f = dem_pair_force(
                            [positions[i][0], positions[i][1], positions[i][2]],
                            [velocities[i][0], velocities[i][1], velocities[i][2]],
                            [positions[j][0], positions[j][1], positions[j][2]],
                            [velocities[j][0], velocities[j][1], velocities[j][2]],
                            2.0 * radius,
                            p.normal_stiffness.max(0.0),
                            p.damping.max(0.0),
                            p.friction.max(0.0),
                        );
                        forces[i][0] += f[0];
                        forces[i][1] += f[1];
                        forces[i][2] += f[2];
                    }
                }
            }
        }
        let pen = (p.floor_y + radius) - positions[i][1];
        if pen > 0.0 {
            let fn_mag = (p.normal_stiffness.max(0.0) * pen
                - p.damping.max(0.0) * velocities[i][1])
                .max(0.0);
            forces[i][1] += fn_mag;
            let vt =
                (velocities[i][0] * velocities[i][0] + velocities[i][2] * velocities[i][2]).sqrt();
            if vt > 1.0e-9 {
                let ft = p.friction.max(0.0) * fn_mag * (vt / (vt + 0.05));
                forces[i][0] -= velocities[i][0] * ft / vt;
                forces[i][2] -= velocities[i][2] * ft / vt;
            }
        }
    }

    for i in 0..n {
        let (tire_dv, tire_push_y) = tire_contact_delta(
            positions[i],
            &input.contacts,
            mass,
            radius,
            dt,
            p.tire_position_push_scale.max(0.0),
        );
        velocities[i][0] += forces[i][0] * dt / mass;
        velocities[i][1] += forces[i][1] * dt / mass;
        velocities[i][2] += forces[i][2] * dt / mass;
        velocities[i][0] += tire_dv[0];
        velocities[i][1] += tire_dv[1];
        velocities[i][2] += tire_dv[2];
        positions[i][0] += velocities[i][0] * dt;
        positions[i][1] += velocities[i][1] * dt;
        positions[i][2] += velocities[i][2] * dt;
        positions[i][1] -= tire_push_y;
    }
    Ok(TireDemGpuOutput {
        positions,
        velocities,
    })
}

/// CPU reference DEM step using the sparse LBVH broadphase.
///
/// This keeps the same force law as [`tire_dem_reference_step`] but traverses
/// only BVH candidates whose grain centres can be within `2 * grain_radius`.
/// It is the correctness oracle for sparse-bed LBVH work before the same node
/// layout is moved fully into GPU DEM stepping.
pub fn tire_dem_reference_sparse_lbvh_step(
    input: &TireDemStepGpuInput,
) -> Result<(TireDemGpuOutput, TireDemSparseLbvh)> {
    let n = input.validate()?;
    let p = input.params;
    let lbvh = tire_dem_build_sparse_lbvh(&input.positions, 8)?;
    let mut positions = input.positions.clone();
    let mut velocities = input.velocities.clone();
    let mass = p.grain_mass.max(1.0e-9);
    let radius = p.grain_radius.max(1.0e-9);
    let dt = p.dt.max(0.0);
    let mut forces = vec![
        [
            p.gravity[0] * mass,
            p.gravity[1] * mass,
            p.gravity[2] * mass
        ];
        n
    ];
    for i in 0..n {
        tire_dem_lbvh_visit_candidates(
            &lbvh,
            [positions[i][0], positions[i][1], positions[i][2]],
            2.0 * radius + 1.0e-6,
            |j| {
                let j = j as usize;
                if i == j {
                    return;
                }
                let f = dem_pair_force(
                    [positions[i][0], positions[i][1], positions[i][2]],
                    [velocities[i][0], velocities[i][1], velocities[i][2]],
                    [positions[j][0], positions[j][1], positions[j][2]],
                    [velocities[j][0], velocities[j][1], velocities[j][2]],
                    2.0 * radius,
                    p.normal_stiffness.max(0.0),
                    p.damping.max(0.0),
                    p.friction.max(0.0),
                );
                forces[i][0] += f[0];
                forces[i][1] += f[1];
                forces[i][2] += f[2];
            },
        );
        let pen = (p.floor_y + radius) - positions[i][1];
        if pen > 0.0 {
            let fn_mag = (p.normal_stiffness.max(0.0) * pen
                - p.damping.max(0.0) * velocities[i][1])
                .max(0.0);
            forces[i][1] += fn_mag;
            let vt =
                (velocities[i][0] * velocities[i][0] + velocities[i][2] * velocities[i][2]).sqrt();
            if vt > 1.0e-9 {
                let ft = p.friction.max(0.0) * fn_mag * (vt / (vt + 0.05));
                forces[i][0] -= velocities[i][0] * ft / vt;
                forces[i][2] -= velocities[i][2] * ft / vt;
            }
        }
    }

    for i in 0..n {
        let (tire_dv, tire_push_y) = tire_contact_delta(
            positions[i],
            &input.contacts,
            mass,
            radius,
            dt,
            p.tire_position_push_scale.max(0.0),
        );
        velocities[i][0] += forces[i][0] * dt / mass;
        velocities[i][1] += forces[i][1] * dt / mass;
        velocities[i][2] += forces[i][2] * dt / mass;
        velocities[i][0] += tire_dv[0];
        velocities[i][1] += tire_dv[1];
        velocities[i][2] += tire_dv[2];
        positions[i][0] += velocities[i][0] * dt;
        positions[i][1] += velocities[i][1] * dt;
        positions[i][2] += velocities[i][2] * dt;
        positions[i][1] -= tire_push_y;
    }
    Ok((
        TireDemGpuOutput {
            positions,
            velocities,
        },
        lbvh,
    ))
}

impl TireDemGpuKernel {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::tire_dem"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/tire_dem.wgsl").into()),
        });
        let layout_io = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::tire_dem io"),
            entries: &[
                storage_entry(0, true),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, false),
                storage_entry(4, false),
            ],
        });
        let layout_grid_io = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::tire_dem grid io"),
            entries: &[
                storage_entry(0, true),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, false),
                storage_entry(4, false),
                storage_entry(5, false),
                storage_entry(6, false),
            ],
        });
        let layout_lbvh_io = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::tire_dem lbvh io"),
            entries: &[
                storage_entry(0, true),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, false),
                storage_entry(4, false),
                storage_entry(10, true),
                storage_entry(11, true),
            ],
        });
        let layout_terrain_io = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::tire_dem terrain raster io"),
            entries: &[
                storage_entry(0, true),
                storage_entry(1, true),
                storage_entry(12, true),
                storage_entry(13, false),
                storage_entry(14, false),
                storage_entry(15, false),
            ],
        });
        let layout_unbounded_grid_io =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("molla::tire_dem unbounded grid io"),
                entries: &[
                    storage_entry(0, true),
                    storage_entry(5, false),
                    storage_entry(6, false),
                    storage_entry(9, false),
                ],
            });
        let layout_sparse_lbvh_build_io =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("molla::tire_dem sparse lbvh build io"),
                entries: &[
                    storage_entry(0, true),
                    storage_entry(3, false),
                    storage_entry(5, false),
                    storage_entry(9, false),
                ],
            });
        let layout_uniforms = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::tire_dem uniforms"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("molla::tire_dem layout"),
            bind_group_layouts: &[Some(&layout_io), Some(&layout_uniforms)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::tire_dem::apply_impulses"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("apply_impulses"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let layout_step_uniforms =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("molla::tire_dem step uniforms"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        let step_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("molla::tire_dem step layout"),
            bind_group_layouts: &[Some(&layout_io), Some(&layout_step_uniforms)],
            immediate_size: 0,
        });
        let step_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::tire_dem::step_dem"),
            layout: Some(&step_pipeline_layout),
            module: &shader,
            entry_point: Some("step_dem"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let grid_step_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("molla::tire_dem grid step layout"),
                bind_group_layouts: &[Some(&layout_grid_io), Some(&layout_step_uniforms)],
                immediate_size: 0,
            });
        let grid_step_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::tire_dem::step_dem_grid"),
            layout: Some(&grid_step_pipeline_layout),
            module: &shader,
            entry_point: Some("step_dem_grid"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let lbvh_step_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("molla::tire_dem lbvh step layout"),
                bind_group_layouts: &[Some(&layout_lbvh_io), Some(&layout_step_uniforms)],
                immediate_size: 0,
            });
        let lbvh_step_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::tire_dem::step_dem_lbvh"),
            layout: Some(&lbvh_step_pipeline_layout),
            module: &shader,
            entry_point: Some("step_dem_lbvh"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let terrain_raster_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("molla::tire_dem terrain raster layout"),
                bind_group_layouts: &[Some(&layout_terrain_io), Some(&layout_step_uniforms)],
                immediate_size: 0,
            });
        let terrain_raster_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("molla::tire_dem::raster_dem_fem_terrain"),
                layout: Some(&terrain_raster_pipeline_layout),
                module: &shader,
                entry_point: Some("raster_dem_fem_terrain"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let unbounded_grid_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("molla::tire_dem unbounded grid layout"),
                bind_group_layouts: &[Some(&layout_unbounded_grid_io), Some(&layout_step_uniforms)],
                immediate_size: 0,
            });
        let unbounded_key_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("molla::tire_dem::build_unbounded_cell_keys"),
                layout: Some(&unbounded_grid_pipeline_layout),
                module: &shader,
                entry_point: Some("build_unbounded_cell_keys"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let unbounded_range_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("molla::tire_dem::scan_unbounded_cell_ranges"),
                layout: Some(&unbounded_grid_pipeline_layout),
                module: &shader,
                entry_point: Some("scan_unbounded_cell_ranges"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let sparse_lbvh_build_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("molla::tire_dem sparse lbvh build layout"),
                bind_group_layouts: &[
                    Some(&layout_sparse_lbvh_build_io),
                    Some(&layout_step_uniforms),
                ],
                immediate_size: 0,
            });
        let sparse_lbvh_morton_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("molla::tire_dem::build_sparse_lbvh_morton_keys"),
                layout: Some(&sparse_lbvh_build_pipeline_layout),
                module: &shader,
                entry_point: Some("build_sparse_lbvh_morton_keys"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let sparse_lbvh_leaf_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("molla::tire_dem::build_sparse_lbvh_leaf_rows"),
                layout: Some(&sparse_lbvh_build_pipeline_layout),
                module: &shader,
                entry_point: Some("build_sparse_lbvh_leaf_rows"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let sparse_lbvh_internal_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("molla::tire_dem::build_sparse_lbvh_internal_level"),
                layout: Some(&sparse_lbvh_build_pipeline_layout),
                module: &shader,
                entry_point: Some("build_sparse_lbvh_internal_level"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        Self {
            pipeline,
            step_pipeline,
            grid_step_pipeline,
            lbvh_step_pipeline,
            terrain_raster_pipeline,
            unbounded_key_pipeline,
            unbounded_range_pipeline,
            sparse_lbvh_morton_pipeline,
            sparse_lbvh_leaf_pipeline,
            sparse_lbvh_internal_pipeline,
            layout_io,
            layout_grid_io,
            layout_lbvh_io,
            layout_terrain_io,
            layout_unbounded_grid_io,
            layout_sparse_lbvh_build_io,
            layout_uniforms,
            layout_step_uniforms,
        }
    }

    pub fn dispatch_blocking(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input: &TireDemGpuInput,
    ) -> Result<TireDemGpuOutput> {
        let n = input.validate()?;
        let contact_len = input.contacts.len().max(1);
        let vec4_bytes = |len: usize| (len.max(1) * core::mem::size_of::<[f32; 4]>()) as u64;
        let contact_bytes =
            |len: usize| (len.max(1) * core::mem::size_of::<TireDemContactCell>()) as u64;
        let mk_buf = |label: &str, bytes: u64, usage: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes.max(4),
                usage,
                mapped_at_creation: false,
            })
        };
        let usage_ro = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let usage_rw = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        let pos_in = mk_buf("tire_dem pos in", vec4_bytes(n), usage_ro);
        let vel_in = mk_buf("tire_dem vel in", vec4_bytes(n), usage_ro);
        let contacts = mk_buf("tire_dem contacts", contact_bytes(contact_len), usage_ro);
        let pos_out = mk_buf("tire_dem pos out", vec4_bytes(n), usage_rw);
        let vel_out = mk_buf("tire_dem vel out", vec4_bytes(n), usage_rw);
        let uniforms = mk_buf(
            "tire_dem uniforms",
            core::mem::size_of::<TireDemStepUniforms>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let stage_pos = stage_vec4(device, "tire_dem stage pos", n);
        let stage_vel = stage_vec4(device, "tire_dem stage vel", n);

        queue.write_buffer(&pos_in, 0, bytemuck::cast_slice(&input.positions));
        queue.write_buffer(&vel_in, 0, bytemuck::cast_slice(&input.velocities));
        if input.contacts.is_empty() {
            queue.write_buffer(
                &contacts,
                0,
                bytemuck::cast_slice(&[TireDemContactCell::default()]),
            );
        } else {
            queue.write_buffer(&contacts, 0, bytemuck::cast_slice(&input.contacts));
        }
        queue.write_buffer(
            &uniforms,
            0,
            bytemuck::bytes_of(&TireDemStepUniforms {
                dims: [n as u32, input.contacts.len() as u32, 0, 0],
                params0: [
                    input.grain_mass,
                    input.grain_radius,
                    input.dt,
                    input.position_push_scale,
                ],
                params1: [0.0; 4],
                gravity: [0.0; 4],
                grid_origin_cell_size: [0.0; 4],
                grid_shape: [0; 4],
            }),
        );

        let bg_io = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_dem io bg"),
            layout: &self.layout_io,
            entries: &[
                entry(0, &pos_in),
                entry(1, &vel_in),
                entry(2, &contacts),
                entry(3, &pos_out),
                entry(4, &vel_out),
            ],
        });
        let bg_uniforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_dem uniforms bg"),
            layout: &self.layout_uniforms,
            entries: &[entry(0, &uniforms)],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::tire_dem encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::tire_dem pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg_io, &[]);
            pass.set_bind_group(1, &bg_uniforms, &[]);
            pass.dispatch_workgroups((n as u32).div_ceil(WORKGROUP_SIZE), 1, 1);
        }
        let bytes = vec4_bytes(n);
        encoder.copy_buffer_to_buffer(&pos_out, 0, &stage_pos, 0, bytes);
        encoder.copy_buffer_to_buffer(&vel_out, 0, &stage_vel, 0, bytes);
        queue.submit(std::iter::once(encoder.finish()));

        Ok(TireDemGpuOutput {
            positions: read_vec4(device, &stage_pos, "tire_dem positions")?,
            velocities: read_vec4(device, &stage_vel, "tire_dem velocities")?,
        })
    }

    pub fn dispatch_step_blocking(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input: &TireDemStepGpuInput,
    ) -> Result<TireDemGpuOutput> {
        let n = input.validate()?;
        let contact_len = input.contacts.len().max(1);
        let vec4_bytes = |len: usize| (len.max(1) * core::mem::size_of::<[f32; 4]>()) as u64;
        let contact_bytes =
            |len: usize| (len.max(1) * core::mem::size_of::<TireDemContactCell>()) as u64;
        let mk_buf = |label: &str, bytes: u64, usage: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes.max(4),
                usage,
                mapped_at_creation: false,
            })
        };
        let usage_ro = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let usage_rw = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        let pos_in = mk_buf("tire_dem step pos in", vec4_bytes(n), usage_ro);
        let vel_in = mk_buf("tire_dem step vel in", vec4_bytes(n), usage_ro);
        let contacts = mk_buf(
            "tire_dem step contacts",
            contact_bytes(contact_len),
            usage_ro,
        );
        let pos_out = mk_buf("tire_dem step pos out", vec4_bytes(n), usage_rw);
        let vel_out = mk_buf("tire_dem step vel out", vec4_bytes(n), usage_rw);
        let uniforms = mk_buf(
            "tire_dem step uniforms",
            core::mem::size_of::<TireDemStepUniforms>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let stage_pos = stage_vec4(device, "tire_dem step stage pos", n);
        let stage_vel = stage_vec4(device, "tire_dem step stage vel", n);

        queue.write_buffer(&pos_in, 0, bytemuck::cast_slice(&input.positions));
        queue.write_buffer(&vel_in, 0, bytemuck::cast_slice(&input.velocities));
        if input.contacts.is_empty() {
            queue.write_buffer(
                &contacts,
                0,
                bytemuck::cast_slice(&[TireDemContactCell::default()]),
            );
        } else {
            queue.write_buffer(&contacts, 0, bytemuck::cast_slice(&input.contacts));
        }
        let p = input.params;
        queue.write_buffer(
            &uniforms,
            0,
            bytemuck::bytes_of(&TireDemStepUniforms {
                dims: [n as u32, input.contacts.len() as u32, 0, 0],
                params0: [
                    p.grain_mass,
                    p.grain_radius,
                    p.dt,
                    p.tire_position_push_scale,
                ],
                params1: [p.normal_stiffness, p.damping, p.friction, p.floor_y],
                gravity: [p.gravity[0], p.gravity[1], p.gravity[2], 0.0],
                grid_origin_cell_size: [0.0; 4],
                grid_shape: [0; 4],
            }),
        );

        let bg_io = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_dem step io bg"),
            layout: &self.layout_io,
            entries: &[
                entry(0, &pos_in),
                entry(1, &vel_in),
                entry(2, &contacts),
                entry(3, &pos_out),
                entry(4, &vel_out),
            ],
        });
        let bg_uniforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_dem step uniforms bg"),
            layout: &self.layout_step_uniforms,
            entries: &[entry(0, &uniforms)],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::tire_dem step encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::tire_dem step pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.step_pipeline);
            pass.set_bind_group(0, &bg_io, &[]);
            pass.set_bind_group(1, &bg_uniforms, &[]);
            pass.dispatch_workgroups((n as u32).div_ceil(WORKGROUP_SIZE), 1, 1);
        }
        let bytes = vec4_bytes(n);
        encoder.copy_buffer_to_buffer(&pos_out, 0, &stage_pos, 0, bytes);
        encoder.copy_buffer_to_buffer(&vel_out, 0, &stage_vel, 0, bytes);
        queue.submit(std::iter::once(encoder.finish()));

        Ok(TireDemGpuOutput {
            positions: read_vec4(device, &stage_pos, "tire_dem step positions")?,
            velocities: read_vec4(device, &stage_vel, "tire_dem step velocities")?,
        })
    }

    pub fn dispatch_grid_step_blocking(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input: &TireDemGridStepGpuInput,
    ) -> Result<TireDemGpuOutput> {
        let n = input.validate()?;
        let contact_len = input.contacts.len().max(1);
        let vec4_bytes = |len: usize| (len.max(1) * core::mem::size_of::<[f32; 4]>()) as u64;
        let contact_bytes =
            |len: usize| (len.max(1) * core::mem::size_of::<TireDemContactCell>()) as u64;
        let u32_bytes = |len: usize| (len.max(1) * core::mem::size_of::<u32>()) as u64;
        let uvec4_bytes = |len: usize| (len.max(1) * core::mem::size_of::<[u32; 4]>()) as u64;
        let mk_buf = |label: &str, bytes: u64, usage: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes.max(4),
                usage,
                mapped_at_creation: false,
            })
        };
        let usage_ro = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let usage_rw = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        let pos_in = mk_buf("tire_dem grid step pos in", vec4_bytes(n), usage_ro);
        let vel_in = mk_buf("tire_dem grid step vel in", vec4_bytes(n), usage_ro);
        let contacts = mk_buf(
            "tire_dem grid step contacts",
            contact_bytes(contact_len),
            usage_ro,
        );
        let pos_out = mk_buf("tire_dem grid step pos out", vec4_bytes(n), usage_rw);
        let vel_out = mk_buf("tire_dem grid step vel out", vec4_bytes(n), usage_rw);
        let sorted_indices = mk_buf(
            "tire_dem grid step sorted indices",
            u32_bytes(input.sorted_indices.len()),
            usage_ro,
        );
        let cell_ranges = mk_buf(
            "tire_dem grid step cell ranges",
            uvec4_bytes(input.cell_ranges.len()),
            usage_ro,
        );
        let uniforms = mk_buf(
            "tire_dem grid step uniforms",
            core::mem::size_of::<TireDemStepUniforms>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let stage_pos = stage_vec4(device, "tire_dem grid step stage pos", n);
        let stage_vel = stage_vec4(device, "tire_dem grid step stage vel", n);

        queue.write_buffer(&pos_in, 0, bytemuck::cast_slice(&input.positions));
        queue.write_buffer(&vel_in, 0, bytemuck::cast_slice(&input.velocities));
        if input.contacts.is_empty() {
            queue.write_buffer(
                &contacts,
                0,
                bytemuck::cast_slice(&[TireDemContactCell::default()]),
            );
        } else {
            queue.write_buffer(&contacts, 0, bytemuck::cast_slice(&input.contacts));
        }
        queue.write_buffer(
            &sorted_indices,
            0,
            bytemuck::cast_slice(&input.sorted_indices),
        );
        queue.write_buffer(&cell_ranges, 0, bytemuck::cast_slice(&input.cell_ranges));
        let p = input.params;
        queue.write_buffer(
            &uniforms,
            0,
            bytemuck::bytes_of(&TireDemStepUniforms {
                dims: [
                    n as u32,
                    input.contacts.len() as u32,
                    input.cell_ranges.len() as u32,
                    0,
                ],
                params0: [
                    p.grain_mass,
                    p.grain_radius,
                    p.dt,
                    p.tire_position_push_scale,
                ],
                params1: [p.normal_stiffness, p.damping, p.friction, p.floor_y],
                gravity: [p.gravity[0], p.gravity[1], p.gravity[2], 0.0],
                grid_origin_cell_size: [
                    input.grid_origin[0],
                    input.grid_origin[1],
                    input.grid_origin[2],
                    input.cell_size,
                ],
                grid_shape: [
                    input.grid_shape[0],
                    input.grid_shape[1],
                    input.grid_shape[2],
                    0,
                ],
            }),
        );

        let bg_io = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_dem grid step io bg"),
            layout: &self.layout_grid_io,
            entries: &[
                entry(0, &pos_in),
                entry(1, &vel_in),
                entry(2, &contacts),
                entry(3, &pos_out),
                entry(4, &vel_out),
                entry(5, &sorted_indices),
                entry(6, &cell_ranges),
            ],
        });
        let bg_uniforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_dem grid step uniforms bg"),
            layout: &self.layout_step_uniforms,
            entries: &[entry(0, &uniforms)],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::tire_dem grid step encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::tire_dem grid step pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.grid_step_pipeline);
            pass.set_bind_group(0, &bg_io, &[]);
            pass.set_bind_group(1, &bg_uniforms, &[]);
            pass.dispatch_workgroups((n as u32).div_ceil(WORKGROUP_SIZE), 1, 1);
        }
        let bytes = vec4_bytes(n);
        encoder.copy_buffer_to_buffer(&pos_out, 0, &stage_pos, 0, bytes);
        encoder.copy_buffer_to_buffer(&vel_out, 0, &stage_vel, 0, bytes);
        queue.submit(std::iter::once(encoder.finish()));

        Ok(TireDemGpuOutput {
            positions: read_vec4(device, &stage_pos, "tire_dem grid step positions")?,
            velocities: read_vec4(device, &stage_vel, "tire_dem grid step velocities")?,
        })
    }

    pub fn dispatch_lbvh_step_blocking(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input: &TireDemLbvhStepGpuInput,
    ) -> Result<TireDemGpuOutput> {
        let dispatch_plan = tire_dem_sparse_lbvh_dispatch_plan(input)?;
        let n = dispatch_plan.grain_count;
        let contact_len = input.contacts.len().max(1);
        let lbvh_rows = tire_dem_sparse_lbvh_gpu_rows(&input.lbvh);
        debug_assert_eq!(lbvh_rows.len(), dispatch_plan.node_row_count);
        let vec4_bytes = |len: usize| (len.max(1) * core::mem::size_of::<[f32; 4]>()) as u64;
        let contact_bytes =
            |len: usize| (len.max(1) * core::mem::size_of::<TireDemContactCell>()) as u64;
        let u32_bytes = |len: usize| (len.max(1) * core::mem::size_of::<u32>()) as u64;
        let mk_buf = |label: &str, bytes: u64, usage: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes.max(4),
                usage,
                mapped_at_creation: false,
            })
        };
        let usage_ro = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let usage_rw = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        let pos_in = mk_buf("tire_dem lbvh step pos in", vec4_bytes(n), usage_ro);
        let vel_in = mk_buf("tire_dem lbvh step vel in", vec4_bytes(n), usage_ro);
        let contacts = mk_buf(
            "tire_dem lbvh step contacts",
            contact_bytes(contact_len),
            usage_ro,
        );
        let pos_out = mk_buf("tire_dem lbvh step pos out", vec4_bytes(n), usage_rw);
        let vel_out = mk_buf("tire_dem lbvh step vel out", vec4_bytes(n), usage_rw);
        let lbvh_nodes = mk_buf(
            "tire_dem lbvh step node rows",
            vec4_bytes(lbvh_rows.len()),
            usage_ro,
        );
        let lbvh_indices = mk_buf(
            "tire_dem lbvh step sorted indices",
            u32_bytes(input.lbvh.sorted_indices.len()),
            usage_ro,
        );
        let uniforms = mk_buf(
            "tire_dem lbvh step uniforms",
            core::mem::size_of::<TireDemStepUniforms>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let stage_pos = stage_vec4(device, "tire_dem lbvh step stage pos", n);
        let stage_vel = stage_vec4(device, "tire_dem lbvh step stage vel", n);

        queue.write_buffer(&pos_in, 0, bytemuck::cast_slice(&input.positions));
        queue.write_buffer(&vel_in, 0, bytemuck::cast_slice(&input.velocities));
        if input.contacts.is_empty() {
            queue.write_buffer(
                &contacts,
                0,
                bytemuck::cast_slice(&[TireDemContactCell::default()]),
            );
        } else {
            queue.write_buffer(&contacts, 0, bytemuck::cast_slice(&input.contacts));
        }
        queue.write_buffer(&lbvh_nodes, 0, bytemuck::cast_slice(&lbvh_rows));
        queue.write_buffer(
            &lbvh_indices,
            0,
            bytemuck::cast_slice(&input.lbvh.sorted_indices),
        );
        let p = input.params;
        queue.write_buffer(
            &uniforms,
            0,
            bytemuck::bytes_of(&TireDemStepUniforms {
                dims: [
                    n as u32,
                    input.contacts.len() as u32,
                    input.lbvh.nodes.len() as u32,
                    input.lbvh.sorted_indices.len() as u32,
                ],
                params0: [
                    p.grain_mass,
                    p.grain_radius,
                    p.dt,
                    p.tire_position_push_scale,
                ],
                params1: [p.normal_stiffness, p.damping, p.friction, p.floor_y],
                gravity: [p.gravity[0], p.gravity[1], p.gravity[2], 0.0],
                grid_origin_cell_size: [0.0; 4],
                grid_shape: [0; 4],
            }),
        );

        let bg_io = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_dem lbvh step io bg"),
            layout: &self.layout_lbvh_io,
            entries: &[
                entry(0, &pos_in),
                entry(1, &vel_in),
                entry(2, &contacts),
                entry(3, &pos_out),
                entry(4, &vel_out),
                entry(10, &lbvh_nodes),
                entry(11, &lbvh_indices),
            ],
        });
        let bg_uniforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_dem lbvh step uniforms bg"),
            layout: &self.layout_step_uniforms,
            entries: &[entry(0, &uniforms)],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::tire_dem lbvh step encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::tire_dem lbvh step pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.lbvh_step_pipeline);
            pass.set_bind_group(0, &bg_io, &[]);
            pass.set_bind_group(1, &bg_uniforms, &[]);
            pass.dispatch_workgroups(dispatch_plan.workgroup_count, 1, 1);
        }
        let bytes = vec4_bytes(n);
        encoder.copy_buffer_to_buffer(&pos_out, 0, &stage_pos, 0, bytes);
        encoder.copy_buffer_to_buffer(&vel_out, 0, &stage_vel, 0, bytes);
        queue.submit(std::iter::once(encoder.finish()));

        Ok(TireDemGpuOutput {
            positions: read_vec4(device, &stage_pos, "tire_dem lbvh step positions")?,
            velocities: read_vec4(device, &stage_vel, "tire_dem lbvh step velocities")?,
        })
    }

    pub fn dispatch_lbvh_step_resident(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input: &TireDemLbvhStepGpuInput,
    ) -> Result<TireDemResidentOutput> {
        let state =
            self.create_resident_state(device, queue, &input.positions, &input.velocities)?;
        self.dispatch_lbvh_step_resident_state(
            device,
            queue,
            &state,
            &input.contacts,
            input.params,
            &input.lbvh,
        )
    }

    pub fn create_resident_state(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        positions: &[[f32; 4]],
        velocities: &[[f32; 4]],
    ) -> Result<TireDemResidentState> {
        if positions.len() != velocities.len() {
            return Err(Error::Solver(
                "tire_dem_resident_state: positions/velocities length mismatch".into(),
            ));
        }
        if positions.is_empty() {
            return Err(Error::Solver(
                "tire_dem_resident_state: at least one grain required".into(),
            ));
        }
        if positions.iter().flatten().any(|v| !v.is_finite())
            || velocities.iter().flatten().any(|v| !v.is_finite())
        {
            return Err(Error::Solver(
                "tire_dem_resident_state: rows must be finite".into(),
            ));
        }
        let bytes = (positions.len().max(1) * core::mem::size_of::<[f32; 4]>()) as u64;
        let usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        let pos = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tire_dem resident state positions"),
            size: bytes.max(4),
            usage,
            mapped_at_creation: false,
        });
        let vel = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tire_dem resident state velocities"),
            size: bytes.max(4),
            usage,
            mapped_at_creation: false,
        });
        queue.write_buffer(&pos, 0, bytemuck::cast_slice(positions));
        queue.write_buffer(&vel, 0, bytemuck::cast_slice(velocities));
        Ok(TireDemResidentState {
            positions: pos,
            velocities: vel,
            grain_count: positions.len(),
        })
    }

    pub fn update_resident_state(
        &self,
        queue: &wgpu::Queue,
        state: &TireDemResidentState,
        positions: &[[f32; 4]],
        velocities: &[[f32; 4]],
    ) -> Result<()> {
        if positions.len() != velocities.len() || positions.len() != state.grain_count {
            return Err(Error::Solver(
                "tire_dem_resident_state_update: positions/velocities/state length mismatch".into(),
            ));
        }
        if positions.is_empty() {
            return Err(Error::Solver(
                "tire_dem_resident_state_update: at least one grain required".into(),
            ));
        }
        if positions.iter().flatten().any(|v| !v.is_finite())
            || velocities.iter().flatten().any(|v| !v.is_finite())
        {
            return Err(Error::Solver(
                "tire_dem_resident_state_update: rows must be finite".into(),
            ));
        }
        queue.write_buffer(&state.positions, 0, bytemuck::cast_slice(positions));
        queue.write_buffer(&state.velocities, 0, bytemuck::cast_slice(velocities));
        Ok(())
    }

    pub fn dispatch_lbvh_step_resident_state(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        state: &TireDemResidentState,
        contacts_in: &[TireDemContactCell],
        params: TireDemStepParams,
        lbvh: &TireDemSparseLbvh,
    ) -> Result<TireDemResidentOutput> {
        let resident_lbvh = self.upload_resident_lbvh(device, queue, state.grain_count, lbvh)?;
        self.dispatch_lbvh_step_resident_state_with_resident_lbvh(
            device,
            queue,
            state,
            contacts_in,
            params,
            &resident_lbvh,
        )
    }

    pub fn upload_resident_lbvh(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        grain_count: usize,
        lbvh: &TireDemSparseLbvh,
    ) -> Result<TireDemResidentLbvh> {
        let dispatch_plan = tire_dem_sparse_lbvh_dispatch_plan_from_parts(grain_count, 0, lbvh)?;
        let lbvh_rows = tire_dem_sparse_lbvh_gpu_rows(lbvh);
        debug_assert_eq!(lbvh_rows.len(), dispatch_plan.node_row_count);
        let vec4_bytes = |len: usize| (len.max(1) * core::mem::size_of::<[f32; 4]>()) as u64;
        let u32_bytes = |len: usize| (len.max(1) * core::mem::size_of::<u32>()) as u64;
        let mk_buf = |label: &str, bytes: u64, usage: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes.max(4),
                usage,
                mapped_at_creation: false,
            })
        };
        let usage_ro = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let lbvh_nodes = mk_buf(
            "tire_dem resident lbvh node rows",
            vec4_bytes(lbvh_rows.len()),
            usage_ro,
        );
        let lbvh_indices = mk_buf(
            "tire_dem resident lbvh sorted indices",
            u32_bytes(lbvh.sorted_indices.len()),
            usage_ro,
        );
        queue.write_buffer(&lbvh_nodes, 0, bytemuck::cast_slice(&lbvh_rows));
        queue.write_buffer(&lbvh_indices, 0, bytemuck::cast_slice(&lbvh.sorted_indices));
        Ok(TireDemResidentLbvh {
            node_rows: lbvh_nodes,
            sorted_indices: lbvh_indices,
            grain_count,
            node_count: dispatch_plan.node_count,
            node_row_count: dispatch_plan.node_row_count,
            sorted_index_count: lbvh.sorted_indices.len(),
            node_row_capacity: lbvh_rows.len().max(1),
            sorted_index_capacity: lbvh.sorted_indices.len().max(1),
            leaf_size: lbvh.leaf_size,
            topology_revision: 0,
        })
    }

    /// Upload a GPU-built sparse-LBVH tree row atlas as resident topology.
    ///
    /// `tire_dem_build_sparse_lbvh_tree_rows_gpu_radix` currently reads the
    /// tree rows back for validation, so this is still a bridge seam rather
    /// than a fully no-readback builder. It does let the DEM resident step use
    /// the GPU-built Morton/leaf/internal row layout directly instead of
    /// reserializing a CPU `TireDemSparseLbvh` tree.
    pub fn upload_resident_lbvh_tree_rows(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        grain_count: usize,
        tree: &TireDemGpuLbvhTreeRows,
    ) -> Result<TireDemResidentLbvh> {
        if grain_count == 0 || tree.sorted_indices.len() != grain_count {
            return Err(Error::Solver(
                "tire_dem_resident_lbvh_tree_rows: grain/index count mismatch".into(),
            ));
        }
        if tree.node_count == 0 || tree.node_rows.len() != tree.node_count * 3 {
            return Err(Error::Solver(
                "tire_dem_resident_lbvh_tree_rows: invalid node row layout".into(),
            ));
        }
        if tree
            .sorted_indices
            .iter()
            .any(|&idx| idx as usize >= grain_count)
        {
            return Err(Error::Solver(
                "tire_dem_resident_lbvh_tree_rows: sorted index out of range".into(),
            ));
        }
        let vec4_bytes = |len: usize| (len.max(1) * core::mem::size_of::<[f32; 4]>()) as u64;
        let u32_bytes = |len: usize| (len.max(1) * core::mem::size_of::<u32>()) as u64;
        let usage_ro = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let lbvh_nodes = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tire_dem resident gpu-built lbvh node rows"),
            size: vec4_bytes(tree.node_rows.len()),
            usage: usage_ro,
            mapped_at_creation: false,
        });
        let lbvh_indices = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tire_dem resident gpu-built lbvh sorted indices"),
            size: u32_bytes(tree.sorted_indices.len()),
            usage: usage_ro,
            mapped_at_creation: false,
        });
        queue.write_buffer(&lbvh_nodes, 0, bytemuck::cast_slice(&tree.node_rows));
        queue.write_buffer(&lbvh_indices, 0, bytemuck::cast_slice(&tree.sorted_indices));
        Ok(TireDemResidentLbvh {
            node_rows: lbvh_nodes,
            sorted_indices: lbvh_indices,
            grain_count,
            node_count: tree.node_count,
            node_row_count: tree.node_rows.len(),
            sorted_index_count: tree.sorted_indices.len(),
            node_row_capacity: tree.node_rows.len().max(1),
            sorted_index_capacity: tree.sorted_indices.len().max(1),
            leaf_size: tree.leaf_size,
            topology_revision: 0,
        })
    }

    /// Build sparse-LBVH tree rows directly into resident GPU topology buffers.
    ///
    /// This is the no-readback counterpart to
    /// [`tire_dem_build_sparse_lbvh_tree_rows_gpu_radix`]. Callers provide the
    /// scene bounds used for Morton quantization; the key generation, radix
    /// sort, leaf AABB rows, arbitrary-leaf-count padding, and bottom-up
    /// internal tree rows all stay on GPU, and the returned
    /// [`TireDemResidentLbvh`] can be passed directly to
    /// [`Self::dispatch_lbvh_step_resident_state_with_resident_lbvh`].
    pub fn build_resident_sparse_lbvh_tree_rows_gpu_radix(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        state: &TireDemResidentState,
        scene_min: [f32; 3],
        scene_extent: [f32; 3],
        leaf_size: usize,
    ) -> Result<TireDemResidentLbvh> {
        let n = state.grain_count;
        if n == 0 {
            return Err(Error::Solver(
                "tire_dem_resident_sparse_lbvh_tree_build: at least one grain required".into(),
            ));
        }
        if n > u32::MAX as usize {
            return Err(Error::Solver(
                "tire_dem_resident_sparse_lbvh_tree_build: grain count exceeds u32 indexing".into(),
            ));
        }
        if scene_min.iter().any(|v| !v.is_finite())
            || scene_extent.iter().any(|v| !v.is_finite() || *v <= 0.0)
        {
            return Err(Error::Solver(
                "tire_dem_resident_sparse_lbvh_tree_build: finite positive scene extent required"
                    .into(),
            ));
        }
        let leaf_size = leaf_size.clamp(1, 32);
        let leaf_count = n.div_ceil(leaf_size);
        let padded_leaf_count = leaf_count.next_power_of_two();
        let leaf_node_offset = padded_leaf_count - 1;
        let node_count = padded_leaf_count * 2 - 1;

        let u32_bytes = |len: usize| (len.max(1) * core::mem::size_of::<u32>()) as u64;
        let vec4_bytes = |len: usize| (len.max(1) * core::mem::size_of::<[f32; 4]>()) as u64;
        let storage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        let keys_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tire_dem resident sparse lbvh gpu keys"),
            size: u32_bytes(n),
            usage: storage,
            mapped_at_creation: false,
        });
        let sorted_indices = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tire_dem resident sparse lbvh gpu sorted indices"),
            size: u32_bytes(n),
            usage: storage,
            mapped_at_creation: false,
        });
        let node_rows = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tire_dem resident sparse lbvh gpu tree rows"),
            size: vec4_bytes(node_count * 3),
            usage: storage,
            mapped_at_creation: false,
        });
        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tire_dem resident sparse lbvh gpu build uniforms"),
            size: core::mem::size_of::<TireDemStepUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bg_data = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_dem resident sparse lbvh build bg"),
            layout: &self.layout_sparse_lbvh_build_io,
            entries: &[
                entry(0, &state.positions),
                entry(3, &node_rows),
                entry(5, &sorted_indices),
                entry(9, &keys_buf),
            ],
        });
        let bg_uniforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_dem resident sparse lbvh build uniforms bg"),
            layout: &self.layout_step_uniforms,
            entries: &[entry(0, &uniforms)],
        });

        queue.write_buffer(
            &uniforms,
            0,
            bytemuck::bytes_of(&TireDemStepUniforms {
                dims: [n as u32, padded_leaf_count as u32, leaf_size as u32, 0],
                params0: [0.0; 4],
                params1: [scene_extent[0], scene_extent[1], scene_extent[2], 0.0],
                gravity: [0.0; 4],
                grid_origin_cell_size: [scene_min[0], scene_min[1], scene_min[2], 0.0],
                grid_shape: [
                    leaf_node_offset as u32,
                    0,
                    node_count as u32,
                    leaf_count as u32,
                ],
            }),
        );
        let mut key_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::tire_dem resident sparse lbvh key encoder"),
        });
        {
            let mut pass = key_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::tire_dem resident sparse lbvh key pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.sparse_lbvh_morton_pipeline);
            pass.set_bind_group(0, &bg_data, &[]);
            pass.set_bind_group(1, &bg_uniforms, &[]);
            pass.dispatch_workgroups((n as u32).div_ceil(WORKGROUP_SIZE), 1, 1);
        }
        queue.submit(std::iter::once(key_encoder.finish()));

        molla_compute_primitives::radix_sort_pairs(
            Arc::new(device.clone()),
            Arc::new(queue.clone()),
            &keys_buf,
            &sorted_indices,
            n,
        )?;

        queue.write_buffer(
            &uniforms,
            0,
            bytemuck::bytes_of(&TireDemStepUniforms {
                dims: [n as u32, padded_leaf_count as u32, leaf_size as u32, 0],
                params0: [0.0; 4],
                params1: [scene_extent[0], scene_extent[1], scene_extent[2], 0.0],
                gravity: [0.0; 4],
                grid_origin_cell_size: [scene_min[0], scene_min[1], scene_min[2], 0.0],
                grid_shape: [
                    leaf_node_offset as u32,
                    0,
                    node_count as u32,
                    leaf_count as u32,
                ],
            }),
        );
        let mut leaf_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::tire_dem resident sparse lbvh leaf encoder"),
        });
        {
            let mut pass = leaf_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::tire_dem resident sparse lbvh leaf pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.sparse_lbvh_leaf_pipeline);
            pass.set_bind_group(0, &bg_data, &[]);
            pass.set_bind_group(1, &bg_uniforms, &[]);
            pass.dispatch_workgroups((padded_leaf_count as u32).div_ceil(WORKGROUP_SIZE), 1, 1);
        }
        queue.submit(std::iter::once(leaf_encoder.finish()));

        let mut child_start = leaf_node_offset;
        let mut child_count = padded_leaf_count;
        while child_count > 1 {
            let parent_count = child_count / 2;
            let parent_start = child_start - parent_count;
            queue.write_buffer(
                &uniforms,
                0,
                bytemuck::bytes_of(&TireDemStepUniforms {
                    dims: [n as u32, parent_count as u32, 0, 0],
                    params0: [0.0; 4],
                    params1: [scene_extent[0], scene_extent[1], scene_extent[2], 0.0],
                    gravity: [0.0; 4],
                    grid_origin_cell_size: [scene_min[0], scene_min[1], scene_min[2], 0.0],
                    grid_shape: [
                        parent_start as u32,
                        child_start as u32,
                        node_count as u32,
                        0,
                    ],
                }),
            );
            let mut level_encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("molla::tire_dem resident sparse lbvh internal encoder"),
                });
            {
                let mut pass = level_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("molla::tire_dem resident sparse lbvh internal pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.sparse_lbvh_internal_pipeline);
                pass.set_bind_group(0, &bg_data, &[]);
                pass.set_bind_group(1, &bg_uniforms, &[]);
                pass.dispatch_workgroups((parent_count as u32).div_ceil(WORKGROUP_SIZE), 1, 1);
            }
            queue.submit(std::iter::once(level_encoder.finish()));
            child_start = parent_start;
            child_count = parent_count;
        }

        Ok(TireDemResidentLbvh {
            node_rows,
            sorted_indices,
            grain_count: n,
            node_count,
            node_row_count: node_count * 3,
            sorted_index_count: n,
            node_row_capacity: (node_count * 3).max(1),
            sorted_index_capacity: n.max(1),
            leaf_size: leaf_size as u32,
            topology_revision: 0,
        })
    }

    /// Refresh resident sparse-LBVH topology buffers in place when the new
    /// CPU-built topology fits the existing allocations.
    ///
    /// Returns `Ok(true)` when the existing buffers were rewritten and can be
    /// reused for the next dispatch. Returns `Ok(false)` when the active
    /// topology needs larger buffers or a different grain count, in which case
    /// callers should allocate a fresh [`TireDemResidentLbvh`] with
    /// [`Self::upload_resident_lbvh`]. This keeps the current CPU builder
    /// honest while removing per-substep topology buffer churn for stable DEM
    /// beds; a true GPU LBVH builder is still the next deeper step.
    /// Returns a [`ResidentLbvhUpdate`] telling the caller exactly
    /// what happened — `GrainCountMismatch` and `CapacityExceeded`
    /// need DIFFERENT remedies (rebuild state vs reallocate buffers);
    /// the old boolean conflated them (2026-07-05 tire audit).
    pub fn update_resident_lbvh(
        &self,
        queue: &wgpu::Queue,
        resident: &mut TireDemResidentLbvh,
        grain_count: usize,
        lbvh: &TireDemSparseLbvh,
    ) -> Result<ResidentLbvhUpdate> {
        let dispatch_plan = tire_dem_sparse_lbvh_dispatch_plan_from_parts(grain_count, 0, lbvh)?;
        let lbvh_rows = tire_dem_sparse_lbvh_gpu_rows(lbvh);
        debug_assert_eq!(lbvh_rows.len(), dispatch_plan.node_row_count);
        if resident.grain_count != grain_count {
            return Ok(ResidentLbvhUpdate::GrainCountMismatch);
        }
        if lbvh_rows.len() > resident.node_row_capacity
            || lbvh.sorted_indices.len() > resident.sorted_index_capacity
        {
            return Ok(ResidentLbvhUpdate::CapacityExceeded);
        }
        queue.write_buffer(&resident.node_rows, 0, bytemuck::cast_slice(&lbvh_rows));
        queue.write_buffer(
            &resident.sorted_indices,
            0,
            bytemuck::cast_slice(&lbvh.sorted_indices),
        );
        resident.node_count = dispatch_plan.node_count;
        resident.node_row_count = dispatch_plan.node_row_count;
        resident.sorted_index_count = lbvh.sorted_indices.len();
        resident.leaf_size = lbvh.leaf_size;
        resident.topology_revision = resident.topology_revision.saturating_add(1);
        Ok(ResidentLbvhUpdate::Updated)
    }

    pub fn dispatch_lbvh_step_resident_state_with_resident_lbvh(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        state: &TireDemResidentState,
        contacts_in: &[TireDemContactCell],
        params: TireDemStepParams,
        lbvh: &TireDemResidentLbvh,
    ) -> Result<TireDemResidentOutput> {
        if lbvh.grain_count != state.grain_count {
            return Err(Error::Solver(
                "tire_dem_lbvh_resident: LBVH grain count must match resident state".into(),
            ));
        }
        if lbvh.node_count == 0 || lbvh.node_row_count != lbvh.node_count * 3 {
            return Err(Error::Solver(
                "tire_dem_lbvh_resident: invalid resident LBVH row layout".into(),
            ));
        }
        let n = state.grain_count;
        let workgroup_count = (n as u32).div_ceil(WORKGROUP_SIZE);
        let contact_len = contacts_in.len().max(1);
        let vec4_bytes = |len: usize| (len.max(1) * core::mem::size_of::<[f32; 4]>()) as u64;
        let contact_bytes =
            |len: usize| (len.max(1) * core::mem::size_of::<TireDemContactCell>()) as u64;
        let mk_buf = |label: &str, bytes: u64, usage: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes.max(4),
                usage,
                mapped_at_creation: false,
            })
        };
        let usage_rw = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        let usage_ro = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let contacts = mk_buf(
            "tire_dem lbvh resident contacts",
            contact_bytes(contact_len),
            usage_ro,
        );
        let pos_out = mk_buf("tire_dem lbvh resident pos out", vec4_bytes(n), usage_rw);
        let vel_out = mk_buf("tire_dem lbvh resident vel out", vec4_bytes(n), usage_rw);
        let uniforms = mk_buf(
            "tire_dem lbvh resident uniforms",
            core::mem::size_of::<TireDemStepUniforms>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );

        if contacts_in.is_empty() {
            queue.write_buffer(
                &contacts,
                0,
                bytemuck::cast_slice(&[TireDemContactCell::default()]),
            );
        } else {
            queue.write_buffer(&contacts, 0, bytemuck::cast_slice(contacts_in));
        }
        let p = params;
        queue.write_buffer(
            &uniforms,
            0,
            bytemuck::bytes_of(&TireDemStepUniforms {
                dims: [
                    n as u32,
                    contacts_in.len() as u32,
                    lbvh.node_count as u32,
                    lbvh.sorted_index_count as u32,
                ],
                params0: [
                    p.grain_mass,
                    p.grain_radius,
                    p.dt,
                    p.tire_position_push_scale,
                ],
                params1: [p.normal_stiffness, p.damping, p.friction, p.floor_y],
                gravity: [p.gravity[0], p.gravity[1], p.gravity[2], 0.0],
                grid_origin_cell_size: [0.0; 4],
                grid_shape: [0; 4],
            }),
        );

        let bg_io = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_dem lbvh resident io bg"),
            layout: &self.layout_lbvh_io,
            entries: &[
                entry(0, &state.positions),
                entry(1, &state.velocities),
                entry(2, &contacts),
                entry(3, &pos_out),
                entry(4, &vel_out),
                entry(10, &lbvh.node_rows),
                entry(11, &lbvh.sorted_indices),
            ],
        });
        let bg_uniforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_dem lbvh resident uniforms bg"),
            layout: &self.layout_step_uniforms,
            entries: &[entry(0, &uniforms)],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::tire_dem lbvh resident encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::tire_dem lbvh resident pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.lbvh_step_pipeline);
            pass.set_bind_group(0, &bg_io, &[]);
            pass.set_bind_group(1, &bg_uniforms, &[]);
            pass.dispatch_workgroups(workgroup_count, 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));

        Ok(TireDemResidentOutput {
            positions: pos_out,
            velocities: vel_out,
            grain_count: n,
            dispatch_plan: TireDemLbvhDispatchPlan {
                grain_count: n,
                contact_count: contacts_in.len(),
                node_count: lbvh.node_count,
                node_row_count: lbvh.node_row_count,
                workgroup_count,
                storage_bytes: (n as u64)
                    .saturating_mul(core::mem::size_of::<[f32; 4]>() as u64)
                    .saturating_mul(4)
                    .saturating_add(
                        (lbvh.node_row_count as u64)
                            .saturating_mul(core::mem::size_of::<[f32; 4]>() as u64),
                    )
                    .saturating_add(
                        (lbvh.sorted_index_count as u64)
                            .saturating_mul(core::mem::size_of::<u32>() as u64),
                    )
                    .saturating_add(
                        (contacts_in.len().max(1) as u64)
                            .saturating_mul(core::mem::size_of::<TireDemContactCell>() as u64),
                    ),
            },
        })
    }

    pub fn readback_resident_state(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        state: &TireDemResidentState,
    ) -> Result<TireDemGpuOutput> {
        let stage_pos = stage_vec4(
            device,
            "tire_dem resident state stage pos",
            state.grain_count,
        );
        let stage_vel = stage_vec4(
            device,
            "tire_dem resident state stage vel",
            state.grain_count,
        );
        let bytes = (state.grain_count.max(1) * core::mem::size_of::<[f32; 4]>()) as u64;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::tire_dem resident state readback encoder"),
        });
        encoder.copy_buffer_to_buffer(&state.positions, 0, &stage_pos, 0, bytes);
        encoder.copy_buffer_to_buffer(&state.velocities, 0, &stage_vel, 0, bytes);
        queue.submit(std::iter::once(encoder.finish()));
        Ok(TireDemGpuOutput {
            positions: read_vec4(device, &stage_pos, "tire_dem resident state positions")?,
            velocities: read_vec4(device, &stage_vel, "tire_dem resident state velocities")?,
        })
    }

    pub fn dispatch_resident_terrain_raster(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        state: &TireDemResidentState,
        reference_heights: &[f32],
        params: TireDemResidentTerrainParams,
    ) -> Result<TireDemResidentTerrainOutput> {
        let row_count =
            validate_dem_resident_terrain_params(state.grain_count, reference_heights, params)?;
        let f32_bytes = |len: usize| (len.max(1) * core::mem::size_of::<f32>()) as u64;
        let vec4_bytes = |len: usize| (len.max(1) * core::mem::size_of::<[f32; 4]>()) as u64;
        let mk_buf = |label: &str, bytes: u64, usage: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes.max(4),
                usage,
                mapped_at_creation: false,
            })
        };
        let reference = mk_buf(
            "tire_dem resident terrain reference heights",
            f32_bytes(row_count),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let heights = mk_buf(
            "tire_dem resident terrain heights",
            f32_bytes(row_count),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let sinkage = mk_buf(
            "tire_dem resident terrain sinkage",
            f32_bytes(row_count),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let surface_velocity_rows = mk_buf(
            "tire_dem resident terrain surface velocity rows",
            vec4_bytes(row_count),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let uniforms = mk_buf(
            "tire_dem resident terrain uniforms",
            core::mem::size_of::<TireDemStepUniforms>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        queue.write_buffer(&reference, 0, bytemuck::cast_slice(reference_heights));
        queue.write_buffer(
            &uniforms,
            0,
            bytemuck::bytes_of(&TireDemStepUniforms {
                dims: [state.grain_count as u32, params.nx, params.nz, 0],
                params0: [0.0, params.grain_radius, 0.0, 0.0],
                params1: [0.0, 0.0, 0.0, params.floor_y],
                gravity: [0.0; 4],
                grid_origin_cell_size: [params.origin[0], params.origin[1], 0.0, params.spacing],
                grid_shape: [params.nx, 1, params.nz, 0],
            }),
        );

        let bg_io = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_dem resident terrain io bg"),
            layout: &self.layout_terrain_io,
            entries: &[
                entry(0, &state.positions),
                entry(1, &state.velocities),
                entry(12, &reference),
                entry(13, &heights),
                entry(14, &sinkage),
                entry(15, &surface_velocity_rows),
            ],
        });
        let bg_uniforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_dem resident terrain uniforms bg"),
            layout: &self.layout_step_uniforms,
            entries: &[entry(0, &uniforms)],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::tire_dem resident terrain encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::tire_dem resident terrain pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.terrain_raster_pipeline);
            pass.set_bind_group(0, &bg_io, &[]);
            pass.set_bind_group(1, &bg_uniforms, &[]);
            pass.dispatch_workgroups((row_count as u32).div_ceil(WORKGROUP_SIZE), 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));

        Ok(TireDemResidentTerrainOutput {
            heights,
            sinkage,
            surface_velocity_rows,
            params,
        })
    }

    pub fn readback_resident_terrain(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        terrain: &TireDemResidentTerrainOutput,
    ) -> Result<TireDemResidentTerrainReadback> {
        let row_count = (terrain.params.nx as usize)
            .checked_mul(terrain.params.nz as usize)
            .ok_or_else(|| Error::Solver("tire_dem_resident_terrain: row count overflow".into()))?;
        let stage_heights = stage_f32(device, "tire_dem resident terrain stage heights", row_count);
        let stage_sinkage = stage_f32(device, "tire_dem resident terrain stage sinkage", row_count);
        let stage_surface_velocity = stage_vec4(
            device,
            "tire_dem resident terrain stage surface velocity",
            row_count,
        );
        let f32_bytes = (row_count.max(1) * core::mem::size_of::<f32>()) as u64;
        let vec4_bytes = (row_count.max(1) * core::mem::size_of::<[f32; 4]>()) as u64;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::tire_dem resident terrain readback encoder"),
        });
        encoder.copy_buffer_to_buffer(&terrain.heights, 0, &stage_heights, 0, f32_bytes);
        encoder.copy_buffer_to_buffer(&terrain.sinkage, 0, &stage_sinkage, 0, f32_bytes);
        encoder.copy_buffer_to_buffer(
            &terrain.surface_velocity_rows,
            0,
            &stage_surface_velocity,
            0,
            vec4_bytes,
        );
        queue.submit(std::iter::once(encoder.finish()));
        let rows = read_vec4(
            device,
            &stage_surface_velocity,
            "tire_dem resident terrain surface velocity rows",
        )?;
        let mut active_cells = 0_u32;
        let mut max_speed = 0.0_f32;
        let mut speed_sum = 0.0_f32;
        for row in &rows {
            if row[3] > 0.0 {
                active_cells += 1;
                let speed = (row[0] * row[0] + row[1] * row[1] + row[2] * row[2]).sqrt();
                max_speed = max_speed.max(speed);
                speed_sum += speed;
            }
        }
        Ok(TireDemResidentTerrainReadback {
            heights: read_f32(device, &stage_heights, "tire_dem resident terrain heights")?,
            sinkage: read_f32(device, &stage_sinkage, "tire_dem resident terrain sinkage")?,
            surface_velocity_map: TireDemSurfaceVelocityMap {
                origin: terrain.params.origin,
                spacing: terrain.params.spacing,
                nx: terrain.params.nx,
                nz: terrain.params.nz,
                rows,
                active_cells,
                max_speed_m_per_s: max_speed,
                mean_speed_m_per_s: if active_cells > 0 {
                    speed_sum / active_cells as f32
                } else {
                    0.0
                },
            },
        })
    }

    pub fn readback_resident_output(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        resident: &TireDemResidentOutput,
    ) -> Result<TireDemGpuOutput> {
        let stage_pos = stage_vec4(device, "tire_dem resident stage pos", resident.grain_count);
        let stage_vel = stage_vec4(device, "tire_dem resident stage vel", resident.grain_count);
        let bytes = (resident.grain_count.max(1) * core::mem::size_of::<[f32; 4]>()) as u64;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::tire_dem resident readback encoder"),
        });
        encoder.copy_buffer_to_buffer(&resident.positions, 0, &stage_pos, 0, bytes);
        encoder.copy_buffer_to_buffer(&resident.velocities, 0, &stage_vel, 0, bytes);
        queue.submit(std::iter::once(encoder.finish()));
        Ok(TireDemGpuOutput {
            positions: read_vec4(device, &stage_pos, "tire_dem resident positions")?,
            velocities: read_vec4(device, &stage_vel, "tire_dem resident velocities")?,
        })
    }

    pub fn dispatch_unbounded_radix_grid_step_blocking(
        &self,
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        input: &TireDemGpuRadixGridStepInput,
    ) -> Result<TireDemGpuOutput> {
        let n = input.validate()?;
        let cell_count = grid_cell_count(input.grid_shape)?;
        let contact_len = input.contacts.len().max(1);
        let vec4_bytes = |len: usize| (len.max(1) * core::mem::size_of::<[f32; 4]>()) as u64;
        let contact_bytes =
            |len: usize| (len.max(1) * core::mem::size_of::<TireDemContactCell>()) as u64;
        let u32_bytes = |len: usize| (len.max(1) * core::mem::size_of::<u32>()) as u64;
        let uvec4_bytes = |len: usize| (len.max(1) * core::mem::size_of::<[u32; 4]>()) as u64;
        let mk_buf = |label: &str, bytes: u64, usage: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes.max(4),
                usage,
                mapped_at_creation: false,
            })
        };
        let usage_ro = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let usage_rw = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        let pos_in = mk_buf("tire_dem unbounded radix pos in", vec4_bytes(n), usage_ro);
        let vel_in = mk_buf("tire_dem unbounded radix vel in", vec4_bytes(n), usage_ro);
        let contacts = mk_buf(
            "tire_dem unbounded radix contacts",
            contact_bytes(contact_len),
            usage_ro,
        );
        let pos_out = mk_buf("tire_dem unbounded radix pos out", vec4_bytes(n), usage_rw);
        let vel_out = mk_buf("tire_dem unbounded radix vel out", vec4_bytes(n), usage_rw);
        let sorted_cell_keys = mk_buf(
            "tire_dem unbounded radix sorted cell keys",
            u32_bytes(n),
            usage_rw,
        );
        let sorted_indices = mk_buf(
            "tire_dem unbounded radix sorted grain ids",
            u32_bytes(n),
            usage_rw,
        );
        let cell_ranges = mk_buf(
            "tire_dem unbounded radix cell ranges",
            uvec4_bytes(cell_count),
            usage_rw,
        );
        let uniforms = mk_buf(
            "tire_dem unbounded radix uniforms",
            core::mem::size_of::<TireDemStepUniforms>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let stage_pos = stage_vec4(device.as_ref(), "tire_dem unbounded radix stage pos", n);
        let stage_vel = stage_vec4(device.as_ref(), "tire_dem unbounded radix stage vel", n);

        queue.write_buffer(&pos_in, 0, bytemuck::cast_slice(&input.positions));
        queue.write_buffer(&vel_in, 0, bytemuck::cast_slice(&input.velocities));
        if input.contacts.is_empty() {
            queue.write_buffer(
                &contacts,
                0,
                bytemuck::cast_slice(&[TireDemContactCell::default()]),
            );
        } else {
            queue.write_buffer(&contacts, 0, bytemuck::cast_slice(&input.contacts));
        }
        let p = input.params;
        queue.write_buffer(
            &uniforms,
            0,
            bytemuck::bytes_of(&TireDemStepUniforms {
                dims: [n as u32, input.contacts.len() as u32, cell_count as u32, 0],
                params0: [
                    p.grain_mass,
                    p.grain_radius,
                    p.dt,
                    p.tire_position_push_scale,
                ],
                params1: [p.normal_stiffness, p.damping, p.friction, p.floor_y],
                gravity: [p.gravity[0], p.gravity[1], p.gravity[2], 0.0],
                grid_origin_cell_size: [
                    input.grid_origin[0],
                    input.grid_origin[1],
                    input.grid_origin[2],
                    input.cell_size,
                ],
                grid_shape: [
                    input.grid_shape[0],
                    input.grid_shape[1],
                    input.grid_shape[2],
                    0,
                ],
            }),
        );

        let bg_unbounded = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_dem unbounded radix build bg"),
            layout: &self.layout_unbounded_grid_io,
            entries: &[
                entry(0, &pos_in),
                entry(5, &sorted_indices),
                entry(6, &cell_ranges),
                entry(9, &sorted_cell_keys),
            ],
        });
        let bg_grid = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_dem unbounded radix step bg"),
            layout: &self.layout_grid_io,
            entries: &[
                entry(0, &pos_in),
                entry(1, &vel_in),
                entry(2, &contacts),
                entry(3, &pos_out),
                entry(4, &vel_out),
                entry(5, &sorted_indices),
                entry(6, &cell_ranges),
            ],
        });
        let bg_uniforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_dem unbounded radix uniforms bg"),
            layout: &self.layout_step_uniforms,
            entries: &[entry(0, &uniforms)],
        });

        let mut key_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::tire_dem unbounded radix key encoder"),
        });
        {
            let mut pass = key_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::tire_dem unbounded radix key pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.unbounded_key_pipeline);
            pass.set_bind_group(0, &bg_unbounded, &[]);
            pass.set_bind_group(1, &bg_uniforms, &[]);
            pass.dispatch_workgroups((n as u32).div_ceil(WORKGROUP_SIZE), 1, 1);
        }
        queue.submit(std::iter::once(key_encoder.finish()));

        molla_compute_primitives::radix_sort_pairs(
            Arc::clone(&device),
            Arc::clone(&queue),
            &sorted_cell_keys,
            &sorted_indices,
            n,
        )?;

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::tire_dem unbounded radix range/step encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::tire_dem unbounded radix range pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.unbounded_range_pipeline);
            pass.set_bind_group(0, &bg_unbounded, &[]);
            pass.set_bind_group(1, &bg_uniforms, &[]);
            pass.dispatch_workgroups((cell_count as u32).div_ceil(WORKGROUP_SIZE), 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::tire_dem unbounded radix step pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.grid_step_pipeline);
            pass.set_bind_group(0, &bg_grid, &[]);
            pass.set_bind_group(1, &bg_uniforms, &[]);
            pass.dispatch_workgroups((n as u32).div_ceil(WORKGROUP_SIZE), 1, 1);
        }
        let bytes = vec4_bytes(n);
        encoder.copy_buffer_to_buffer(&pos_out, 0, &stage_pos, 0, bytes);
        encoder.copy_buffer_to_buffer(&vel_out, 0, &stage_vel, 0, bytes);
        queue.submit(std::iter::once(encoder.finish()));

        Ok(TireDemGpuOutput {
            positions: read_vec4(
                device.as_ref(),
                &stage_pos,
                "tire_dem unbounded radix positions",
            )?,
            velocities: read_vec4(
                device.as_ref(),
                &stage_vel,
                "tire_dem unbounded radix velocities",
            )?,
        })
    }

    pub fn dispatch_gpu_built_grid_step_blocking(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input: &TireDemGpuBuiltGridStepInput,
    ) -> Result<TireDemGpuOutput> {
        input.validate()?;
        let radix_input = TireDemGpuRadixGridStepInput {
            positions: input.positions.clone(),
            velocities: input.velocities.clone(),
            contacts: input.contacts.clone(),
            params: input.params,
            grid_origin: input.grid_origin,
            cell_size: input.cell_size,
            grid_shape: input.grid_shape,
        };
        self.dispatch_unbounded_radix_grid_step_blocking(
            Arc::new((*device).clone()),
            Arc::new((*queue).clone()),
            &radix_input,
        )
    }
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn stage_vec4(device: &wgpu::Device, label: &str, len: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: (len.max(1) * core::mem::size_of::<[f32; 4]>()) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    })
}

fn stage_f32(device: &wgpu::Device, label: &str, len: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: (len.max(1) * core::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    })
}

fn stage_u32(device: &wgpu::Device, label: &str, len: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: (len.max(1) * core::mem::size_of::<u32>()) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    })
}

fn stage_uvec4(device: &wgpu::Device, label: &str, len: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: (len.max(1) * core::mem::size_of::<[u32; 4]>()) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    })
}

fn read_vec4(device: &wgpu::Device, buf: &wgpu::Buffer, label: &str) -> Result<Vec<[f32; 4]>> {
    let slice = buf.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv()
        .map_err(|_| Error::Solver(format!("{label}: map callback dropped")))?
        .map_err(|e| Error::Solver(format!("{label}: map failed: {e:?}")))?;
    let data = slice.get_mapped_range();
    let out = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    buf.unmap();
    Ok(out)
}

fn read_f32(device: &wgpu::Device, buf: &wgpu::Buffer, label: &str) -> Result<Vec<f32>> {
    let slice = buf.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv()
        .map_err(|_| Error::Solver(format!("{label}: map callback dropped")))?
        .map_err(|e| Error::Solver(format!("{label}: map failed: {e:?}")))?;
    let data = slice.get_mapped_range();
    let out = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    buf.unmap();
    Ok(out)
}

fn read_u32(device: &wgpu::Device, buf: &wgpu::Buffer, label: &str) -> Result<Vec<u32>> {
    let slice = buf.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv()
        .map_err(|_| Error::Solver(format!("{label}: map callback dropped")))?
        .map_err(|e| Error::Solver(format!("{label}: map failed: {e:?}")))?;
    let data = slice.get_mapped_range();
    let out = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    buf.unmap();
    Ok(out)
}

fn read_uvec4(device: &wgpu::Device, buf: &wgpu::Buffer, label: &str) -> Result<Vec<[u32; 4]>> {
    let slice = buf.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv()
        .map_err(|_| Error::Solver(format!("{label}: map callback dropped")))?
        .map_err(|e| Error::Solver(format!("{label}: map failed: {e:?}")))?;
    let data = slice.get_mapped_range();
    let out = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    buf.unmap();
    Ok(out)
}
