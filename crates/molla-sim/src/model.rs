//! `Model` — static, immutable simulation description.

use molla_core::{Error, Result};
use molla_math::{Mat3, Real, Transform, Vec3};
use molla_storage::{Buffer, BufferUsage};

/// Regular grid of height samples (heightmap) — terrain primitive.
/// Surface is sampled at `(origin.x + i·cell_size, origin.y + height[i,j],
/// origin.z + j·cell_size_z)` for `i ∈ [0, nx)`, `j ∈ [0, nz)`.
/// Collision routing converts only the cells overlapping the
/// query AABB into triangles, so heightmaps stay cheap even when
/// the grid is large.
#[derive(Clone, Debug, Default)]
pub struct Heightfield {
    /// Number of samples along the X axis.
    pub nx: u32,
    /// Number of samples along the Z axis.
    pub nz: u32,
    /// Distance between adjacent X samples (m).
    pub cell_size: Real,
    /// Distance between adjacent Z samples (m).
    pub cell_size_z: Real,
    /// Origin in shape-local frame: the `(0, 0)` sample sits at this point.
    pub origin: Vec3,
    /// Flat `nx · nz` array — row-major, `height[j*nx + i]`.
    pub heights: Vec<Real>,
}

impl Heightfield {
    /// Centered row-major grid: rows follow Z, columns follow X, heights scale by Y.
    pub fn from_scaled(rows: usize, cols: usize, heights: Vec<Real>, scale: Vec3) -> Result<Self> {
        if rows < 2
            || cols < 2
            || rows > u32::MAX as usize
            || cols > u32::MAX as usize
            || rows.checked_mul(cols) != Some(heights.len())
            || !scale.is_finite()
            || scale.x <= 0.0
            || scale.z <= 0.0
        {
            return Err(Error::Build("invalid scaled heightfield dimensions".into()));
        }
        let field = Self {
            nx: cols as u32,
            nz: rows as u32,
            cell_size: scale.x / (cols - 1) as Real,
            cell_size_z: scale.z / (rows - 1) as Real,
            origin: Vec3::new(-scale.x * 0.5, 0.0, -scale.z * 0.5),
            heights: heights.into_iter().map(|h| h * scale.y).collect(),
        };
        field.validate()?;
        Ok(field)
    }

    pub fn validate(&self) -> Result<()> {
        if self.nx < 2
            || self.nz < 2
            || self.nx as u64 * self.nz as u64 != self.heights.len() as u64
            || !self.origin.is_finite()
            || !self.cell_size.is_finite()
            || self.cell_size <= 0.0
            || !self.cell_size_z.is_finite()
            || self.cell_size_z <= 0.0
            || self
                .heights
                .iter()
                .any(|h| !h.is_finite() || !(self.origin.y + h).is_finite())
            || !(self.origin.x + (self.nx - 1) as Real * self.cell_size).is_finite()
            || !(self.origin.z + (self.nz - 1) as Real * self.cell_size_z).is_finite()
        {
            return Err(Error::Build("invalid heightfield".into()));
        }
        Ok(())
    }

    /// Sample height at integer cell coords. Out-of-range returns 0.
    pub fn sample(&self, i: u32, j: u32) -> Real {
        if i >= self.nx || j >= self.nz {
            return 0.0;
        }
        self.heights[j as usize * self.nx as usize + i as usize]
    }

    pub fn vertex(&self, i: u32, j: u32) -> Vec3 {
        Vec3::new(
            self.origin.x + i as Real * self.cell_size,
            self.origin.y + self.sample(i, j),
            self.origin.z + j as Real * self.cell_size_z,
        )
    }

    /// Upward-wound triangles split across the cell's X-max/Z-min to X-min/Z-max diagonal.
    pub fn cell_triangles(&self, i: u32, j: u32) -> Option<[[Vec3; 3]; 2]> {
        if i >= self.nx.saturating_sub(1) || j >= self.nz.saturating_sub(1) {
            return None;
        }
        let p00 = self.vertex(i, j);
        let p10 = self.vertex(i + 1, j);
        let p01 = self.vertex(i, j + 1);
        let p11 = self.vertex(i + 1, j + 1);
        Some([[p00, p01, p10], [p01, p11, p10]])
    }

    /// Piecewise-planar local height and unit normal on the collision triangles.
    pub fn height_and_normal(&self, x: Real, z: Real) -> Option<(Real, Vec3)> {
        if self.nx < 2 || self.nz < 2 {
            return None;
        }
        let gx = (x - self.origin.x) / self.cell_size;
        let gz = (z - self.origin.z) / self.cell_size_z;
        if !gx.is_finite()
            || !gz.is_finite()
            || gx < 0.0
            || gz < 0.0
            || gx > (self.nx - 1) as Real
            || gz > (self.nz - 1) as Real
        {
            return None;
        }
        let i = (gx.floor() as u32).min(self.nx - 2);
        let j = (gz.floor() as u32).min(self.nz - 2);
        let tx = gx - i as Real;
        let tz = gz - j as Real;
        let h00 = self.sample(i, j);
        let h10 = self.sample(i + 1, j);
        let h01 = self.sample(i, j + 1);
        let h11 = self.sample(i + 1, j + 1);
        let (height, dx, dz) = if tx + tz <= 1.0 {
            (
                h00 + (h10 - h00) * tx + (h01 - h00) * tz,
                h10 - h00,
                h01 - h00,
            )
        } else {
            (
                h11 + (h01 - h11) * (1.0 - tx) + (h10 - h11) * (1.0 - tz),
                h11 - h01,
                h11 - h10,
            )
        };
        Some((
            self.origin.y + height,
            Vec3::new(-dx / self.cell_size, 1.0, -dz / self.cell_size_z).normalize(),
        ))
    }

    /// Heightfield's local-frame AABB (min, max) — the range over
    /// the X/Z grid extents and the height sample range.
    ///
    /// Empty heightfields collapse to a zero-height degenerate AABB
    /// (min == max along Y); non-empty heightfields are required to
    /// contain only finite samples — a NaN height indicates an
    /// upstream loader bug and is caught by `debug_assert!`.
    pub fn local_aabb(&self) -> (Vec3, Vec3) {
        let max_x = self.origin.x + (self.nx.saturating_sub(1) as Real) * self.cell_size;
        let max_z = self.origin.z + (self.nz.saturating_sub(1) as Real) * self.cell_size_z;
        let mut min_h = Real::INFINITY;
        let mut max_h = -Real::INFINITY;
        for &h in &self.heights {
            debug_assert!(
                h.is_finite(),
                "Heightfield::local_aabb: non-finite height sample {h}"
            );
            if h < min_h {
                min_h = h;
            }
            if h > max_h {
                max_h = h;
            }
        }
        if self.heights.is_empty() {
            min_h = 0.0;
            max_h = 0.0;
        }
        (
            Vec3::new(self.origin.x, self.origin.y + min_h, self.origin.z),
            Vec3::new(max_x, self.origin.y + max_h, max_z),
        )
    }
}

/// Analytic signed-distance shape descriptor. v1 ships built-in
/// torus + sphere primitives; arbitrary user-defined SDFs need
/// procedural callbacks (Stage 13 follow-up).
#[derive(Copy, Clone, Debug)]
pub enum SdfShape {
    /// Torus along the local +Y axis.
    Torus {
        /// Distance from torus centre to ring centre (m).
        major: Real,
        /// Tube radius (m).
        minor: Real,
    },
    /// Sphere of radius `r` — useful as a smoke test against the
    /// existing analytic Sphere primitive.
    Sphere { r: Real },
    /// Helical thread around the local +Y axis (INSPECTION.md F15 /
    /// NutBoltSdf): a circle of radius `minor` swept along the helix
    /// of radius `major` with pitch `pitch` (advance per turn, m),
    /// capped to `|y| ≤ half_height`. The distance is evaluated in
    /// the wrapped profile plane `(r − major, y − pitch·θ/τ mod
    /// pitch)` — exact in that plane, with the standard small swept
    /// error on steep helices (`pitch ≪ τ·major` keeps it tight).
    HelicalThread {
        major: Real,
        minor: Real,
        pitch: Real,
        half_height: Real,
    },
}

impl SdfShape {
    /// Signed distance at point `p` in shape-local frame. `< 0` is
    /// inside, `> 0` is outside, `0` is the surface.
    pub fn signed_distance(self, p: Vec3) -> Real {
        match self {
            SdfShape::Torus { major, minor } => {
                let r_xz = (p.x * p.x + p.z * p.z).sqrt();
                let q = Vec3::new(r_xz - major, p.y, 0.0);
                (q.x * q.x + q.y * q.y).sqrt() - minor
            }
            SdfShape::Sphere { r } => p.length() - r,
            SdfShape::HelicalThread {
                major,
                minor,
                pitch,
                half_height,
            } => {
                let r_xz = (p.x * p.x + p.z * p.z).sqrt();
                let theta = p.z.atan2(p.x);
                let tau = core::f64::consts::TAU as Real;
                // Screw coordinate: distance along Y from the nearest
                // thread crest passing at angle θ.
                let v_raw = p.y - pitch * theta / tau;
                let v = v_raw - pitch * (v_raw / pitch).round();
                let d_profile = ((r_xz - major) * (r_xz - major) + v * v).sqrt();
                let d_thread = d_profile - minor;
                // Cap the helix to a finite bolt segment.
                d_thread.max(p.y.abs() - half_height)
            }
        }
    }

    /// Outward-facing surface normal at `p` in shape-local frame
    /// — finite-difference gradient of `signed_distance`.
    pub fn normal(self, p: Vec3) -> Vec3 {
        let h: Real = 1.0e-4;
        let dx = self.signed_distance(p + Vec3::new(h, 0.0, 0.0))
            - self.signed_distance(p - Vec3::new(h, 0.0, 0.0));
        let dy = self.signed_distance(p + Vec3::new(0.0, h, 0.0))
            - self.signed_distance(p - Vec3::new(0.0, h, 0.0));
        let dz = self.signed_distance(p + Vec3::new(0.0, 0.0, h))
            - self.signed_distance(p - Vec3::new(0.0, 0.0, h));
        let n = Vec3::new(dx, dy, dz);
        if n.length_squared() > 1.0e-24 {
            n.normalize()
        } else {
            Vec3::Y
        }
    }
}

/// vertex indices forming triangle `t`, wound CCW when looking
/// against the outward normal (right-hand rule).
#[derive(Clone, Debug, Default)]
pub struct TriMesh {
    pub vertices: Vec<Vec3>,
    pub indices: Vec<[u32; 3]>,
}

impl TriMesh {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Compatibility no-op retained while mesh acceleration ownership moves
    /// out of `sim` and into `geometry`.
    ///
    /// `TriMesh` is now raw simulation data only. Geometry callers build
    /// mesh BVH caches from `vertices` + `indices` at the query/bake boundary
    /// instead of storing a geometry cache in the model.
    pub fn build_bvh(&mut self, _max_leaf: usize) {}
}

/// Static description of a simulation. Built once from a `ModelBuilder`
/// and shared across many [`State`](super::State) instances.
#[derive(Debug)]
pub struct Model {
    // ── Particles ───────────────────────────────────────────────────
    /// Number of particles.
    pub particle_count: usize,
    /// Per-particle mass (kg).
    pub particle_mass: Buffer<Real>,
    /// Per-particle inverse mass (1/kg), 0 for kinematic.
    pub particle_inv_mass: Buffer<Real>,
    /// Per-particle radius (m).
    pub particle_radius: Buffer<Real>,
    /// Per-particle bitflags (see [`super::ParticleFlags`]).
    pub particle_flags: Buffer<u32>,
    /// Per-particle world index (-1 = global).
    pub particle_world: Buffer<i32>,
    /// Initial particle positions snapshotted at build time. Reset via
    /// `State::for_model`.
    pub initial_particle_q: Buffer<Vec3>,

    // ── Springs / distance constraints ──────────────────────────────
    /// Number of distance constraints (cloth edges, ropes, springs).
    pub spring_count: usize,
    /// Endpoint-A particle index.
    pub spring_a: Buffer<u32>,
    /// Endpoint-B particle index.
    pub spring_b: Buffer<u32>,
    /// Rest length of each constraint (m).
    pub spring_rest_length: Buffer<Real>,
    /// XPBD compliance α (m/N). 0 = perfectly rigid, ~1e-7 = stiff
    /// stretch, ~1e-2 = soft. `(1 / stiffness)` if you'd think in
    /// terms of a spring constant.
    pub spring_compliance: Buffer<Real>,
    /// Per-spring hard stretch cap as a ratio of rest length:
    /// `length ≤ rest · (1 + max_stretch)` is enforced as a
    /// post-XPBD non-linear projection. 0.0 means perfectly
    /// inextensible (length capped at rest length). 0.05 = 5 %
    /// stretchy linen. >= ~1.0 effectively disables the cap.
    pub spring_max_stretch: Buffer<Real>,
    /// One for tension-only axial elements, zero for bilateral springs.
    pub spring_tension_only: Buffer<u32>,

    // ── Spring graph colouring ──────────────────────────────────────
    // Computed once at build time so the GPU XPBD kernel can solve
    // distance / strain-limit constraints in parallel: within a
    // colour, no two springs share a particle, so threads can
    // read-modify-write `particle_q` without races.
    /// Number of colours used by the greedy graph coloring.
    pub spring_color_count: usize,
    /// CSR offsets over `spring_indices_by_color` — colour `k` owns
    /// springs in range `[spring_color_offsets[k], spring_color_offsets[k+1])`.
    /// Length `spring_color_count + 1`. Empty when `spring_count == 0`.
    pub spring_color_offsets: Buffer<u32>,
    /// Spring indices sorted by colour. Length `spring_count`. The
    /// GPU kernel dispatches `spring_color_count` passes; pass `k`
    /// reads slot `i` and looks up `spring = spring_indices_by_color[offset_k + i]`.
    pub spring_indices_by_color: Buffer<u32>,

    // ── VBD adjacency + per-vertex coloring ─────────────────────────
    // Computed at build time alongside the spring colouring above.
    // Used by the Vertex Block Descent solver
    // ([`crate::solvers::VbdSolver`]); the existing XPBD kernels
    // ignore these fields entirely. Unlike the spring colouring
    // (where two springs share a colour iff they don't share a
    // particle), VBD colours **particles** so two same-colour
    // particles never share a constraint — the per-vertex block
    // solve then becomes embarrassingly parallel within a colour.
    /// Per-vertex CSR offsets into `vertex_adjacent_springs` (length
    /// `particle_count + 1`). Vertex `v` owns spring slots
    /// `[vertex_adjacent_springs_offsets[v] .. v+1]`.
    pub vertex_adjacent_springs_offsets: Buffer<u32>,
    /// Flat list of spring indices keyed by vertex.
    pub vertex_adjacent_springs: Buffer<u32>,
    /// Number of particle colours.
    pub particle_color_count: usize,
    /// CSR offsets over `particle_indices_by_color` — colour `k`
    /// owns particles in `[particle_color_offsets[k], k+1])`. Length
    /// `particle_color_count + 1`.
    pub particle_color_offsets: Buffer<u32>,
    /// Particle indices sorted by colour. Length `particle_count`.
    pub particle_indices_by_color: Buffer<u32>,

    // ── Triangle topology (cloth surfaces) ──────────────────────────
    /// Number of triangles. Used for rendering today; bending /
    /// area-preservation constraints will read this in stage 12.x.
    pub tri_count: usize,
    /// Flat `tri_count * 3` array of particle indices (CCW winding).
    pub tri_indices: Buffer<u32>,

    // ── Cable joints (Stage 20.1) ───────────────────────────────────
    // A cable is a polyline of CABLE particles linked by per-segment
    // distance constraints. The springs that enforce inextensibility
    // are already in the spring arrays above (added by `add_cable` at
    // build time); these arrays just record the polyline grouping so
    // the viewer/tests/Dahl friction kernel can address "the i-th
    // cable" as one logical unit.
    /// Number of cable polylines.
    pub cable_count: usize,
    /// CSR offsets into `cable_particles` — cable `c` owns particles
    /// in range `[cable_particle_offsets[c], cable_particle_offsets[c+1])`.
    /// Length `cable_count + 1`.
    pub cable_particle_offsets: Buffer<u32>,
    /// Flat list of particle indices grouped by cable. Length is the
    /// sum of per-cable particle counts.
    pub cable_particles: Buffer<u32>,
    /// Per-cable rest length of the whole polyline (m). Computed at
    /// build time as the sum of segment rest lengths.
    pub cable_rest_length: Buffer<Real>,
    /// Per-cable stretch stiffness (N/m). The builder converts to the
    /// per-segment XPBD compliance `α = 1/k`. Stored here so the
    /// solver / viewer can introspect cable tuning.
    pub cable_stiffness: Buffer<Real>,
    /// Per-cable bending stiffness `k_b` (N·m). Drives the discrete
    /// hinge energy `E_b = ½ · k_b · (2·sin(θ/2))²` between adjacent
    /// segments — Stage 20.2. 0 disables bending for that cable
    /// (it then behaves as a pure CABLE polyline as in Stage 20.1).
    pub cable_bending_stiffness: Buffer<Real>,

    // ── Per-vertex cable bending CSR (Stage 20.2) ──────────────────
    // Each entry is a 4-u32 record `(v_prev, v_mid, v_next, role)`
    // packed flat. Role ∈ {0, 1, 2} encodes whether this particle
    // is `v_prev`, `v_mid`, or `v_next` — letting the kernel pick
    // the right Hessian + force sign without a branch on equality.
    // Bending stiffness lives at index `cable_bending_k[entry_idx]`
    // (one f32 per CSR entry; pre-expanded so the kernel needn't
    // walk the cable to find its stiffness).
    /// Per-vertex CSR offsets into `cable_bending_entries` /
    /// `cable_bending_k`. Length `particle_count + 1`. Each step is
    /// `4` (one bending entry contributes 4 u32s); `cable_bending_k`
    /// has `(offsets[v+1] - offsets[v]) / 4` entries per vertex.
    pub vertex_cable_bending_offsets: Buffer<u32>,
    /// Flat `(v_prev, v_mid, v_next, role)` 4-u32 records.
    pub vertex_cable_bending_entries: Buffer<u32>,
    /// Per-entry bending stiffness `k_b` (N·m). Length =
    /// `vertex_cable_bending_entries.len() / 4`.
    pub vertex_cable_bending_k: Buffer<Real>,

    // ── Tetrahedra (Stage 20.2 volumetric soft bodies) ──────────────
    // A tetrahedron-soup soft body shares the particle pool with
    // springs / cables; per-tet adjacency is CSR-encoded and consumed
    // by the VBD per-vertex block solve to fold a Neo-Hookean
    // (corotational) energy contribution into the local Hessian +
    // gradient. Mirrors `cable_*` in shape — one TetMesh = one
    // contiguous slice of `tet_indices` with shared material params.
    /// Number of tetrahedra in the model. The kernel iterates over
    /// `tet_count` per particle via the `vertex_adjacent_tets` CSR
    /// (similar to `vertex_adjacent_springs` for cloth).
    pub tet_count: usize,
    /// Flat `tet_count * 4` array of particle indices (one tet =
    /// 4 vertex ids). Per-tet ordering matches `add_tet_mesh`'s input.
    pub tet_indices: Buffer<u32>,
    /// Per-tet inverse rest-state basis `Dm⁻¹` (3×3 column-major).
    /// `Dm = [v1-v0 | v2-v0 | v3-v0]` evaluated at rest; the
    /// deformation gradient is `F = Ds · Dm⁻¹`. Pre-inverted at
    /// build time to keep the kernel hot loop branch-free. Flattened
    /// to `Mat3` so the existing `Buffer<Mat3>` projection can be
    /// reused.
    pub tet_dm_inv: Buffer<Mat3>,
    /// Per-tet rest volume (m³). Used to weight the per-tet Hessian
    /// + force contribution: `Ψ_total = rest_volume · Ψ_density`. We
    ///   compute this once at build time as `det(Dm) / 6`; pre-storing
    ///   avoids the determinant in the hot loop.
    pub tet_rest_volume: Buffer<Real>,
    /// Per-tet first Lamé parameter `μ` (shear modulus, N/m²). The
    /// builder converts the user's `(E, ν)` (Young's modulus + Poisson
    /// ratio) into Lamé via the standard formulas. One value per tet
    /// so multi-material meshes (Stage 20.3) can mix soft + stiff
    /// elements within a single `TetMeshId`.
    pub tet_mu: Buffer<Real>,
    /// Per-tet second Lamé parameter `λ` (N/m²).
    pub tet_lambda: Buffer<Real>,
    /// Per-tet shear and bulk viscosity on total Green strain (Pa s).
    pub tet_viscosity: Buffer<crate::TetViscosity>,

    // ── Per-tet plasticity (Stage 20.3) ─────────────────────────────
    // Multiplicative split `F = F_e · F_p` with isotropic von Mises
    // return mapping. `tet_F_plastic` stores `F_p` (3×3, identity at
    // build time); the elastic Neo-Hookean energy is evaluated on
    // `F_e = F · F_p⁻¹`. After each substep a return-mapping pass
    // updates `F_p` so the trial elastic state is pulled back to the
    // yield surface; `tet_plastic_strain` accumulates the equivalent
    // plastic strain (used by linear hardening: `σ_y(ε_p) = σ_y0 + H · ε_p`).
    /// Per-tet plastic deformation gradient `F_p` (3×3 column-major).
    /// Initialised to identity; mutated by the return-mapping pass.
    pub tet_f_plastic: Buffer<Mat3>,
    /// Per-tet accumulated equivalent plastic strain (scalar). Starts
    /// at zero; used by the linear-hardening rule.
    pub tet_plastic_strain: Buffer<Real>,
    /// Per-tet von Mises yield stress `σ_y0` (Pa). When the elastic
    /// trial deviatoric stress norm exceeds `σ_y(ε_p)`, the return
    /// mapping pulls the trial state back to the yield surface.
    /// Set to `Real::INFINITY` (or just a very large number) to keep
    /// a tet purely elastic. Default 1e30 = effectively infinite.
    pub tet_yield_stress: Buffer<Real>,
    /// Per-tet linear hardening modulus `H` (Pa). Yield stress grows
    /// as `σ_y(ε_p) = σ_y0 + H · ε_p`. `H = 0` is perfectly plastic.
    pub tet_hardening: Buffer<Real>,
    /// Per-tet kinematic-hardening (Ziegler / Prager) back-stress
    /// modulus `c` (Pa) — Stage 20.4. The yield function shifts from
    /// `‖dev(τ)‖ - √(2/3)·σ_y` to `‖dev(τ) - α‖ - √(2/3)·σ_y` where
    /// `α` is the per-tet symmetric back-stress tensor (in `State`).
    /// The Prager evolution rule for α is `Δα = c · (2/3) · n̂ · Δγ`
    /// where `n̂ = (dev(τ) - α) / ‖dev(τ) - α‖` is the deviator-aligned
    /// flow direction. `c = 0` reduces to pure isotropic hardening
    /// (back-compat with Stage 20.3); a small positive `c` produces the
    /// canonical Bauschinger effect (lower yield in the reverse direction
    /// after forward yielding) so cyclic loading accumulates `‖α‖`
    /// monotonically rather than driving `F_p` back to identity.
    pub tet_kinematic_hardening: Buffer<Real>,

    /// Number of tet meshes (groupings).
    pub tet_mesh_count: usize,
    /// CSR offsets into `tet_indices` — mesh `m` owns tets in range
    /// `[tet_mesh_offsets[m], tet_mesh_offsets[m+1])`. Length
    /// `tet_mesh_count + 1`.
    pub tet_mesh_offsets: Buffer<u32>,

    // ── Stage 33: cross-mesh soft-vs-soft VT contact ────────────────
    /// Soft-surface registry — one entry per tet mesh (boundary tris)
    /// or cloth mesh (tessellation tris). Each entry references a
    /// contiguous slice of `soft_surface_tris` and a contiguous range
    /// of `particle_q`. See [`crate::SoftSurface`] for the
    /// per-entry layout.
    pub soft_surfaces: Vec<crate::SoftSurface>,
    /// Flat tri list across every soft surface. Indexed by
    /// `SoftSurface.tri_offset..tri_offset + tri_count`. CPU-only
    /// in v1; the GPU mirror is uploaded by [`Self::upload_to_device`]
    /// into [`Self::soft_surface_tris_gpu`] (Stage 33.2).
    pub soft_surface_tris: Vec<[u32; 3]>,
    /// Stage 33.1 — flat edge list across every soft surface. Each
    /// entry is `[min_vert_id, max_vert_id]` (unique undirected edge);
    /// per-surface entries occupy `SoftSurface.edge_offset .. + edge_count`.
    /// Used by the EE narrow-phase to enumerate candidate edges
    /// without re-deriving them per substep.
    pub soft_surface_edges: Vec<[u32; 2]>,

    /// Stage 33.3 — per-vertex incident-tri set, CSR-encoded.
    /// `soft_vertex_incident_tris[soft_vertex_incident_tris_offsets[v]..v+1]`
    /// gives the **global** triangle indices (into
    /// `soft_surface_tris`) that touch particle `v`. The VT contact
    /// gather uses this for self-collision adjacency filtering: when
    /// the vertex's surface has `self_collision = true`, candidate
    /// tris on the same surface that are in this set are skipped
    /// (they would always be in contact since the vertex is one of
    /// the triangle's corners). Length =
    /// `particle_count + 1`.
    pub soft_vertex_incident_tris_offsets: Vec<u32>,
    /// Stage 33.3 — flat global tri-id payload for the per-vertex
    /// incident-tri CSR. See
    /// [`Self::soft_vertex_incident_tris_offsets`].
    pub soft_vertex_incident_tris: Vec<u32>,
    /// Stage 33.2 — GPU mirror of [`Self::soft_surface_tris`]. Each
    /// triangle is stored as a `vec4<u32>` (`x=v0, y=v1, z=v2,
    /// w=tri_surface_owner`) — the surface owner is folded in so the
    /// VT gather kernel can apply the same-surface filter without a
    /// second indirection. Lazily allocated on the first
    /// [`Self::upload_to_device`] when `soft_surface_tris` is non-empty;
    /// `None` otherwise.
    #[cfg(feature = "gpu")]
    pub soft_surface_tris_gpu: Option<wgpu::Buffer>,
    /// Stage 33.2 — GPU mirror of [`Self::soft_surfaces`]. One
    /// `SoftSurfaceGpu` (32 bytes — kind, vertex_offset, vertex_count,
    /// tri_offset, tri_count, padding) per registered soft surface.
    #[cfg(feature = "gpu")]
    pub soft_surfaces_gpu: Option<wgpu::Buffer>,

    // ── Per-vertex tet adjacency CSR (Stage 20.2) ──────────────────
    // Computed at build time alongside the spring adjacency. Each
    // entry pairs a tet id with the local vertex index `v_order ∈
    // {0, 1, 2, 3}` so the kernel can fetch the right column of
    // `Dm⁻¹` (== the gradient `m` for that vertex). Flat layout:
    //   `vertex_adjacent_tets[2*k]   = tet_id`
    //   `vertex_adjacent_tets[2*k+1] = v_order`
    /// Per-vertex CSR offsets into `vertex_adjacent_tets`. Length
    /// `particle_count + 1`. Each step in the offset is `2` (one
    /// tet contributes a `(tet_id, v_order)` pair).
    pub vertex_adjacent_tets_offsets: Buffer<u32>,
    /// Flat `(tet_id, v_order)` pairs.
    pub vertex_adjacent_tets: Buffer<u32>,

    // ── Rigid bodies ────────────────────────────────────────────────
    /// Number of rigid bodies.
    pub body_count: usize,
    /// Per-body mass (kg).
    pub body_mass: Buffer<Real>,
    /// Per-body inverse mass (1/kg), 0 for static/kinematic.
    pub body_inv_mass: Buffer<Real>,
    /// Per-body inertia tensor at COM, body frame (kg·m²).
    pub body_inertia: Buffer<Mat3>,
    /// Per-body inverse inertia tensor.
    pub body_inv_inertia: Buffer<Mat3>,
    /// Per-body centre of mass in body frame (m).
    pub body_com: Buffer<Vec3>,
    /// Per-body bitflags (see [`super::BodyFlags`]).
    pub body_flags: Buffer<u32>,
    /// Per-body world index.
    pub body_world: Buffer<i32>,
    /// Initial body pose snapshotted at build time. [`super::State::for_model`]
    /// copies this into the working state.
    pub initial_body_q: Buffer<Transform>,

    // ── Shapes ──────────────────────────────────────────────────────
    /// Number of collision shapes.
    pub shape_count: usize,
    /// Owning body index, or -1 for static-world shapes.
    pub shape_body: Buffer<i32>,
    /// **GPU-side shape kind**, parallel to `shape_geometry`. The
    /// `ShapeGeometry` enum is a Rust-side convenience; for the GPU we
    /// flatten it into two parallel arrays — kind discriminant +
    /// packed scalar parameters — so kernels can branch on it.
    /// Values: 0=Sphere · 1=Plane · 2=Box · 3=Capsule · 4=Cylinder ·
    /// 5=Cone · 6=Ellipsoid · 0xff=unsupported-on-gpu (skipped, e.g.
    /// Triangle / TriMesh / ConvexHull / Heightfield / Sdf).
    pub shape_geom_kind: Buffer<u32>,
    /// **GPU-side shape parameters**, parallel to `shape_geometry`.
    /// Per-kind packing:
    /// - Sphere    → `(radius, 0, 0)`
    /// - Plane     → `(nx, ny, nz)` (local-frame normal)
    /// - Box       → `(hx, hy, hz)` (half-extents)
    /// - Capsule   → `(half_height, radius, 0)`
    /// - Cylinder  → `(half_height, radius, 0)`
    /// - Cone      → `(half_height, radius, 0)`
    /// - Ellipsoid → `(ax, ay, az)` (per-axis semi-axes)
    /// - Triangle  → `(0, 0, 0)` (vertices held host-side; see
    ///   `ShapeGeometry::Triangle`)
    /// - TriMesh   → `(mesh_id_as_f32, 0, 0)` (mesh BVH lives on host
    ///   until stage 11 ports it)
    pub shape_geom_data: Buffer<Vec3>,
    /// Shape transform in owning body frame.
    pub shape_transform: Buffer<Transform>,
    /// Per-shape geometry tags (sphere radius, etc).
    pub shape_geometry: Vec<super::ShapeGeometry>,
    /// Per-shape bitflags.
    pub shape_flags: Buffer<u32>,
    /// Per-shape Coulomb friction coefficient, combined using `shape_contact_material` rules.
    pub shape_friction: Buffer<Real>,
    /// CPU penalty-contact parameters and friction combination rules.
    pub shape_contact_material: Vec<super::ContactMaterial>,
    /// Per-shape surface velocity in the owning body frame (world for
    /// static shapes): a belt surface moving over the shape, which CPU
    /// penalty friction drags contacts along.
    pub shape_surface_velocity: Vec<Vec3>,
    /// Per-shape **rolling** friction (m·μ) — opposes the angular
    /// velocity component perpendicular to the contact normal.
    /// Without this, balls/cylinders roll forever on a flat plane.
    /// Pair-rule: `max(μ_a, μ_b)`. `NaN` = `0.0` (no rolling damping).
    pub shape_rolling_friction: Buffer<Real>,
    /// Per-shape **torsional** friction — opposes the angular
    /// velocity component ALONG the contact normal (spinning).
    /// Stops a ball from spinning forever about the contact axis.
    /// Same convention as `shape_rolling_friction`.
    pub shape_torsional_friction: Buffer<Real>,
    /// Per-shape restitution. Same `NaN` sentinel + max-rule as friction.
    pub shape_restitution: Buffer<Real>,
    /// Triangle-mesh storage. `ShapeGeometry::TriMesh { mesh_id }`
    /// references entries here.
    pub meshes: Vec<TriMesh>,
    /// Convex-hull vertex storage. `ShapeGeometry::ConvexHull { hull_id }`
    /// references entries here. Each hull is a flat `Vec<Vec3>` of
    /// vertex positions in the shape's local frame; the GJK support
    /// function scans them linearly.
    pub hulls: Vec<Vec<Vec3>>,
    /// Heightfield storage. `ShapeGeometry::Heightfield { field_id }`
    /// references entries here.
    pub heightfields: Vec<Heightfield>,
    /// SDF shape storage. `ShapeGeometry::Sdf { sdf_id }` references
    /// entries here.
    pub sdfs: Vec<SdfShape>,
    /// Per-shape membership mask, checked against the other shape's filter in both directions.
    pub shape_collision_group: Buffer<u32>,
    /// Per-shape filter mask, checked against the other shape's membership mask.
    pub shape_collision_mask: Buffer<u32>,
    /// Explicit shape-pair exclusions (always filtered out, even
    /// when group bitmasks would allow). Stored as `(min, max)`
    /// shape-index pairs; broad phase tests pair membership against
    /// this set.
    pub shape_collision_filter_pairs: Vec<(u32, u32)>,

    // ── Per-shape visual material (Stage 14 — camera sensors) ──────
    /// Per-shape base RGB colour, in linear space. Default white
    /// `(1, 1, 1)`. Consumed by the GPU camera-trace kernel
    /// ([`crate::sensors::TiledCamera`]); ignored by collision /
    /// dynamics. Length = `shape_count`.
    pub shape_color: Buffer<Vec3>,
    /// Per-shape PBR-style metallic factor in `[0, 1]`. Default `0`.
    /// Reserved for future use by shading paths; the current
    /// camera-trace kernel does Lambertian shading and only reads
    /// `shape_color`. Length = `shape_count`.
    pub shape_metallic: Buffer<Real>,
    /// Per-shape PBR-style roughness factor in `[0, 1]`. Default
    /// `0.5`. Same status as `shape_metallic`. Length = `shape_count`.
    pub shape_roughness: Buffer<Real>,

    // ── Equality constraints (CONNECT / WELD) ──────────────────────
    /// Number of equality constraints.
    pub equality_count: usize,
    /// Per-constraint type — `EqualityType::{Connect, Weld}` as `u32`.
    pub equality_type: Buffer<u32>,
    /// Per-constraint body A index (-1 = world).
    pub equality_body_a: Buffer<i32>,
    /// Per-constraint body B index (-1 = world).
    pub equality_body_b: Buffer<i32>,
    /// Per-constraint anchor in body A's local frame.
    pub equality_anchor_a: Buffer<Vec3>,
    /// Per-constraint anchor in body B's local frame.
    pub equality_anchor_b: Buffer<Vec3>,
    /// Per-constraint relative orientation (used by `Weld`). For
    /// `Connect` it's ignored.
    pub equality_relative_rotation: Buffer<molla_math::Quat>,
    /// Per-constraint enabled flag (1 = active).
    pub equality_enabled: Buffer<u32>,
    /// Per-constraint world index (defaults to body A's world).
    pub equality_constraint_world: Buffer<i32>,
    /// Per-constraint driver joint index — only consulted when
    /// `equality_type[c] == EqualityType::JointMimic` (Stage 32).
    /// `-1` for non-mimic constraints. Length = `equality_count`.
    pub equality_mimic_driver_joint: Buffer<i32>,
    /// Per-constraint driven joint index — see
    /// [`Self::equality_mimic_driver_joint`]. Length = `equality_count`.
    pub equality_mimic_driven_joint: Buffer<i32>,
    /// Per-constraint polynomial coefficients (5 per slot; flat layout):
    ///
    /// ```text
    /// q_driven = poly[0] + poly[1]·q_dr + poly[2]·q_dr²
    ///          + poly[3]·q_dr³ + poly[4]·q_dr⁴
    /// ```
    ///
    /// Only consulted when `equality_type` for the slot equals
    /// `EqualityType::JointMimic`. Length = `equality_count * 5`.
    /// Linear-mimic (gripper) is `[0, 1, 0, 0, 0]` after offset;
    /// linear with bias `c0 + c1·q` is `[c0, c1, 0, 0, 0]`. Stage 32.
    pub equality_mimic_poly: Buffer<Real>,

    // ── Mimic constraints (q_a = c0 + c1 · q_b) ────────────────────
    /// Number of mimic constraints.
    pub mimic_count: usize,
    /// Per-mimic driven joint index (the one whose q is overridden).
    pub mimic_joint_a: Buffer<i32>,
    /// Per-mimic driver joint index (the one whose q is read).
    pub mimic_joint_b: Buffer<i32>,
    /// Per-mimic constant offset.
    pub mimic_c0: Buffer<Real>,
    /// Per-mimic linear coefficient.
    pub mimic_c1: Buffer<Real>,
    /// Per-mimic enabled flag.
    pub mimic_enabled: Buffer<u32>,
    /// Per-mimic world index (defaults to driver joint's world).
    pub mimic_constraint_world: Buffer<i32>,

    // ── Joints ──────────────────────────────────────────────────────
    /// Number of joints.
    pub joint_count: usize,
    /// Joint kind, as a `u32` mirror of [`super::JointType`].
    pub joint_type: Buffer<u32>,
    /// Per-joint parent body index (-1 = world).
    pub joint_parent: Buffer<i32>,
    /// Per-joint child body index.
    pub joint_child: Buffer<i32>,
    /// Joint frame in parent body.
    pub joint_x_p: Buffer<Transform>,
    /// Joint frame in child body.
    pub joint_x_c: Buffer<Transform>,
    /// Per-joint axis in joint-local frame. Unused for fixed/free.
    pub joint_axis: Buffer<Vec3>,
    /// CSR-style offset into `state.joint_q` for joint `i` —
    /// length `joint_count + 1`, last entry = total coord count.
    pub joint_coord_offset: Buffer<u32>,
    /// CSR-style offset into `state.joint_qd` — length `joint_count + 1`.
    pub joint_dof_offset: Buffer<u32>,
    /// Total joint coordinates (== last entry of `joint_coord_offset`).
    pub joint_coord_total: usize,
    /// Total joint DoFs (== last entry of `joint_dof_offset`).
    pub joint_dof_total: usize,
    /// Initial joint coordinates snapshotted at build time —
    /// [`super::State::for_model`] copies these into `joint_q`.
    pub initial_joint_q: Buffer<Real>,
    /// Per-DOF lower position limit (rad or m). `NaN` = unlimited.
    /// Length = `joint_dof_total`.
    pub joint_limit_lower: Buffer<Real>,
    /// Per-DOF upper position limit. `NaN` = unlimited.
    pub joint_limit_upper: Buffer<Real>,
    /// Per-DOF Coulomb-style damping coefficient (N·m·s/rad or N·s/m).
    /// 0 = none. Damping torque each step is `−friction·qd`.
    pub joint_friction: Buffer<Real>,
    /// Per-DOF effort cap (N·m or N). `NaN` = unlimited. Applied
    /// torque (joint_force + drive + friction) is clamped to
    /// `[−effort_limit, +effort_limit]`.
    pub joint_effort_limit: Buffer<Real>,
    /// Per-DOF velocity cap (rad/s or m/s). `NaN` = unlimited.
    /// Post-step `qd` is clamped to `[−velocity_limit, +velocity_limit]`.
    pub joint_velocity_limit: Buffer<Real>,
    pub joint_compliance: Buffer<crate::joint_control::JointCompliance>,

    // ── Articulations ───────────────────────────────────────────────
    /// Number of independent articulations (kinematic chains). The
    /// Featherstone solver dispatches one workgroup invocation per
    /// articulation. Equals `articulation_start.len() − 1`.
    pub articulation_count: usize,
    /// CSR offsets into the joint arrays — length `articulation_count + 1`.
    /// Articulation `a` owns joints `articulation_start[a]..articulation_start[a+1]`.
    pub articulation_start: Buffer<u32>,
    /// Per-articulation world index — defaults to the world of the
    /// first joint's child body. Length = `articulation_count`.
    pub articulation_world: Buffer<i32>,
    /// Per-joint world index. Defaults to the child body's world.
    /// Length = `joint_count`.
    pub joint_world: Buffer<i32>,
    /// Per-joint enable flag (1 = active, 0 = disabled). Disabled
    /// joints don't contribute torque or constrain motion. Length =
    /// `joint_count`.
    pub joint_enabled: Buffer<u32>,
    /// Invert the authored joint map while retaining its coordinate and effort signs.
    pub joint_reversed: Buffer<u32>,
    /// Per-joint ancestor — index of the parent joint in the
    /// articulation tree, or -1 for a root. Length = `joint_count`.
    /// Currently informational; future tree traversals consume it.
    pub joint_ancestor: Buffer<i32>,
    /// Per-DOF stiffness (N·m/rad or N/m). Distinct from
    /// `Control.joint_target_kp` — `target_ke` is a static joint
    /// property used by Featherstone for spring-style joint drives;
    /// `target_kp` is the per-step PD gain. Length = `joint_dof_total`.
    pub joint_target_ke: Buffer<Real>,

    // ── Cable joints (Stage 39E) ────────────────────────────────────
    // Per-joint static parameters consulted only when joint type is
    // [`super::JointType::Cable`]. All four buffers have length
    // `joint_count` and store `0.0` for non-cable joints.
    /// Per-joint linear (stretch) stiffness `k_l` (N/m).
    pub joint_cable_stiffness_linear: Buffer<Real>,
    /// Per-joint isotropic bend stiffness `k_b` (N·m/rad).
    pub joint_cable_stiffness_bend: Buffer<Real>,
    /// Per-joint linear (stretch) damping `c_l` (N·s/m).
    pub joint_cable_damping_linear: Buffer<Real>,
    /// Per-joint isotropic bend damping `c_b` (N·m·s/rad).
    pub joint_cable_damping_bend: Buffer<Real>,

    // ── Muscles (Stage 42 v2) ──────────────────────────────────────
    // Hill-type muscle actuators attached to CABLE joints. Each
    // muscle drives the joint's **linear stretch DoF** with an active
    // contractile force computed from the classic
    //   f = a · f_max · f_L(l/l_opt) · f_V(v/v_max)
    // Hill three-element model. v2 adds (1) co-contraction by storing
    // multiple muscles per joint via a CSR, (2) first-order
    // excitation→activation lag, and (3) Thelen-style series-elastic
    // tendon with pennation. See `actuators::muscle` for the curve
    // definitions.
    //
    // Most arrays are parallel and `muscle_count`-sized. The
    // `joint_muscle_*` pair is a CSR mapping joint → muscle ids:
    //   `joint_muscle_offsets[j] .. joint_muscle_offsets[j+1]`
    // indexes into `joint_muscle_indices`. Length of `offsets` is
    // `joint_count + 1` so the empty-muscle case still has a
    // sentinel.
    /// Number of muscle actuators on this model.
    pub muscle_count: usize,
    /// CSR offsets — `joint_muscle_offsets[j] .. [j+1]` indexes the
    /// list of muscle ids attached to joint `j`. Length =
    /// `joint_count + 1` (single `0` sentinel when no muscles).
    pub joint_muscle_offsets: Buffer<u32>,
    /// CSR muscle indices keyed by `joint_muscle_offsets`. Length =
    /// `muscle_count`.
    pub joint_muscle_indices: Buffer<u32>,
    /// Per-muscle maximum isometric force (N). Length = `muscle_count`.
    pub muscle_f_max: Buffer<Real>,
    /// Per-muscle optimal fibre length (m). Length = `muscle_count`.
    pub muscle_l_opt: Buffer<Real>,
    /// Per-muscle maximum shortening velocity (m/s). Length =
    /// `muscle_count`.
    pub muscle_v_max: Buffer<Real>,
    /// Per-muscle filtered activation, `a ∈ [0, 1]`. Updated each
    /// step by [`crate::actuators::update_muscle_activation`] from
    /// the neural drive in `muscle_excitation`. Length =
    /// `muscle_count`.
    pub muscle_activation: Buffer<Real>,
    /// Per-muscle neural drive / excitation, `u ∈ [0, 1]`. **User
    /// input** — set this each step (constant for tonic muscles).
    /// The first-order filter
    /// `da/dt = (u − a)/τ_act` converts it into the filtered
    /// activation. Length = `muscle_count`.
    pub muscle_excitation: Buffer<Real>,
    /// Per-muscle activation time constant `τ_act` (s). Length =
    /// `muscle_count`. Defaults to 20 ms per Thelen-2003.
    pub muscle_tau_activation: Buffer<Real>,
    /// Per-muscle tendon slack length `l_t_slack` (m). `0.0` ⇒
    /// rigid-tendon mode (v1 behaviour). Length = `muscle_count`.
    pub muscle_l_tendon_slack: Buffer<Real>,
    /// Per-muscle tendon stiffness ratio `E·A / f_max`. Length =
    /// `muscle_count`.
    pub muscle_tendon_stiffness_ratio: Buffer<Real>,
    /// Per-muscle pennation angle `α` (rad). Length = `muscle_count`.
    pub muscle_pennation_angle: Buffer<Real>,
    /// Per-muscle mounting polarity (`+1` agonist, `−1` antagonist).
    /// Antagonist muscles pull the cable longer (`q_l → +`) and
    /// experience reversed length/velocity kinematics inside Hill's
    /// curves. Length = `muscle_count`.
    pub muscle_polarity: Buffer<Real>,

    // ── Worlds ──────────────────────────────────────────────────────
    /// Number of independent worlds (RL/multi-env grouping).
    pub world_count: usize,
    /// Per-world gravity vector (m/s²).
    pub gravity: Buffer<Vec3>,

    // ── Global solver knobs ─────────────────────────────────────────
    /// Angular damping coefficient applied each step (1/s).
    pub angular_damping: Real,
    /// Linear damping coefficient applied to body velocities each step
    /// (1/s). Models air drag / rolling resistance crudely:
    /// `v *= exp(-linear_damping · dt)`. Default 0 — keeps existing
    /// tests bit-identical.
    pub linear_damping: Real,
    /// Velocity cap applied to particles (m/s) for stability.
    pub particle_velocity_max: Real,
    /// Slack added to AABBs in the broad phase (m). Catches contacts
    /// that would otherwise be missed at the boundary of a step.
    pub contact_margin: Real,
    /// Default Coulomb friction coefficient applied to fresh contacts
    /// when shapes don't carry a per-shape material.
    pub default_friction: Real,
    /// Default restitution applied to fresh contacts when shapes
    /// don't carry a per-shape material.
    pub default_restitution: Real,
}

impl Model {
    pub fn combined_shape_friction(&self, a: usize, b: usize) -> Result<Real> {
        let coefficients = self.shape_friction.host()?;
        let ma = self
            .shape_contact_material
            .get(a)
            .ok_or_else(|| molla_core::Error::Build("missing shape contact material".into()))?;
        let mb = self
            .shape_contact_material
            .get(b)
            .ok_or_else(|| molla_core::Error::Build("missing shape contact material".into()))?;
        let fa = coefficients
            .get(a)
            .ok_or_else(|| molla_core::Error::Build("missing shape friction".into()))?;
        let fb = coefficients
            .get(b)
            .ok_or_else(|| molla_core::Error::Build("missing shape friction".into()))?;
        Ok(ma.friction_combine.combine(mb.friction_combine, *fa, *fb))
    }

    /// An empty model: zero particles, zero bodies, zero shapes, one
    /// world with standard gravity.
    #[must_use]
    pub fn empty() -> Self {
        let s = BufferUsage::Storage;
        Self {
            particle_count: 0,
            particle_mass: Buffer::from_vec(Vec::new(), s),
            particle_inv_mass: Buffer::from_vec(Vec::new(), s),
            particle_radius: Buffer::from_vec(Vec::new(), s),
            particle_flags: Buffer::from_vec(Vec::new(), s),
            particle_world: Buffer::from_vec(Vec::new(), s),
            initial_particle_q: Buffer::from_vec(Vec::new(), s),

            spring_count: 0,
            spring_a: Buffer::from_vec(Vec::new(), s),
            spring_b: Buffer::from_vec(Vec::new(), s),
            spring_rest_length: Buffer::from_vec(Vec::new(), s),
            spring_compliance: Buffer::from_vec(Vec::new(), s),
            spring_max_stretch: Buffer::from_vec(Vec::new(), s),
            spring_tension_only: Buffer::from_vec(Vec::new(), s),
            spring_color_count: 0,
            // Single sentinel offset so empty models don't need a
            // separate device-side branch.
            spring_color_offsets: Buffer::from_vec(vec![0u32], s),
            spring_indices_by_color: Buffer::from_vec(Vec::new(), s),

            vertex_adjacent_springs_offsets: Buffer::from_vec(vec![0u32], s),
            vertex_adjacent_springs: Buffer::from_vec(Vec::new(), s),
            particle_color_count: 0,
            particle_color_offsets: Buffer::from_vec(vec![0u32], s),
            particle_indices_by_color: Buffer::from_vec(Vec::new(), s),

            tri_count: 0,
            tri_indices: Buffer::from_vec(Vec::new(), s),

            cable_count: 0,
            cable_particle_offsets: Buffer::from_vec(vec![0u32], s),
            cable_particles: Buffer::from_vec(Vec::new(), s),
            cable_rest_length: Buffer::from_vec(Vec::new(), s),
            cable_stiffness: Buffer::from_vec(Vec::new(), s),
            cable_bending_stiffness: Buffer::from_vec(Vec::new(), s),
            vertex_cable_bending_offsets: Buffer::from_vec(vec![0u32], s),
            vertex_cable_bending_entries: Buffer::from_vec(Vec::new(), s),
            vertex_cable_bending_k: Buffer::from_vec(Vec::new(), s),

            tet_count: 0,
            tet_indices: Buffer::from_vec(Vec::new(), s),
            tet_dm_inv: Buffer::from_vec(Vec::new(), s),
            tet_rest_volume: Buffer::from_vec(Vec::new(), s),
            tet_mu: Buffer::from_vec(Vec::new(), s),
            tet_lambda: Buffer::from_vec(Vec::new(), s),
            tet_viscosity: Buffer::from_vec(Vec::new(), s),
            tet_f_plastic: Buffer::from_vec(Vec::new(), s),
            tet_plastic_strain: Buffer::from_vec(Vec::new(), s),
            tet_yield_stress: Buffer::from_vec(Vec::new(), s),
            tet_hardening: Buffer::from_vec(Vec::new(), s),
            tet_kinematic_hardening: Buffer::from_vec(Vec::new(), s),
            tet_mesh_count: 0,
            tet_mesh_offsets: Buffer::from_vec(vec![0u32], s),
            soft_surfaces: Vec::new(),
            soft_surface_tris: Vec::new(),
            // Stage 33.3 — empty CSR (single sentinel offset). When no
            // soft surfaces exist, the per-vertex incident set is
            // empty for every particle.
            soft_vertex_incident_tris_offsets: vec![0u32],
            soft_vertex_incident_tris: Vec::new(),
            // Stage 33.1 — empty edge list when no soft surfaces.
            soft_surface_edges: Vec::new(),
            #[cfg(feature = "gpu")]
            soft_surface_tris_gpu: None,
            #[cfg(feature = "gpu")]
            soft_surfaces_gpu: None,
            vertex_adjacent_tets_offsets: Buffer::from_vec(vec![0u32], s),
            vertex_adjacent_tets: Buffer::from_vec(Vec::new(), s),

            body_count: 0,
            body_mass: Buffer::from_vec(Vec::new(), s),
            body_inv_mass: Buffer::from_vec(Vec::new(), s),
            body_inertia: Buffer::from_vec(Vec::new(), s),
            body_inv_inertia: Buffer::from_vec(Vec::new(), s),
            body_com: Buffer::from_vec(Vec::new(), s),
            body_flags: Buffer::from_vec(Vec::new(), s),
            body_world: Buffer::from_vec(Vec::new(), s),
            initial_body_q: Buffer::from_vec(Vec::new(), s),

            shape_count: 0,
            shape_body: Buffer::from_vec(Vec::new(), s),
            shape_geom_kind: Buffer::from_vec(Vec::new(), s),
            shape_geom_data: Buffer::from_vec(Vec::new(), s),
            shape_transform: Buffer::from_vec(Vec::new(), s),
            shape_geometry: Vec::new(),
            shape_flags: Buffer::from_vec(Vec::new(), s),
            shape_friction: Buffer::from_vec(Vec::new(), s),
            shape_contact_material: Vec::new(),
            shape_surface_velocity: Vec::new(),
            shape_rolling_friction: Buffer::from_vec(Vec::new(), s),
            shape_torsional_friction: Buffer::from_vec(Vec::new(), s),
            shape_restitution: Buffer::from_vec(Vec::new(), s),
            meshes: Vec::new(),
            hulls: Vec::new(),
            heightfields: Vec::new(),
            sdfs: Vec::new(),
            shape_collision_group: Buffer::from_vec(Vec::new(), s),
            shape_collision_mask: Buffer::from_vec(Vec::new(), s),
            shape_collision_filter_pairs: Vec::new(),
            shape_color: Buffer::from_vec(Vec::new(), s),
            shape_metallic: Buffer::from_vec(Vec::new(), s),
            shape_roughness: Buffer::from_vec(Vec::new(), s),

            equality_count: 0,
            equality_type: Buffer::from_vec(Vec::new(), s),
            equality_body_a: Buffer::from_vec(Vec::new(), s),
            equality_body_b: Buffer::from_vec(Vec::new(), s),
            equality_anchor_a: Buffer::from_vec(Vec::new(), s),
            equality_anchor_b: Buffer::from_vec(Vec::new(), s),
            equality_relative_rotation: Buffer::from_vec(Vec::new(), s),
            equality_enabled: Buffer::from_vec(Vec::new(), s),
            equality_constraint_world: Buffer::from_vec(Vec::new(), s),
            equality_mimic_driver_joint: Buffer::from_vec(Vec::new(), s),
            equality_mimic_driven_joint: Buffer::from_vec(Vec::new(), s),
            equality_mimic_poly: Buffer::from_vec(Vec::new(), s),

            mimic_count: 0,
            mimic_joint_a: Buffer::from_vec(Vec::new(), s),
            mimic_joint_b: Buffer::from_vec(Vec::new(), s),
            mimic_c0: Buffer::from_vec(Vec::new(), s),
            mimic_c1: Buffer::from_vec(Vec::new(), s),
            mimic_enabled: Buffer::from_vec(Vec::new(), s),
            mimic_constraint_world: Buffer::from_vec(Vec::new(), s),

            joint_count: 0,
            joint_type: Buffer::from_vec(Vec::new(), s),
            joint_parent: Buffer::from_vec(Vec::new(), s),
            joint_child: Buffer::from_vec(Vec::new(), s),
            joint_x_p: Buffer::from_vec(Vec::new(), s),
            joint_x_c: Buffer::from_vec(Vec::new(), s),
            joint_axis: Buffer::from_vec(Vec::new(), s),
            joint_coord_offset: Buffer::from_vec(vec![0u32], s),
            joint_dof_offset: Buffer::from_vec(vec![0u32], s),
            joint_coord_total: 0,
            joint_dof_total: 0,
            initial_joint_q: Buffer::from_vec(Vec::new(), s),
            joint_limit_lower: Buffer::from_vec(Vec::new(), s),
            joint_limit_upper: Buffer::from_vec(Vec::new(), s),
            joint_friction: Buffer::from_vec(Vec::new(), s),
            joint_effort_limit: Buffer::from_vec(Vec::new(), s),
            joint_velocity_limit: Buffer::from_vec(Vec::new(), s),
            joint_compliance: Buffer::from_vec(Vec::new(), s),

            articulation_count: 0,
            articulation_start: Buffer::from_vec(vec![0u32], s),
            articulation_world: Buffer::from_vec(Vec::new(), s),
            joint_world: Buffer::from_vec(Vec::new(), s),
            joint_enabled: Buffer::from_vec(Vec::new(), s),
            joint_reversed: Buffer::from_vec(Vec::new(), s),
            joint_ancestor: Buffer::from_vec(Vec::new(), s),
            joint_target_ke: Buffer::from_vec(Vec::new(), s),
            joint_cable_stiffness_linear: Buffer::from_vec(Vec::new(), s),
            joint_cable_stiffness_bend: Buffer::from_vec(Vec::new(), s),
            joint_cable_damping_linear: Buffer::from_vec(Vec::new(), s),
            joint_cable_damping_bend: Buffer::from_vec(Vec::new(), s),

            muscle_count: 0,
            // Stage 42 v2 — CSR offsets need a single 0 sentinel even
            // when no muscles are attached so consumers can safely
            // index `offsets[j+1] - offsets[j]`.
            joint_muscle_offsets: Buffer::from_vec(vec![0u32], s),
            joint_muscle_indices: Buffer::from_vec(Vec::new(), s),
            muscle_f_max: Buffer::from_vec(Vec::new(), s),
            muscle_l_opt: Buffer::from_vec(Vec::new(), s),
            muscle_v_max: Buffer::from_vec(Vec::new(), s),
            muscle_activation: Buffer::from_vec(Vec::new(), s),
            muscle_excitation: Buffer::from_vec(Vec::new(), s),
            muscle_tau_activation: Buffer::from_vec(Vec::new(), s),
            muscle_l_tendon_slack: Buffer::from_vec(Vec::new(), s),
            muscle_tendon_stiffness_ratio: Buffer::from_vec(Vec::new(), s),
            muscle_pennation_angle: Buffer::from_vec(Vec::new(), s),
            muscle_polarity: Buffer::from_vec(Vec::new(), s),

            world_count: 1,
            gravity: Buffer::from_vec(vec![Vec3::new(0.0, -molla_math::GRAVITY, 0.0)], s),

            angular_damping: 0.0,
            linear_damping: 0.0,
            particle_velocity_max: 1.0e6,
            contact_margin: 0.01,
            default_friction: 0.5,
            default_restitution: 0.0,
        }
    }

    /// Allocate a fresh [`super::State`] sized for this model.
    pub fn state(&self) -> Result<super::State> {
        super::State::for_model(self)
    }

    /// Stage 33.3 — toggle the per-surface "self-collision enabled"
    /// flag at runtime. The VT contact gather consults this on every
    /// substep, so toggling it on/off mid-simulation is safe (no
    /// rebuilt registries).
    ///
    /// `surface_id` is the 0-based index into [`Self::soft_surfaces`].
    /// Out-of-range indices are silently ignored — the typical caller
    /// (viewer toggle, programmatic control) shouldn't have to handle
    /// the "no soft surface" case explicitly.
    pub fn set_soft_surface_self_collision(&mut self, surface_id: usize, enable: bool) {
        if let Some(s) = self.soft_surfaces.get_mut(surface_id) {
            s.self_collision = enable;
        }
    }

    /// Allocate a fresh [`super::Control`] sized for this model.
    #[must_use]
    pub fn control(&self) -> super::Control {
        super::Control::for_model(self)
    }

    /// Validate the immutable per-tet viscosity array before simulation or upload.
    pub fn validate_tet_viscosity(&self) -> Result<()> {
        if self.tet_viscosity.len() != self.tet_count {
            return Err(Error::Build(
                "tet viscosity length must equal tet count".into(),
            ));
        }
        for material in self.tet_viscosity.host()? {
            material.validate()?;
        }
        Ok(())
    }

    /// Allocate device buffers for every model array and upload host data.
    ///
    /// The model is read-only at simulation time, so this only needs to
    /// run once before the first GPU step. Re-running is harmless but
    /// re-uploads the same data.
    #[cfg(feature = "gpu")]
    pub fn upload_to_device<D: molla_storage::BufferDevice + ?Sized>(
        &mut self,
        device: &D,
        queue: &wgpu::Queue,
    ) -> Result<()> {
        self.validate_tet_viscosity()?;
        macro_rules! up {
            ($field:ident) => {{
                self.$field.ensure_device(device)?;
                self.$field.upload(queue)?;
            }};
        }
        up!(particle_mass);
        up!(particle_inv_mass);
        up!(particle_radius);
        up!(particle_flags);
        up!(particle_world);
        up!(initial_particle_q);
        up!(spring_a);
        up!(spring_b);
        up!(spring_rest_length);
        up!(spring_compliance);
        up!(spring_max_stretch);
        up!(spring_tension_only);
        up!(spring_color_offsets);
        up!(spring_indices_by_color);
        up!(vertex_adjacent_springs_offsets);
        up!(vertex_adjacent_springs);
        up!(particle_color_offsets);
        up!(particle_indices_by_color);
        up!(tri_indices);
        up!(cable_particle_offsets);
        up!(cable_particles);
        up!(cable_rest_length);
        up!(cable_stiffness);
        up!(cable_bending_stiffness);
        up!(vertex_cable_bending_offsets);
        up!(vertex_cable_bending_entries);
        up!(vertex_cable_bending_k);
        up!(tet_indices);
        up!(tet_dm_inv);
        up!(tet_rest_volume);
        up!(tet_mu);
        up!(tet_lambda);
        up!(tet_viscosity);
        up!(tet_f_plastic);
        up!(tet_plastic_strain);
        up!(tet_yield_stress);
        up!(tet_hardening);
        up!(tet_kinematic_hardening);
        up!(tet_mesh_offsets);
        up!(vertex_adjacent_tets_offsets);
        up!(vertex_adjacent_tets);
        up!(body_mass);
        up!(body_inv_mass);
        up!(body_inertia);
        up!(body_inv_inertia);
        up!(body_com);
        up!(body_flags);
        up!(body_world);
        up!(initial_body_q);
        up!(shape_body);
        up!(shape_geom_kind);
        up!(shape_geom_data);
        up!(shape_transform);
        up!(shape_flags);
        up!(joint_type);
        up!(joint_parent);
        up!(joint_child);
        up!(joint_x_p);
        up!(joint_x_c);
        up!(joint_axis);
        up!(joint_coord_offset);
        up!(joint_dof_offset);
        up!(initial_joint_q);
        up!(joint_limit_lower);
        up!(joint_limit_upper);
        up!(joint_friction);
        up!(joint_effort_limit);
        up!(joint_velocity_limit);
        up!(joint_compliance);
        up!(articulation_start);
        up!(articulation_world);
        up!(joint_world);
        up!(joint_enabled);
        up!(joint_reversed);
        up!(joint_ancestor);
        up!(joint_target_ke);
        up!(shape_collision_group);
        up!(shape_color);
        up!(shape_metallic);
        up!(shape_roughness);
        // Equality + mimic constraint arrays — added when the GPU
        // `XpbdEqualityKernel` replaced the host-side shim.
        up!(equality_type);
        up!(equality_body_a);
        up!(equality_body_b);
        up!(equality_anchor_a);
        up!(equality_anchor_b);
        up!(equality_relative_rotation);
        up!(equality_enabled);
        // Joint-mimic (Stage 32): per-equality driver/driven joint
        // indices + 5-coeff polynomial. Only consulted by the GPU
        // `project_joint_mimic` entry point when the slot's
        // `equality_type == JointMimic`. The joint coord/dof offsets
        // were already uploaded above as part of the joint section.
        up!(equality_mimic_driver_joint);
        up!(equality_mimic_driven_joint);
        up!(equality_mimic_poly);
        // Stage 39E — cable spring/damper coefficients. Always
        // uploaded so the Featherstone GPU shader's bind-group layout
        // is uniform across scenes (buffers are zeros for non-cable
        // joints).
        if self.joint_count > 0 {
            up!(joint_cable_stiffness_linear);
            up!(joint_cable_stiffness_bend);
            up!(joint_cable_damping_linear);
            up!(joint_cable_damping_bend);
        }
        // Stage 42 v2 — muscle actuators. Only uploaded when at least
        // one muscle exists so empty-model paths skip the no-op. The
        // CSR offsets array has a sentinel even at zero count, so we
        // skip it here too to avoid uploading a single-element buffer
        // pointlessly.
        if self.muscle_count > 0 {
            up!(joint_muscle_offsets);
            up!(joint_muscle_indices);
            up!(muscle_f_max);
            up!(muscle_l_opt);
            up!(muscle_v_max);
            up!(muscle_activation);
            up!(muscle_excitation);
            up!(muscle_tau_activation);
            up!(muscle_l_tendon_slack);
            up!(muscle_tendon_stiffness_ratio);
            up!(muscle_pennation_angle);
            up!(muscle_polarity);
        }
        up!(gravity);

        // ── Stage 33.2 — soft surface registry GPU mirror ─────────
        // Triangles are uploaded as `vec4<u32>` (x=v0, y=v1, z=v2,
        // w=tri_surface_owner) so the VT gather kernel can apply the
        // same-surface filter. Self-collision (Stage 33.3) is a CPU-
        // only feature in v1; the GPU shader applies the cross-mesh
        // surface-owner test only and skips per-vertex adjacency.
        // Each surface is uploaded as a 32-byte `SoftSurfaceGpu`
        // (kind, vertex_offset, vertex_count, tri_offset, tri_count,
        // friction, restitution, _pad).
        if !self.soft_surface_tris.is_empty() && !self.soft_surfaces.is_empty() {
            let mut tri_owner: Vec<u32> = vec![0u32; self.soft_surface_tris.len()];
            for (s_idx, s) in self.soft_surfaces.iter().enumerate() {
                let lo = s.tri_offset as usize;
                let hi = lo + s.tri_count as usize;
                for slot in tri_owner.iter_mut().take(hi).skip(lo) {
                    *slot = s_idx as u32;
                }
            }
            let mut tri_data: Vec<[u32; 4]> = Vec::with_capacity(self.soft_surface_tris.len());
            for (i, t) in self.soft_surface_tris.iter().enumerate() {
                tri_data.push([t[0], t[1], t[2], tri_owner[i]]);
            }
            let tris_buf = device.device().create_buffer(&wgpu::BufferDescriptor {
                label: Some("molla::soft_surface_tris"),
                size: (tri_data.len() * core::mem::size_of::<[u32; 4]>()) as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            queue.write_buffer(&tris_buf, 0, bytemuck::cast_slice(&tri_data));
            self.soft_surface_tris_gpu = Some(tris_buf);

            let mut surf_data: Vec<SoftSurfaceGpu> = Vec::with_capacity(self.soft_surfaces.len());
            for s in &self.soft_surfaces {
                use crate::SoftSurfaceKind;
                let kind = match s.kind {
                    SoftSurfaceKind::TetMesh => 0u32,
                    SoftSurfaceKind::Cloth => 1u32,
                };
                surf_data.push(SoftSurfaceGpu {
                    kind,
                    vertex_offset: s.vertex_offset,
                    vertex_count: s.vertex_count,
                    tri_offset: s.tri_offset,
                    tri_count: s.tri_count,
                    friction: s.friction.max(0.0) as f32,
                    restitution: s.restitution.clamp(0.0, 1.0) as f32,
                    _pad: 0u32,
                });
            }
            let surf_buf = device.device().create_buffer(&wgpu::BufferDescriptor {
                label: Some("molla::soft_surfaces"),
                size: (surf_data.len() * core::mem::size_of::<SoftSurfaceGpu>()) as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            queue.write_buffer(&surf_buf, 0, bytemuck::cast_slice(&surf_data));
            self.soft_surfaces_gpu = Some(surf_buf);
        }

        Ok(())
    }
}

/// GPU-side mirror of [`crate::SoftSurface`] — Stage 33.2.
/// 32 bytes (8 × u32) so it aligns cleanly into a WGSL storage buffer
/// and matches the `SoftSurfaceGpu` struct in
/// `shaders/soft_vt_contact.wgsl`.
///
/// Stage 33.2.1 — added `friction` + `restitution` (bit-cast `f32` into
/// the trailing `u32` slots so the struct layout stays 32 bytes total).
#[cfg(feature = "gpu")]
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SoftSurfaceGpu {
    /// `0 = TetMesh`, `1 = Cloth`. Matches
    /// [`crate::SoftSurfaceKind`].
    pub kind: u32,
    pub vertex_offset: u32,
    pub vertex_count: u32,
    pub tri_offset: u32,
    pub tri_count: u32,
    /// Stage 33.2.1 — Coulomb friction coefficient at contacts on this
    /// surface. Two contacting surfaces' values combine in the kernel
    /// via the geometric mean (`μ = sqrt(μ_a · μ_b)`).
    pub friction: f32,
    /// Stage 33.2.1 — restitution coefficient (0 = perfectly inelastic,
    /// 1 = full velocity reversal). Two contacting surfaces' values
    /// combine via `min(e_a, e_b)` in the kernel velocity-recovery pass.
    pub restitution: f32,
    /// Reserved padding to keep the struct 32 B on both Rust + WGSL.
    /// Earmarked for the GPU port of Stage 33.3 self-collision
    /// (would carry the per-surface `self_collision` flag); CPU
    /// already supports it via `SoftSurface::self_collision`.
    pub _pad: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_model_has_one_world_with_gravity() {
        let m = Model::empty();
        assert_eq!(m.body_count, 0);
        assert_eq!(m.particle_count, 0);
        assert_eq!(m.world_count, 1);
        let g = m.gravity.host().unwrap();
        assert_eq!(g.len(), 1);
        assert!((g[0].y + molla_math::GRAVITY).abs() < 1e-12);
    }
}
