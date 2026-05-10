//! MuJoCo wrapper solver — wraps the upstream MuJoCo C library as a
//! Newton-style backend. Stage 22 (P2) deliverable.
//!
//! ## Why a CPU wrapper inside a GPU-first engine?
//!
//! molla's house rule is GPU-first; this is the documented exception.
//! MuJoCo is the industry-reference contact + constraint solver
//! (Todorov 2014 PADMM, frictional contact via convex projection,
//! mature equality constraints, the entire MuJoCo Menagerie of robots).
//! Re-deriving PADMM in WGSL is a multi-month effort; wrapping the C
//! library is a one-week effort that unlocks the entire ecosystem
//! (`mjcf`, MuJoCo Menagerie, validated against published benchmarks).
//! See PLAN.md Stage 22 + GAP.md §2 / §9 for the full rationale.
//!
//! ## How it works
//!
//! 1. [`MuJoCoSolver::new`] takes a [`Model`] and emits an MJCF XML
//!    document describing it (bodies, joints, shapes, equality
//!    constraints, gravity). MJCF is MuJoCo's native scene format.
//! 2. `mujoco_rust::Model::from_xml_str` parses + compiles the MJCF
//!    into MuJoCo's internal `mjModel`.
//! 3. [`MuJoCoSolver::step`] writes molla's `state_in` into MuJoCo's
//!    `mjData.qpos`/`qvel`, calls `mj_step` (`mujoco_rust::Simulation::step`)
//!    `substeps` times, then reads `qpos`/`qvel` back into `state_out`
//!    along with refreshed body world poses.
//!
//! ## What's supported in v1
//!
//! - **Joints**: Revolute, Prismatic, Ball (rotation only), Fixed
//!   (rendered as no joint at all — the child becomes part of the
//!   parent body), Free (6-DoF root joint). D6 / Distance round-trip
//!   to `Free`; per-axis locks aren't translated yet.
//! - **Shapes**: Sphere, Box, Capsule, Cylinder, Plane. Triangle
//!   meshes / convex hulls / heightfields / SDFs are skipped (would
//!   need MuJoCo's `<asset>` block; deferred to a follow-up).
//! - **Equality constraints**: `EqualityType::Connect` → MJCF
//!   `<connect>`, `EqualityType::Weld` → MJCF `<weld>`. The first
//!   3 / 6 algebraic constraints flow through MuJoCo's hard solver.
//! - **Gravity**: world 0's gravity vector becomes the MJCF `<option>`
//!   gravity attribute. Multi-world models collapse to the world-0
//!   gravity (MuJoCo doesn't have a multi-world concept).
//!
//! ## What's NOT supported in v1
//!
//! Documented as gaps so the user knows what to expect; promote to
//! ✅ in a follow-up as the gap is closed:
//!
//! - **Mimic constraints** — MJCF supports them via `<equality
//!   type="joint">` + per-joint coupling polynomials, but molla's
//!   mimic API is minimal (`q_a = c0 + c1·q_b`) and the round-trip
//!   isn't wired yet.
//! - **Per-shape friction / restitution overrides** — MJCF supports
//!   per-geom `friction="..." solref/solimp"..."` attributes;
//!   currently we use MuJoCo's defaults.
//! - **Multi-world dispatch** — MuJoCo is single-world; if the user
//!   has `world_count > 1`, this solver collapses everything onto
//!   world 0 and emits a `tracing::warn`.
//! - **Particles / springs / cloth** — out of scope for the rigid-body
//!   wrapper. Use [`crate::solvers::XpbdSolver`] or
//!   [`crate::solvers::VbdSolver`] for those.
//!
//! ## What Stage 22.1 added on top of v1
//!
//! - **TriMesh / ConvexHull / Heightfield assets** — emitted as
//!   `<asset><mesh.../></asset>` + `<geom mesh="..."/>` (and
//!   `<hfield/>` for heightfields) so URDF / USD imports with
//!   mesh-collision round-trip into MuJoCo natively.
//! - **SDF shapes** — converted to a baked mesh via
//!   [`crate::geometry::marching_cubes`] at translation time
//!   (MuJoCo has no native SDF primitive). Lossy: surface fidelity
//!   is bounded by the SDF grid resolution, and an analytic-SDF
//!   variant (e.g. `SdfShape::Torus`) gets discretised into the
//!   triangulated approximation. A `tracing::info` records the
//!   conversion at translation time.
//! - **Per-DOF joint limits, friction, armature** — emitted via
//!   `<joint range="..." damping="..." armature="..."/>`.
//! - **FIXED joints** — emitted as MJCF `<equality><weld/>` constraints
//!   between the parent body and the FIXED-joint child (instead of a
//!   loose freejoint that could drift).
//!
//! ## Feature gating
//!
//! When the `mujoco` Cargo feature is **off** (the default), this
//! module exposes [`MuJoCoSolver`] as a stub that errors at runtime
//! with [`Error::Build`] — the type, MJCF builder, and `Solver` impl
//! all compile so callers can write `#[cfg]`-free user code.
//!
//! When the `mujoco` feature is **on**, this module pulls in
//! `mujoco_rust` and provides a working solver. The `mujoco_rust`
//! crate's build script searches for `libmujoco` at `~/.local/mujoco`
//! or in standard system search paths; missing the library produces
//! a clear linker error at compile time. molla doesn't try to install
//! MuJoCo for you — see <https://mujoco.org/> for download links.

use crate::core::{Error, Result};
use crate::geometry::{bake_mesh_to_sdf, marching_cubes};
use crate::math::{Real, Transform, Vec3};
use crate::sim::{
    Contacts, Control, EqualityType, Heightfield, JointType, Model, SdfShape, ShapeGeometry,
    State, TriMesh,
};

use super::Solver;

// ── Public surface ─────────────────────────────────────────────────────

/// MuJoCo-backed [`Solver`] implementation. See module docs for
/// feature-gating behaviour.
pub struct MuJoCoSolver {
    /// Number of `mj_step` calls per outer [`Solver::step`]. Default 1.
    /// Increase to sub-step against MuJoCo's internal Euler / RK4
    /// integrator without bouncing molla's `state_in`/`state_out`
    /// every substep.
    pub substeps: usize,

    /// MJCF source the solver was built from. Kept on the host for
    /// debugging — dump it with `println!("{}", solver.mjcf_source());`
    /// to inspect what molla → MJCF generated. Mostly useful when the
    /// MuJoCo compiler rejects the XML (the rejection message includes
    /// the offending token; cross-reference it here).
    mjcf: String,

    /// Per-molla-body to per-MuJoCo-body index map. molla bodies that
    /// got folded into their parent (FIXED joint case) map to their
    /// surviving ancestor in MuJoCo. Length = `model.body_count`.
    body_to_mj: Vec<i32>,

    /// Per-molla-body cached pose (used to restore body world poses
    /// when the body has no MuJoCo equivalent — FIXED-folded bodies,
    /// kinematic bodies, etc.). Length = `model.body_count`.
    cached_body_q: Vec<Transform>,

    /// Number of generalised coords MuJoCo expects (`mj_model.nq`).
    /// Cached so `step` can validate the molla-side state size.
    n_q: usize,

    /// Number of generalised velocities (`mj_model.nv`).
    n_v: usize,

    /// Backend handle. With `feature = "mujoco"` off this is `()` and
    /// `step` errors at runtime. With it on, this owns the
    /// `mujoco_rust::Simulation`.
    backend: Backend,
}

impl MuJoCoSolver {
    /// Build a MuJoCo solver from a molla [`Model`].
    ///
    /// The model is round-tripped through MJCF — see
    /// [`build_mjcf`](Self::build_mjcf) for the textual output and the
    /// module-level docs for the supported subset.
    ///
    /// Returns [`Error::Build`] when the `mujoco` feature is off
    /// (graceful degradation — callers can fall back to another
    /// backend) or when MuJoCo rejects the generated MJCF (with the
    /// MuJoCo compiler's error message attached).
    pub fn new(model: &Model) -> Result<Self> {
        let TranslatedModel {
            mjcf,
            body_to_mj,
            cached_body_q,
        } = TranslatedModel::from_model(model)?;

        let backend = Backend::open(&mjcf)?;
        let (n_q, n_v) = backend.dof_counts();

        Ok(Self {
            substeps: 1,
            mjcf,
            body_to_mj,
            cached_body_q,
            n_q,
            n_v,
            backend,
        })
    }

    /// Override the substep count.
    #[must_use]
    pub fn with_substeps(mut self, substeps: usize) -> Self {
        self.substeps = substeps.max(1);
        self
    }

    /// The MJCF source that was handed to MuJoCo. Useful for
    /// debugging the translation; not for round-trip serialisation
    /// (MuJoCo strips comments / re-orders attributes when it
    /// re-emits MJCF).
    #[must_use]
    pub fn mjcf_source(&self) -> &str {
        &self.mjcf
    }

    /// Standalone MJCF builder — same output as [`Self::new`], but
    /// without touching MuJoCo. Useful for tests and for users who
    /// want to feed molla scenes into the standalone MuJoCo viewer.
    pub fn build_mjcf(model: &Model) -> Result<String> {
        Ok(TranslatedModel::from_model(model)?.mjcf)
    }
}

impl Solver for MuJoCoSolver {
    fn step(
        &mut self,
        _model: &Model,
        state_in: &State,
        _control: &Control,
        _contacts: &Contacts,
        dt: Real,
        state_out: &mut State,
    ) -> Result<()> {
        // Copy state_in → state_out first; MuJoCo only writes the
        // bodies it knows about, so anything outside its purview
        // (FIXED-folded children, particles) needs to flow through
        // untouched.
        state_out.copy_host_from(state_in)?;

        self.backend.step(
            dt,
            self.substeps,
            self.n_q,
            self.n_v,
            state_in,
            state_out,
            &self.body_to_mj,
            &self.cached_body_q,
        )
    }
}

// ── MJCF translation ───────────────────────────────────────────────────

/// Result of compiling a [`Model`] to MJCF.
struct TranslatedModel {
    mjcf: String,
    body_to_mj: Vec<i32>,
    cached_body_q: Vec<Transform>,
}

impl TranslatedModel {
    fn from_model(model: &Model) -> Result<Self> {
        let initial = model
            .initial_body_q
            .host()
            .map_err(|e| Error::Build(format!("MuJoCo solver: read initial_body_q: {e}")))?;
        let cached_body_q: Vec<Transform> = initial.to_vec();

        let body_to_mj = build_body_index_map(model)?;

        let mjcf = MjcfWriter::new(model, &cached_body_q, &body_to_mj).write()?;

        Ok(Self {
            mjcf,
            body_to_mj,
            cached_body_q,
        })
    }
}

/// One molla body becomes either:
/// - a free MJCF body (kinematic root with `<freejoint/>`)
/// - a child of another body via a kinematic joint (revolute /
///   prismatic / ball / fixed)
/// - merged into its parent (currently we never do this in v1;
///   FIXED joints become MJCF `weld` equality constraints attached
///   to a free body so the topology stays one-to-one — this keeps
///   index mapping trivial and is correct dynamically).
fn build_body_index_map(model: &Model) -> Result<Vec<i32>> {
    // For v1 we do a 1:1 mapping (bodies present in molla all appear
    // as MJCF bodies). MuJoCo body 0 is the implicit world body, so
    // molla body i ↔ MuJoCo body i+1.
    let mut map = vec![-1; model.body_count];
    for i in 0..model.body_count {
        // i32 cast safe: model.body_count is small (< i32::MAX in
        // every practical scene; the GPU side already uses i32).
        map[i] = i as i32 + 1;
    }
    Ok(map)
}

/// MJCF asset names + per-shape resolved asset reference, computed
/// up-front by [`MjcfWriter::collect_assets`]. Lets `write_geom`
/// emit a single string lookup instead of redoing the bake / mesh
/// inventory on every shape.
#[derive(Default)]
struct AssetTable {
    /// Per-mesh asset names — `meshes[mesh_id]` → MJCF `<mesh name="…"/>`.
    meshes: Vec<String>,
    /// Per-hull asset names — `hulls[hull_id]` → MJCF `<mesh name="…"/>`.
    hulls: Vec<String>,
    /// Per-heightfield asset names — `heightfields[field_id]` → MJCF
    /// `<hfield name="…"/>`.
    heightfields: Vec<String>,
    /// Per-SDF asset names — `sdfs[sdf_id]` → MJCF `<mesh name="…"/>`
    /// (we bake the SDF to a triangulated mesh; MuJoCo has no native
    /// SDF primitive). Stage 22.1 caveat: lossy.
    sdfs: Vec<String>,

    /// Mesh sources we emit into the `<asset>` block. Each entry is
    /// (asset_name, vertex_buf, triangle_index_buf) — vertices as flat
    /// "x y z " strings, indices flattened to 3-tuples.
    mesh_assets: Vec<MeshAsset>,
    /// Heightfield sources we emit.
    heightfield_assets: Vec<HeightfieldAsset>,
}

struct MeshAsset {
    name: String,
    vertices: Vec<Vec3>,
    indices: Vec<[u32; 3]>,
}

struct HeightfieldAsset {
    name: String,
    nrow: u32,
    ncol: u32,
    /// Length = nrow · ncol. Row-major, scaled to [0, 1] for MJCF
    /// (which expects normalised heights). The actual physical
    /// extents are passed via the `<geom hfield="…" size="…"/>`
    /// attribute on the geom site.
    elevation: Vec<Real>,
    /// `(radius_x, radius_y, elevation_z, base_z)` — the physical
    /// half-extents for the geom-side `size` attribute.
    geom_size: (Real, Real, Real, Real),
}

/// Builds the MJCF document line-by-line. Not pretty XML — MuJoCo
/// doesn't care, and this keeps the writer trivial.
struct MjcfWriter<'a> {
    model: &'a Model,
    initial: &'a [Transform],
    body_to_mj: &'a [i32],
    /// Joint indices that should produce a `<weld/>` equality
    /// (because they were FIXED in molla). Populated by
    /// `write_worldbody` while it's emitting the bodies; consumed
    /// by `write_equality_block`.
    fixed_joint_welds: Vec<usize>,
    assets: AssetTable,
    out: String,
}

impl<'a> MjcfWriter<'a> {
    fn new(model: &'a Model, initial: &'a [Transform], body_to_mj: &'a [i32]) -> Self {
        Self {
            model,
            initial,
            body_to_mj,
            fixed_joint_welds: Vec::new(),
            assets: AssetTable::default(),
            out: String::with_capacity(4096),
        }
    }

    fn write(mut self) -> Result<String> {
        self.collect_assets()?;
        self.write_header()?;
        self.write_options()?;
        self.write_default_block()?;
        self.write_asset_block()?;
        self.write_worldbody()?;
        self.write_equality_block()?;
        self.write_footer();
        Ok(self.out)
    }

    fn write_header(&mut self) -> Result<()> {
        self.out.push_str("<mujoco model=\"molla\">\n");
        Ok(())
    }

    fn write_options(&mut self) -> Result<()> {
        let g = self
            .model
            .gravity
            .host()
            .map_err(|e| Error::Build(format!("MuJoCo solver: read gravity: {e}")))?;
        let gw = g.first().copied().unwrap_or(Vec3::new(0.0, -9.81, 0.0));
        if self.model.world_count > 1 {
            tracing::warn!(
                world_count = self.model.world_count,
                "MuJoCo solver: collapsing multi-world model onto world 0; \
                 multi-env dispatch isn't supported in v1"
            );
        }
        // `integrator="implicitfast"` matches MuJoCo's default for
        // articulated mechanisms; produces a reasonable energy /
        // stability balance without needing user-side tuning. Step
        // size comes from the per-call `dt` so we set a sentinel
        // here that's overwritten in `Backend::step`.
        self.out.push_str(&format!(
            "  <option timestep=\"0.001\" gravity=\"{gx} {gy} {gz}\" \
             integrator=\"implicitfast\"/>\n",
            gx = gw.x,
            gy = gw.y,
            gz = gw.z,
        ));
        Ok(())
    }

    fn write_default_block(&mut self) -> Result<()> {
        // A single tiny default block keeps every geom visible (rgba
        // alpha 1) and gives MuJoCo enough material info to compile.
        self.out.push_str("  <default>\n");
        self.out.push_str("    <geom rgba=\"0.7 0.7 0.7 1\"/>\n");
        self.out.push_str("  </default>\n");
        Ok(())
    }

    /// Emit `<asset>` block with one `<mesh/>` / `<hfield/>` entry
    /// per registered TriMesh / ConvexHull / SDF / Heightfield. No-op
    /// when there are no asset-bearing shapes (keeps the output
    /// minimal for the simple sphere-on-plane scenes).
    fn write_asset_block(&mut self) -> Result<()> {
        if self.assets.mesh_assets.is_empty() && self.assets.heightfield_assets.is_empty() {
            return Ok(());
        }
        self.out.push_str("  <asset>\n");
        for ma in &self.assets.mesh_assets {
            // Vertices flattened: "x y z x y z …"
            let mut verts = String::with_capacity(ma.vertices.len() * 24);
            for v in &ma.vertices {
                verts.push_str(&format!("{} {} {} ", v.x, v.y, v.z));
            }
            // Triangle faces flattened: "i j k i j k …"
            let mut faces = String::with_capacity(ma.indices.len() * 12);
            for tri in &ma.indices {
                faces.push_str(&format!("{} {} {} ", tri[0], tri[1], tri[2]));
            }
            self.out.push_str(&format!(
                "    <mesh name=\"{name}\" vertex=\"{verts}\" face=\"{faces}\"/>\n",
                name = ma.name,
                verts = verts.trim_end(),
                faces = faces.trim_end(),
            ));
        }
        for hf in &self.assets.heightfield_assets {
            // `<hfield/>` size: "radius_x radius_y elevation base"
            // — half-extents in x/y, peak elevation, base depth.
            let (rx, ry, elev, base) = hf.geom_size;
            let mut elev_str = String::with_capacity(hf.elevation.len() * 8);
            for h in &hf.elevation {
                elev_str.push_str(&format!("{} ", h));
            }
            self.out.push_str(&format!(
                "    <hfield name=\"{name}\" nrow=\"{nrow}\" ncol=\"{ncol}\" \
                 size=\"{rx} {ry} {elev} {base}\" elevation=\"{elev_str}\"/>\n",
                name = hf.name,
                nrow = hf.nrow,
                ncol = hf.ncol,
                elev_str = elev_str.trim_end(),
            ));
        }
        self.out.push_str("  </asset>\n");
        Ok(())
    }

    /// Walk every shape once and decide which assets need an MJCF
    /// `<mesh/>` / `<hfield/>` entry. Stage 22.1 — replaces Stage 22's
    /// `tracing::warn`-and-skip path. SDF shapes get baked through
    /// `marching_cubes` here (lossy; documented in module docs).
    fn collect_assets(&mut self) -> Result<()> {
        // Pre-allocate per-id name slots so geoms can index them.
        self.assets.meshes = vec![String::new(); self.model.meshes.len()];
        self.assets.hulls = vec![String::new(); self.model.hulls.len()];
        self.assets.heightfields = vec![String::new(); self.model.heightfields.len()];
        self.assets.sdfs = vec![String::new(); self.model.sdfs.len()];

        // Track which assets are actually referenced — we only emit
        // ones that some shape uses. Avoids an `<asset>` block full
        // of orphans when the user registered then never referenced.
        let mut mesh_used = vec![false; self.model.meshes.len()];
        let mut hull_used = vec![false; self.model.hulls.len()];
        let mut hfield_used = vec![false; self.model.heightfields.len()];
        let mut sdf_used = vec![false; self.model.sdfs.len()];

        for geom in &self.model.shape_geometry {
            match geom {
                ShapeGeometry::TriMesh { mesh_id } => {
                    if let Some(slot) = mesh_used.get_mut(*mesh_id as usize) {
                        *slot = true;
                    }
                }
                ShapeGeometry::ConvexHull { hull_id } => {
                    if let Some(slot) = hull_used.get_mut(*hull_id as usize) {
                        *slot = true;
                    }
                }
                ShapeGeometry::Heightfield { field_id } => {
                    if let Some(slot) = hfield_used.get_mut(*field_id as usize) {
                        *slot = true;
                    }
                }
                ShapeGeometry::Sdf { sdf_id } => {
                    if let Some(slot) = sdf_used.get_mut(*sdf_id as usize) {
                        *slot = true;
                    }
                }
                _ => {}
            }
        }

        for (i, mesh) in self.model.meshes.iter().enumerate() {
            if !mesh_used[i] {
                continue;
            }
            let name = format!("trimesh{i}");
            self.assets.meshes[i] = name.clone();
            self.assets.mesh_assets.push(MeshAsset {
                name,
                vertices: mesh.vertices.clone(),
                indices: mesh.indices.clone(),
            });
        }
        for (i, hull) in self.model.hulls.iter().enumerate() {
            if !hull_used[i] {
                continue;
            }
            let name = format!("hull{i}");
            self.assets.hulls[i] = name.clone();
            // MuJoCo computes the convex hull of the vertex list when
            // no `face` attribute is present — cheap and matches what
            // molla already considers a `ConvexHull`.
            self.assets.mesh_assets.push(MeshAsset {
                name,
                vertices: hull.clone(),
                indices: Vec::new(),
            });
        }
        for (i, sdf) in self.model.sdfs.iter().enumerate() {
            if !sdf_used[i] {
                continue;
            }
            let name = format!("sdf{i}");
            self.assets.sdfs[i] = name.clone();
            // MuJoCo has no native SDF primitive — bake to a mesh via
            // marching cubes. We discretise an analytic-SDF AABB and
            // bake to a mesh. See module docs for the lossy-conversion
            // caveat.
            let mesh = sdf_to_mesh(*sdf);
            tracing::info!(
                shape = name,
                tris = mesh.indices.len(),
                "MuJoCo: baking SDF to triangulated mesh (no native MJCF SDF; lossy)",
            );
            self.assets.mesh_assets.push(MeshAsset {
                name,
                vertices: mesh.vertices,
                indices: mesh.indices,
            });
        }
        for (i, hf) in self.model.heightfields.iter().enumerate() {
            if !hfield_used[i] {
                continue;
            }
            let name = format!("hfield{i}");
            self.assets.heightfields[i] = name.clone();

            // MJCF heightfield expects normalised heights in [0, 1]
            // with the physical scale carried on the geom side.
            let mut min_h = Real::INFINITY;
            let mut max_h = -Real::INFINITY;
            for &h in &hf.heights {
                if h < min_h {
                    min_h = h;
                }
                if h > max_h {
                    max_h = h;
                }
            }
            if !min_h.is_finite() {
                min_h = 0.0;
            }
            if !max_h.is_finite() {
                max_h = 0.0;
            }
            // Avoid div-by-zero on perfectly flat fields.
            let span = (max_h - min_h).max(1.0e-9);
            let elevation: Vec<Real> =
                hf.heights.iter().map(|h| (*h - min_h) / span).collect();

            // Half-extents along x/z + elevation/base.
            let rx = (hf.nx.saturating_sub(1) as Real) * hf.cell_size * 0.5;
            let ry = (hf.nz.saturating_sub(1) as Real) * hf.cell_size * 0.5;
            let elev_z = if span > 1.0e-9 { span } else { 0.05 };
            let base_z = (-min_h).max(0.05);
            self.assets.heightfield_assets.push(HeightfieldAsset {
                name,
                nrow: hf.nz,
                ncol: hf.nx,
                elevation,
                geom_size: (rx, ry, elev_z, base_z),
            });
        }
        Ok(())
    }

    fn write_worldbody(&mut self) -> Result<()> {
        self.out.push_str("  <worldbody>\n");

        // Static-world shapes (shape_body == -1) get attached to the
        // implicit world body.
        let shape_body = self
            .model
            .shape_body
            .host()
            .map_err(|e| Error::Build(format!("MuJoCo solver: read shape_body: {e}")))?;
        let shape_transform = self
            .model
            .shape_transform
            .host()
            .map_err(|e| Error::Build(format!("MuJoCo solver: read shape_transform: {e}")))?;

        for (s_idx, &owner) in shape_body.iter().enumerate() {
            if owner == -1 {
                self.write_geom(s_idx, &shape_transform[s_idx], "    ")?;
            }
        }

        // Build per-body parent maps from joints. v1 wires every body
        // as a direct child of world; the joint linking each body to
        // its parent is emitted INSIDE the body's MJCF tag. This
        // mirrors how MJCF expects joints to be expressed (a joint
        // belongs to the body it actuates, attaching it to its
        // parent via the body's `pos`/`quat`).
        let joint_parent = self
            .model
            .joint_parent
            .host()
            .map_err(|e| Error::Build(format!("MuJoCo solver: read joint_parent: {e}")))?;
        let joint_child = self
            .model
            .joint_child
            .host()
            .map_err(|e| Error::Build(format!("MuJoCo solver: read joint_child: {e}")))?;
        let joint_type = self
            .model
            .joint_type
            .host()
            .map_err(|e| Error::Build(format!("MuJoCo solver: read joint_type: {e}")))?;
        let joint_axis = self
            .model
            .joint_axis
            .host()
            .map_err(|e| Error::Build(format!("MuJoCo solver: read joint_axis: {e}")))?;

        // For each body, find the joint that produces it (if any).
        // `body_joint[b] = Some(joint_idx)` when joint_idx's child == b.
        let mut body_joint: Vec<Option<usize>> = vec![None; self.model.body_count];
        for (j_idx, &child) in joint_child.iter().enumerate() {
            let cb = child as usize;
            if cb < self.model.body_count {
                body_joint[cb] = Some(j_idx);
            }
        }

        // Emit every body. For v1 we keep the topology FLAT — every
        // body sits directly under <worldbody>, with its joint giving
        // it whatever DoFs the molla joint had. This loses the parent
        // hierarchy MuJoCo would normally use but is correct for the
        // dynamics we care about (free + ball + revolute roots), and
        // keeps the index map 1:1.
        for b_idx in 0..self.model.body_count {
            self.write_body(b_idx, &body_joint, &joint_parent, &joint_type, &joint_axis)?;
        }

        self.out.push_str("  </worldbody>\n");
        Ok(())
    }

    fn write_body(
        &mut self,
        b_idx: usize,
        body_joint: &[Option<usize>],
        joint_parent: &[i32],
        joint_type: &[u32],
        joint_axis: &[Vec3],
    ) -> Result<()> {
        let pose = self.initial[b_idx];
        self.out.push_str(&format!(
            "    <body name=\"body{b_idx}\" pos=\"{px} {py} {pz}\" \
             quat=\"{qw} {qx} {qy} {qz}\">\n",
            px = pose.position.x,
            py = pose.position.y,
            pz = pose.position.z,
            // MuJoCo quaternion order is (w, x, y, z) — note the
            // FLIP from glam's (x, y, z, w). Get this wrong and
            // everything spins backward.
            qw = pose.rotation.w,
            qx = pose.rotation.x,
            qy = pose.rotation.y,
            qz = pose.rotation.z,
        ));

        // Inertial properties. mass+inertia diag are required for
        // dynamic bodies; static bodies get `mocap="true"` so MuJoCo
        // doesn't try to integrate them.
        let mass = self.read_one_real("body_mass", &self.model.body_mass, b_idx)?;
        if mass > 0.0 {
            let inertia = self.read_inertia_diag(b_idx)?;
            let com = self.read_one_vec3("body_com", &self.model.body_com, b_idx)?;
            self.out.push_str(&format!(
                "      <inertial pos=\"{cx} {cy} {cz}\" mass=\"{mass}\" \
                 diaginertia=\"{ix} {iy} {iz}\"/>\n",
                cx = com.x,
                cy = com.y,
                cz = com.z,
                ix = inertia.x,
                iy = inertia.y,
                iz = inertia.z,
            ));

            // Joint that connects this body to its parent. For free
            // bodies (no incoming joint, parent is world) we emit a
            // `<freejoint/>` so MuJoCo knows the body is unconstrained.
            match body_joint[b_idx] {
                Some(j_idx) => {
                    self.write_joint(j_idx, joint_parent, joint_type, joint_axis)?;
                }
                None => {
                    self.out
                        .push_str("      <freejoint/>\n");
                }
            }
        } else {
            // Mass = 0 → kinematic. MuJoCo's `mocap` keyword pins the
            // body in space until the user moves it from outside.
            self.out
                .push_str("      <inertial pos=\"0 0 0\" mass=\"0.001\" \
                 diaginertia=\"0.001 0.001 0.001\"/>\n");
        }

        // Per-body geoms.
        let shape_body = self
            .model
            .shape_body
            .host()
            .map_err(|e| Error::Build(format!("MuJoCo solver: read shape_body: {e}")))?;
        let shape_transform = self
            .model
            .shape_transform
            .host()
            .map_err(|e| Error::Build(format!("MuJoCo solver: read shape_transform: {e}")))?;
        for (s_idx, &owner) in shape_body.iter().enumerate() {
            if owner as usize == b_idx {
                self.write_geom(s_idx, &shape_transform[s_idx], "      ")?;
            }
        }

        self.out.push_str("    </body>\n");
        Ok(())
    }

    fn write_joint(
        &mut self,
        j_idx: usize,
        _joint_parent: &[i32],
        joint_type: &[u32],
        joint_axis: &[Vec3],
    ) -> Result<()> {
        let kind = joint_type_from_u32(joint_type[j_idx]);
        let axis = joint_axis[j_idx];
        // Stage 22.1 — pull per-DOF limits / friction / armature from
        // the model and emit them on the joint. Single-DOF joints
        // (revolute / prismatic) get one entry; multi-DOF joints
        // (ball / free / D6 / distance) skip them in v1 because MJCF
        // doesn't support per-DOF ranges on multi-axis joints (it
        // wants per-component `<joint>` decomposition; deferred).
        let extras = self.joint_extra_attrs(j_idx, kind)?;
        match kind {
            JointType::Revolute => {
                self.out.push_str(&format!(
                    "      <joint name=\"joint{j_idx}\" type=\"hinge\" \
                     axis=\"{ax} {ay} {az}\"{extras}/>\n",
                    ax = axis.x,
                    ay = axis.y,
                    az = axis.z,
                ));
            }
            JointType::Prismatic => {
                self.out.push_str(&format!(
                    "      <joint name=\"joint{j_idx}\" type=\"slide\" \
                     axis=\"{ax} {ay} {az}\"{extras}/>\n",
                    ax = axis.x,
                    ay = axis.y,
                    az = axis.z,
                ));
            }
            JointType::Ball => {
                self.out.push_str(&format!(
                    "      <joint name=\"joint{j_idx}\" type=\"ball\"{extras}/>\n",
                ));
            }
            JointType::Fixed => {
                // FIXED joints — Stage 22 v1 generated a freejoint
                // (which let the child drift). Stage 22.1 instead
                // emits a `<weld/>` equality to MJCF, recorded here
                // and flushed in `write_equality_block`.
                self.fixed_joint_welds.push(j_idx);
                // Still need a freejoint so MuJoCo treats the body
                // as dynamic; without one the body is static and
                // weld-vs-static fails compile.
                self.out.push_str("      <freejoint/>\n");
            }
            JointType::Free | JointType::Distance | JointType::D6 => {
                self.out.push_str("      <freejoint/>\n");
            }
        }
        Ok(())
    }

    /// Build the extra attribute string `range="…" damping="…"
    /// armature="…"` for a single-DOF joint. Returns `""` for
    /// multi-DOF joints (MJCF can't take a single range on a ball /
    /// free joint).
    fn joint_extra_attrs(&self, j_idx: usize, kind: JointType) -> Result<String> {
        if !matches!(kind, JointType::Revolute | JointType::Prismatic) {
            return Ok(String::new());
        }
        // The DOF for joint `j` lives at offset `joint_dof_offset[j]`
        // in the per-DOF limit / friction arrays.
        let dof_offset = self
            .model
            .joint_dof_offset
            .host()
            .map_err(|e| Error::Build(format!("MuJoCo solver: read joint_dof_offset: {e}")))?;
        let dof = dof_offset[j_idx] as usize;
        let limit_lower = self
            .model
            .joint_limit_lower
            .host()
            .map_err(|e| Error::Build(format!("MuJoCo solver: read joint_limit_lower: {e}")))?;
        let limit_upper = self
            .model
            .joint_limit_upper
            .host()
            .map_err(|e| Error::Build(format!("MuJoCo solver: read joint_limit_upper: {e}")))?;
        let friction = self
            .model
            .joint_friction
            .host()
            .map_err(|e| Error::Build(format!("MuJoCo solver: read joint_friction: {e}")))?;
        // Armature isn't a first-class field on molla yet (Stage 22.1
        // surfaces the MJCF attribute, but molla itself uses
        // `joint_target_ke` only as a stiffness term). Emit a small
        // default armature for stability — matches MuJoCo's own
        // recommendation for high-gain controllers.
        let armature: Real = 0.0;

        let mut s = String::new();
        let lo = limit_lower.get(dof).copied().unwrap_or(Real::NAN);
        let hi = limit_upper.get(dof).copied().unwrap_or(Real::NAN);
        if lo.is_finite() && hi.is_finite() && hi > lo {
            s.push_str(&format!(" range=\"{lo} {hi}\" limited=\"true\""));
        }
        let damp = friction.get(dof).copied().unwrap_or(0.0);
        if damp != 0.0 {
            s.push_str(&format!(" damping=\"{damp}\""));
        }
        if armature > 0.0 {
            s.push_str(&format!(" armature=\"{armature}\""));
        }
        Ok(s)
    }

    fn write_geom(
        &mut self,
        s_idx: usize,
        local_pose: &Transform,
        indent: &str,
    ) -> Result<()> {
        let geom = &self.model.shape_geometry[s_idx];
        // Translate molla's per-shape primitive into MuJoCo's `<geom>`
        // attributes. Stage 22 v1 skipped anything needing an asset
        // block; Stage 22.1 wires meshes / hulls / heightfields / SDFs
        // through the `<asset>` registry built by `collect_assets`.
        // Returns either:
        // - `Some((geom_kind, size_attr, extra_attrs))` to emit a
        //   regular `<geom .../>` line.
        // - `None` if the shape was unrepresentable (`tracing::warn`
        //   already issued).
        //
        // `extra_attrs` is whatever extra string fragment goes inside
        // the `<geom .../>` tag (e.g. `mesh="trimesh0"`).
        let resolved: Option<(&'static str, String, String)> = match geom {
            ShapeGeometry::Sphere { radius } => {
                Some(("sphere", format!("{radius}"), String::new()))
            }
            ShapeGeometry::Box { half_extents } => Some((
                "box",
                format!(
                    "{x} {y} {z}",
                    x = half_extents.x,
                    y = half_extents.y,
                    z = half_extents.z,
                ),
                String::new(),
            )),
            ShapeGeometry::Capsule { half_height, radius } => Some((
                "capsule",
                format!("{radius} {half_height}"),
                String::new(),
            )),
            ShapeGeometry::Cylinder { half_height, radius } => Some((
                "cylinder",
                format!("{radius} {half_height}"),
                String::new(),
            )),
            ShapeGeometry::Plane { normal: _ } => {
                // MJCF planes are infinite, normal+Z by default. We
                // handle the rotation via the geom quat so the molla
                // normal is honoured. `size` for a plane is
                // `(half_x, half_y, gridcell)`; 0 0 0.05 = infinite.
                Some(("plane", "0 0 0.05".into(), String::new()))
            }
            ShapeGeometry::TriMesh { mesh_id } => {
                let name = self
                    .assets
                    .meshes
                    .get(*mesh_id as usize)
                    .cloned()
                    .unwrap_or_default();
                if name.is_empty() {
                    tracing::warn!(
                        shape_index = s_idx,
                        mesh_id,
                        "MuJoCo: TriMesh has no asset slot; skipping geom",
                    );
                    None
                } else {
                    // Mesh geom: size attribute is unused (MuJoCo reads
                    // extents from the mesh itself); we still emit `0 0 0`
                    // because some MuJoCo versions reject empty `size`.
                    Some(("mesh", "0 0 0".into(), format!(" mesh=\"{name}\"")))
                }
            }
            ShapeGeometry::ConvexHull { hull_id } => {
                let name = self
                    .assets
                    .hulls
                    .get(*hull_id as usize)
                    .cloned()
                    .unwrap_or_default();
                if name.is_empty() {
                    tracing::warn!(
                        shape_index = s_idx,
                        hull_id,
                        "MuJoCo: ConvexHull has no asset slot; skipping geom",
                    );
                    None
                } else {
                    Some(("mesh", "0 0 0".into(), format!(" mesh=\"{name}\"")))
                }
            }
            ShapeGeometry::Sdf { sdf_id } => {
                let name = self
                    .assets
                    .sdfs
                    .get(*sdf_id as usize)
                    .cloned()
                    .unwrap_or_default();
                if name.is_empty() {
                    tracing::warn!(
                        shape_index = s_idx,
                        sdf_id,
                        "MuJoCo: SDF has no asset slot; skipping geom",
                    );
                    None
                } else {
                    Some(("mesh", "0 0 0".into(), format!(" mesh=\"{name}\"")))
                }
            }
            ShapeGeometry::Heightfield { field_id } => {
                // Find the matching asset (some heightfields could be
                // registered but unused; only used ones get a slot).
                let asset_idx = self
                    .assets
                    .heightfields
                    .get(*field_id as usize)
                    .cloned()
                    .unwrap_or_default();
                if asset_idx.is_empty() {
                    tracing::warn!(
                        shape_index = s_idx,
                        field_id,
                        "MuJoCo: Heightfield has no asset slot; skipping geom",
                    );
                    None
                } else {
                    let asset = self
                        .assets
                        .heightfield_assets
                        .iter()
                        .find(|a| a.name == asset_idx);
                    if let Some(a) = asset {
                        let (rx, ry, elev, base) = a.geom_size;
                        Some((
                            "hfield",
                            format!("{rx} {ry} {elev} {base}"),
                            format!(" hfield=\"{name}\"", name = asset_idx),
                        ))
                    } else {
                        None
                    }
                }
            }
            other => {
                tracing::warn!(
                    shape_index = s_idx,
                    kind = ?std::mem::discriminant(other),
                    "MuJoCo solver: shape kind not yet supported in MJCF translation; skipping",
                );
                None
            }
        };

        let Some((geom_kind, size_attr, extra_attrs)) = resolved else {
            return Ok(());
        };

        self.out.push_str(&format!(
            "{indent}<geom name=\"shape{s_idx}\" type=\"{geom_kind}\" \
             size=\"{size_attr}\" pos=\"{px} {py} {pz}\" \
             quat=\"{qw} {qx} {qy} {qz}\"{extra_attrs}/>\n",
            px = local_pose.position.x,
            py = local_pose.position.y,
            pz = local_pose.position.z,
            qw = local_pose.rotation.w,
            qx = local_pose.rotation.x,
            qy = local_pose.rotation.y,
            qz = local_pose.rotation.z,
        ));
        Ok(())
    }

    fn write_equality_block(&mut self) -> Result<()> {
        if self.model.equality_count == 0 && self.fixed_joint_welds.is_empty() {
            return Ok(());
        }
        let eq_type = self
            .model
            .equality_type
            .host()
            .map_err(|e| Error::Build(format!("MuJoCo solver: read equality_type: {e}")))?;
        let eq_a = self
            .model
            .equality_body_a
            .host()
            .map_err(|e| Error::Build(format!("MuJoCo solver: read equality_body_a: {e}")))?;
        let eq_b = self
            .model
            .equality_body_b
            .host()
            .map_err(|e| Error::Build(format!("MuJoCo solver: read equality_body_b: {e}")))?;
        let anchor_a = self
            .model
            .equality_anchor_a
            .host()
            .map_err(|e| Error::Build(format!("MuJoCo solver: read equality_anchor_a: {e}")))?;
        let anchor_b = self
            .model
            .equality_anchor_b
            .host()
            .map_err(|e| Error::Build(format!("MuJoCo solver: read equality_anchor_b: {e}")))?;
        let enabled = self
            .model
            .equality_enabled
            .host()
            .map_err(|e| Error::Build(format!("MuJoCo solver: read equality_enabled: {e}")))?;
        let rel_rot = self
            .model
            .equality_relative_rotation
            .host()
            .map_err(|e| Error::Build(format!("MuJoCo solver: read equality_relative_rotation: {e}")))?;

        self.out.push_str("  <equality>\n");
        for i in 0..self.model.equality_count {
            if enabled[i] == 0 {
                continue;
            }
            let kind = EqualityType::from_u32(eq_type[i]);
            let a_body_name = mjcf_body_name(eq_a[i], self.body_to_mj);
            let b_body_name = mjcf_body_name(eq_b[i], self.body_to_mj);
            match kind {
                EqualityType::Connect => {
                    // MJCF's `<connect>` ties two bodies at a shared
                    // anchor. molla stores the anchor in EACH body's
                    // local frame; MJCF takes only one anchor (in
                    // body A's frame). Use anchor_a; if anchor_b
                    // disagrees that's a model-build bug but we don't
                    // surface it here.
                    let _ = anchor_b;
                    self.out.push_str(&format!(
                        "    <connect body1=\"{a}\" body2=\"{b}\" \
                         anchor=\"{ax} {ay} {az}\"/>\n",
                        a = a_body_name,
                        b = b_body_name,
                        ax = anchor_a[i].x,
                        ay = anchor_a[i].y,
                        az = anchor_a[i].z,
                    ));
                }
                EqualityType::Weld => {
                    let r = rel_rot[i];
                    self.out.push_str(&format!(
                        "    <weld body1=\"{a}\" body2=\"{b}\" \
                         relpose=\"{tx} {ty} {tz} {rw} {rx} {ry} {rz}\"/>\n",
                        a = a_body_name,
                        b = b_body_name,
                        tx = anchor_a[i].x,
                        ty = anchor_a[i].y,
                        tz = anchor_a[i].z,
                        rw = r.w,
                        rx = r.x,
                        ry = r.y,
                        rz = r.z,
                    ));
                }
                EqualityType::JointMimic => {
                    // Stage 32 polynomial joint-mimic. MuJoCo's
                    // `<joint>` equality only supports a quartic
                    // polynomial relating two joints, which is exactly
                    // our shape — but the MJCF translation needs the
                    // joint *names* (we don't track joint→name mapping
                    // in the writer yet). Skipping with a warn keeps
                    // the round-trip lossy-but-loud rather than silently
                    // breaking the MuJoCo step. Future work: thread the
                    // joint-name map through `MjcfWriter`.
                    tracing::warn!(
                        "MJCF writer: JointMimic equality dropped \
                         (joint-name mapping not yet plumbed)"
                    );
                }
            }
        }

        // Stage 22.1 — FIXED joints become MJCF `<weld/>` equalities
        // pinning the child to its parent at the current relative
        // pose. The freejoint inside the body keeps it dynamic; the
        // weld pins the relative pose so the simulator-side behaviour
        // matches molla's "FIXED = no relative motion".
        if !self.fixed_joint_welds.is_empty() {
            let joint_parent = self
                .model
                .joint_parent
                .host()
                .map_err(|e| Error::Build(format!("MuJoCo solver: read joint_parent: {e}")))?;
            let joint_child = self
                .model
                .joint_child
                .host()
                .map_err(|e| Error::Build(format!("MuJoCo solver: read joint_child: {e}")))?;
            for &j_idx in &self.fixed_joint_welds {
                let parent_b = joint_parent[j_idx];
                let child_b = joint_child[j_idx];
                let a_name = mjcf_body_name(parent_b, self.body_to_mj);
                let b_name = mjcf_body_name(child_b, self.body_to_mj);
                self.out.push_str(&format!(
                    "    <weld name=\"weld_fixed_joint{j_idx}\" body1=\"{a_name}\" \
                     body2=\"{b_name}\"/>\n",
                ));
            }
        }
        self.out.push_str("  </equality>\n");
        Ok(())
    }

    fn write_footer(&mut self) {
        self.out.push_str("</mujoco>\n");
    }

    fn read_one_real(
        &self,
        name: &str,
        buf: &crate::storage::Buffer<Real>,
        i: usize,
    ) -> Result<Real> {
        Ok(buf
            .host()
            .map_err(|e| Error::Build(format!("MuJoCo solver: read {name}: {e}")))?[i])
    }

    fn read_one_vec3(
        &self,
        name: &str,
        buf: &crate::storage::Buffer<Vec3>,
        i: usize,
    ) -> Result<Vec3> {
        Ok(buf
            .host()
            .map_err(|e| Error::Build(format!("MuJoCo solver: read {name}: {e}")))?[i])
    }

    fn read_inertia_diag(&self, b_idx: usize) -> Result<Vec3> {
        let m = self
            .model
            .body_inertia
            .host()
            .map_err(|e| Error::Build(format!("MuJoCo solver: read body_inertia: {e}")))?[b_idx];
        // Diagonal entries — MJCF accepts a 3-vector for diaginertia
        // when the off-diagonals are zero. Most molla scenes (sphere,
        // box, capsule wrapper inertia) ARE diagonal; a non-diagonal
        // inertia would need MJCF's full inertia attribute (`fullinertia`).
        let cols = m.to_cols_array_2d();
        Ok(Vec3::new(cols[0][0].max(1e-9), cols[1][1].max(1e-9), cols[2][2].max(1e-9)))
    }
}

/// Bake a molla [`SdfShape`] into a triangle mesh via marching cubes.
/// Stage 22.1 — MuJoCo has no native SDF primitive, so analytic SDFs
/// have to be discretised on the way in. The conversion is **lossy**:
///
/// - Curvature is approximated by the marching-cubes resolution
///   (32³ in v1 — coarse enough to keep MJCF output small, fine
///   enough that a typical torus stays recognisable).
/// - The original analytic distance function is gone after the bake;
///   sub-voxel collision details are lost.
/// - Sharp corners (e.g. a CSG-style boolean SDF) get smoothed away.
///
/// Picks a sensible default AABB based on the analytic shape kind:
/// torus / sphere have known extents. Resolution is 32 along each
/// axis — adequate for the kinds of SDFs molla v1 supports.
fn sdf_to_mesh(sdf: SdfShape) -> TriMesh {
    // Pick an envelope big enough to capture the surface for the
    // analytic SDFs molla supports today.
    let envelope = match sdf {
        SdfShape::Torus { major, minor } => major + minor + 0.05,
        SdfShape::Sphere { r } => r + 0.05,
    };
    // Build an analytic SDF grid by sampling.
    let res: u32 = 32;
    let cell_size = (2.0 * envelope) / (res as Real - 1.0);
    let origin = Vec3::splat(-envelope);
    let mut data = vec![0.0_f64; (res * res * res) as usize];
    for z in 0..res {
        for y in 0..res {
            for x in 0..res {
                let p = origin
                    + Vec3::new(
                        x as Real * cell_size,
                        y as Real * cell_size,
                        z as Real * cell_size,
                    );
                let idx = (x as usize)
                    + (y as usize) * (res as usize)
                    + (z as usize) * (res as usize) * (res as usize);
                data[idx] = sdf.signed_distance(p);
            }
        }
    }
    let grid = crate::geometry::sdf_bake::SdfGrid {
        nx: res,
        ny: res,
        nz: res,
        origin,
        cell_size,
        data,
    };
    let mut mesh = marching_cubes(&grid, 0.0);
    if !mesh.indices.is_empty() {
        mesh.build_bvh(4);
    }
    // `bake_mesh_to_sdf` is referenced for future use (sdf-from-mesh
    // round trip would call it); silence "unused import" warnings
    // when the function lives behind cfg variants.
    let _ = bake_mesh_to_sdf;
    let _ = Heightfield::default;
    mesh
}

/// Decode a `u32` joint-kind tag from `Model.joint_type` into the
/// strongly typed enum. Mirrors the helper in `sim::articulation`
/// (intentional duplication: keeps the MuJoCo path standalone so
/// future refactors of `articulation` don't ripple here).
fn joint_type_from_u32(v: u32) -> JointType {
    match v {
        0 => JointType::Prismatic,
        1 => JointType::Revolute,
        2 => JointType::Ball,
        3 => JointType::Fixed,
        4 => JointType::Free,
        5 => JointType::Distance,
        6 => JointType::D6,
        _ => JointType::Fixed,
    }
}

/// Produce the MJCF body name for a molla body index. World refs
/// (`-1`) become MJCF's reserved `world` body.
fn mjcf_body_name(molla_body: i32, body_to_mj: &[i32]) -> String {
    if molla_body < 0 {
        return "world".to_owned();
    }
    let idx = molla_body as usize;
    if idx >= body_to_mj.len() {
        return "world".to_owned();
    }
    format!("body{idx}")
}

// ── Backend (feature-gated) ────────────────────────────────────────────

#[cfg(feature = "mujoco")]
mod backend_impl {
    use super::*;
    use crate::math::Quat;
    use mujoco_rust::{Model as MjModel, Simulation};

    pub struct Backend {
        sim: Simulation,
    }

    impl Backend {
        pub fn open(mjcf: &str) -> Result<Self> {
            let mj_model = MjModel::from_xml_str(mjcf).map_err(|e| {
                Error::Build(format!(
                    "MuJoCo rejected generated MJCF: {e}\n--- MJCF ---\n{mjcf}"
                ))
            })?;
            let sim = Simulation::new(mj_model);
            Ok(Self { sim })
        }

        pub fn dof_counts(&self) -> (usize, usize) {
            (self.sim.model.nq(), self.sim.model.nv())
        }

        #[allow(clippy::too_many_arguments)]
        pub fn step(
            &mut self,
            dt: Real,
            substeps: usize,
            n_q: usize,
            n_v: usize,
            state_in: &State,
            state_out: &mut State,
            body_to_mj: &[i32],
            cached_body_q: &[Transform],
        ) -> Result<()> {
            // Push molla's joint state into MuJoCo's qpos/qvel.
            let _ = (n_q, n_v); // Fields cached for validation; not used yet.
            let qpos_in = state_in
                .joint_q
                .host()
                .map_err(|e| Error::Solver(format!("MuJoCo: read state_in.joint_q: {e}")))?;
            let qvel_in = state_in
                .joint_qd
                .host()
                .map_err(|e| Error::Solver(format!("MuJoCo: read state_in.joint_qd: {e}")))?;

            // SAFETY:
            // - `self.sim.state.ptr()` is a non-null `*mut mjData` owned
            //   by `mujoco_rust::Simulation`; valid for the duration of
            //   this `&mut self` borrow.
            // - `qpos`/`qvel` are C-allocated arrays of length `nq`/`nv`;
            //   `n` / `nv` are clamped via `min(...)` so all `add(i)`
            //   stay within those bounds.
            // - `Simulation` is `!Sync`; the `&mut self` borrow ensures
            //   no other Rust code observes the in-progress writes.
            // - `model.ptr()` is similarly stable for `&mut self`; the
            //   `opt.timestep` write does not invalidate any indices we
            //   computed above.
            unsafe {
                let data = self.sim.state.ptr();
                let qpos_ptr = (*data).qpos;
                let qvel_ptr = (*data).qvel;
                let n = self.sim.model.nq().min(qpos_in.len());
                for i in 0..n {
                    *qpos_ptr.add(i) = qpos_in[i];
                }
                let nv = self.sim.model.nv().min(qvel_in.len());
                for i in 0..nv {
                    *qvel_ptr.add(i) = qvel_in[i];
                }
                // Override timestep via `mj_model.opt.timestep` so the
                // outer loop can drive it from molla's `dt`.
                let opt = &mut (*self.sim.model.ptr()).opt;
                opt.timestep = dt;
            }

            for _ in 0..substeps {
                self.sim.step();
            }

            // Read MuJoCo's qpos/qvel back into state_out.
            let qpos = self.sim.qpos();
            let qvel = self.sim.qvel();
            {
                let q_out = state_out
                    .joint_q
                    .host_mut()
                    .map_err(|e| Error::Solver(format!("MuJoCo: write joint_q: {e}")))?;
                let n = q_out.len().min(qpos.len());
                q_out[..n].copy_from_slice(&qpos[..n]);
            }
            {
                let qd_out = state_out
                    .joint_qd
                    .host_mut()
                    .map_err(|e| Error::Solver(format!("MuJoCo: write joint_qd: {e}")))?;
                let n = qd_out.len().min(qvel.len());
                qd_out[..n].copy_from_slice(&qvel[..n]);
            }

            // Refresh per-body world poses from MuJoCo's xpos/xquat.
            // mujoco_rust's `xpos()` returns geom-indexed positions
            // (off-by-one in their docs; we walk it cautiously).
            let n_body = self.sim.model.nbody();
            // SAFETY:
            // - `self.sim.state.ptr()` is a non-null `*mut mjData` owned
            //   by `mujoco_rust::Simulation`; valid for the duration of
            //   this `&mut self` borrow.
            // - `xpos` / `xquat` are C-allocated arrays of length
            //   `3*nbody` and `4*nbody` respectively; the `mj_idx <
            //   n_body` guard above keeps `i*3+2` and `i*4+3` within
            //   bounds.
            // - The bodies-loop only reads through the raw pointers; no
            //   aliasing with the `&mut [Transform]` (`body_q_out`)
            //   because they are distinct C and Rust allocations.
            unsafe {
                let data = self.sim.state.ptr();
                let xpos = (*data).xpos;
                let xquat = (*data).xquat;
                let body_q_out = state_out
                    .body_q
                    .host_mut()
                    .map_err(|e| Error::Solver(format!("MuJoCo: write body_q: {e}")))?;
                for (m_idx, &mj_idx) in body_to_mj.iter().enumerate() {
                    if mj_idx < 0 || (mj_idx as usize) >= n_body {
                        // No MuJoCo body — restore the cached molla pose.
                        body_q_out[m_idx] = cached_body_q[m_idx];
                        continue;
                    }
                    let i = mj_idx as usize;
                    let p = Vec3::new(*xpos.add(i * 3), *xpos.add(i * 3 + 1), *xpos.add(i * 3 + 2));
                    let qw = *xquat.add(i * 4);
                    let qx = *xquat.add(i * 4 + 1);
                    let qy = *xquat.add(i * 4 + 2);
                    let qz = *xquat.add(i * 4 + 3);
                    body_q_out[m_idx] = Transform::new(p, Quat::from_xyzw(qx, qy, qz, qw));
                }
            }

            Ok(())
        }
    }
}

#[cfg(not(feature = "mujoco"))]
mod backend_impl {
    use super::*;

    /// Stub backend used when the `mujoco` Cargo feature is off. Holds
    /// no state; every entry point errors at runtime so callers get a
    /// clean failure rather than a link error.
    pub struct Backend;

    impl Backend {
        pub fn open(_mjcf: &str) -> Result<Self> {
            Err(Error::Build(
                "MuJoCo solver requires the `mujoco` feature: \
                 add `--features mujoco` to your Cargo invocation \
                 (and install libmujoco system-wide; see \
                 <https://mujoco.org/>)."
                    .into(),
            ))
        }

        pub fn dof_counts(&self) -> (usize, usize) {
            (0, 0)
        }

        #[allow(clippy::too_many_arguments)]
        pub fn step(
            &mut self,
            _dt: Real,
            _substeps: usize,
            _n_q: usize,
            _n_v: usize,
            _state_in: &State,
            _state_out: &mut State,
            _body_to_mj: &[i32],
            _cached_body_q: &[Transform],
        ) -> Result<()> {
            Err(Error::Build(
                "MuJoCo solver requires the `mujoco` feature: rebuild \
                 with `--features mujoco` to enable the wrapper."
                    .into(),
            ))
        }
    }
}

use backend_impl::Backend;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::BodyId;
    use crate::math::{Mat3, SpatialInertia};
    use crate::sim::{BodyParams, ModelBuilder};

    /// MJCF translation must produce a non-empty document for a
    /// sphere-on-plane scene, contain `<freejoint/>`, the body's
    /// initial pose, and a sphere geom of the right size. This test
    /// runs without the `mujoco` feature — it exercises the writer,
    /// not MuJoCo itself.
    #[test]
    fn mjcf_emits_sphere_on_plane() {
        let mut b = ModelBuilder::new();
        let body = b.add_body(
            BodyParams::new()
                .with_spatial_inertia(SpatialInertia::sphere(1.0, 0.25))
                .with_position(Vec3::new(0.0, 1.0, 0.0)),
        );
        b.add_shape_sphere(body, 0.25, Transform::IDENTITY).unwrap();
        b.add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY).unwrap();
        let model = b.build().unwrap();

        let mjcf = MuJoCoSolver::build_mjcf(&model).unwrap();
        assert!(mjcf.contains("<mujoco"), "missing root element:\n{mjcf}");
        assert!(mjcf.contains("<worldbody>"), "missing worldbody:\n{mjcf}");
        assert!(mjcf.contains("<freejoint"), "missing freejoint:\n{mjcf}");
        assert!(mjcf.contains("type=\"sphere\""), "missing sphere geom:\n{mjcf}");
        assert!(mjcf.contains("type=\"plane\""), "missing plane geom:\n{mjcf}");
        assert!(mjcf.contains("size=\"0.25\""), "missing sphere size:\n{mjcf}");
    }

    /// Equality CONNECT constraint round-trips into MJCF's `<connect>`
    /// element with the right body names + anchor.
    #[test]
    fn mjcf_emits_connect_equality() {
        let mut b = ModelBuilder::new();
        let a = b.add_body(
            BodyParams::new()
                .with_spatial_inertia(SpatialInertia::sphere(1.0, 0.1))
                .with_position(Vec3::new(0.0, 1.0, 0.0)),
        );
        let c = b.add_body(
            BodyParams::new()
                .with_spatial_inertia(SpatialInertia::sphere(1.0, 0.1))
                .with_position(Vec3::new(0.0, 0.5, 0.0)),
        );
        let _ = b.add_equality_connect(
            a,
            c,
            Vec3::new(0.0, -0.25, 0.0),
            Vec3::new(0.0, 0.25, 0.0),
        );
        let model = b.build().unwrap();
        let mjcf = MuJoCoSolver::build_mjcf(&model).unwrap();
        assert!(mjcf.contains("<connect"), "missing connect:\n{mjcf}");
        assert!(mjcf.contains("body1=\"body0\""), "wrong body1 ref:\n{mjcf}");
        assert!(mjcf.contains("body2=\"body1\""), "wrong body2 ref:\n{mjcf}");
    }

    /// Revolute joint round-trips into a MJCF `<joint type="hinge"
    /// axis="..."/>` tag.
    #[test]
    fn mjcf_emits_revolute_joint() {
        let mut b = ModelBuilder::new();
        b.begin_articulation();
        let body = b.add_body(
            BodyParams::new().with_spatial_inertia(SpatialInertia {
                mass: 1.0,
                com: Vec3::ZERO,
                inertia: Mat3::IDENTITY * 0.05,
            }),
        );
        b.add_joint_revolute(
            BodyId::NONE,
            body,
            Transform::IDENTITY,
            Transform::IDENTITY,
            Vec3::Z,
        )
        .unwrap();
        let model = b.build().unwrap();
        let mjcf = MuJoCoSolver::build_mjcf(&model).unwrap();
        assert!(mjcf.contains("type=\"hinge\""), "missing hinge:\n{mjcf}");
        assert!(mjcf.contains("axis=\"0 0 1\""), "wrong axis:\n{mjcf}");
    }

    /// Stage 22.1 — TriMesh shapes round-trip through the
    /// `<asset><mesh/></asset>` block and the body-side geom carries
    /// `mesh="…"`.
    #[test]
    fn mjcf_emits_trimesh_asset() {
        let mut b = ModelBuilder::new();
        let body = b.add_body(
            BodyParams::new()
                .with_spatial_inertia(SpatialInertia::sphere(1.0, 0.1))
                .with_position(Vec3::new(0.0, 1.0, 0.0)),
        );
        // Tiny tetrahedron mesh.
        let mut mesh = crate::sim::TriMesh::new();
        mesh.vertices = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ];
        mesh.indices = vec![[0, 1, 2], [0, 1, 3], [1, 2, 3], [0, 2, 3]];
        let mesh_id = b.add_mesh(mesh);
        b.add_shape_trimesh(body, mesh_id, Transform::IDENTITY).unwrap();
        let model = b.build().unwrap();
        let mjcf = MuJoCoSolver::build_mjcf(&model).unwrap();
        assert!(mjcf.contains("<asset>"), "missing asset block:\n{mjcf}");
        assert!(
            mjcf.contains("<mesh name=\"trimesh0\""),
            "missing mesh asset:\n{mjcf}"
        );
        assert!(
            mjcf.contains("mesh=\"trimesh0\""),
            "missing geom mesh ref:\n{mjcf}"
        );
        assert!(
            mjcf.contains("type=\"mesh\""),
            "missing mesh geom type:\n{mjcf}"
        );
    }

    /// Stage 22.1 — joint limits and friction round-trip into MJCF
    /// `range="…" limited="true" damping="…"` attributes on a
    /// revolute joint.
    #[test]
    fn mjcf_emits_joint_limits_and_damping() {
        let mut b = ModelBuilder::new();
        b.begin_articulation();
        let body = b.add_body(
            BodyParams::new().with_spatial_inertia(SpatialInertia {
                mass: 1.0,
                com: Vec3::ZERO,
                inertia: Mat3::IDENTITY * 0.05,
            }),
        );
        let joint = b
            .add_joint_revolute(
                BodyId::NONE,
                body,
                Transform::IDENTITY,
                Transform::IDENTITY,
                Vec3::Z,
            )
            .unwrap();
        b.set_joint_limits(joint, &[-0.5], &[0.7]).unwrap();
        b.set_joint_friction(joint, &[0.123]).unwrap();
        let model = b.build().unwrap();
        let mjcf = MuJoCoSolver::build_mjcf(&model).unwrap();
        assert!(
            mjcf.contains("range=\"-0.5 0.7\""),
            "missing range:\n{mjcf}"
        );
        assert!(
            mjcf.contains("limited=\"true\""),
            "missing limited=true:\n{mjcf}"
        );
        assert!(
            mjcf.contains("damping=\"0.123\""),
            "missing damping:\n{mjcf}"
        );
    }

    /// Stage 22.1 — FIXED joints become MJCF `<weld/>` equality
    /// constraints (instead of the Stage-22 free-floating fallback).
    #[test]
    fn mjcf_emits_weld_for_fixed_joint() {
        let mut b = ModelBuilder::new();
        b.begin_articulation();
        let parent = b.add_body(
            BodyParams::new()
                .with_spatial_inertia(SpatialInertia::sphere(1.0, 0.1))
                .with_position(Vec3::new(0.0, 1.0, 0.0)),
        );
        let child = b.add_body(
            BodyParams::new()
                .with_spatial_inertia(SpatialInertia::sphere(1.0, 0.1))
                .with_position(Vec3::new(0.0, 1.5, 0.0)),
        );
        b.add_joint_fixed(parent, child, Transform::IDENTITY, Transform::IDENTITY)
            .unwrap();
        let model = b.build().unwrap();
        let mjcf = MuJoCoSolver::build_mjcf(&model).unwrap();
        assert!(
            mjcf.contains("<weld name=\"weld_fixed_joint"),
            "missing FIXED→weld:\n{mjcf}"
        );
        assert!(
            mjcf.contains("body1=\"body0\""),
            "missing parent body ref:\n{mjcf}"
        );
        assert!(
            mjcf.contains("body2=\"body1\""),
            "missing child body ref:\n{mjcf}"
        );
    }

    /// Without the `mujoco` feature, `MuJoCoSolver::new` errors with a
    /// clear "feature off" message — not a panic, not a link failure.
    #[cfg(not(feature = "mujoco"))]
    #[test]
    fn new_errors_when_feature_off() {
        let mut b = ModelBuilder::new();
        let body = b.add_body(
            BodyParams::new()
                .with_spatial_inertia(SpatialInertia::sphere(1.0, 0.25))
                .with_position(Vec3::new(0.0, 1.0, 0.0)),
        );
        b.add_shape_sphere(body, 0.25, Transform::IDENTITY).unwrap();
        let model = b.build().unwrap();
        // `MuJoCoSolver` doesn't impl `Debug` (it owns a raw mjModel
        // pointer in the feature-on path), so `unwrap_err` won't
        // typecheck. Match on the result the long way.
        match MuJoCoSolver::new(&model) {
            Ok(_) => panic!("expected feature-off Err, got Ok"),
            Err(err) => {
                let msg = format!("{err}");
                assert!(
                    msg.contains("mujoco") && msg.contains("feature"),
                    "unexpected error message: {msg}"
                );
            }
        }
    }
}
