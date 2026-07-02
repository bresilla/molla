//! V0 — terrain query abstraction.
//!
//! A tire model is a force element: each step it queries the ground
//! under the wheel rather than running collision detection. [`Terrain`]
//! is that query — height, surface normal, and friction at a world
//! `(x, z)`. Flat, regular-grid heightfield, triangle-mesh, and simple SDF
//! terrains are provided; an MPM/SPH-surface terrain arrives with V5.

/// A drivable ground surface.
pub trait Terrain {
    /// Ground height (world `y`) at `(x, z)`.
    fn height(&self, x: f64, z: f64) -> f64;
    /// Unit surface normal at `(x, z)`.
    fn normal(&self, x: f64, z: f64) -> [f64; 3];
    /// Coulomb friction coefficient of the surface.
    fn friction(&self) -> f64;
    /// Coulomb friction coefficient at a specific world `(x, z)`.
    ///
    /// Defaults to the terrain-wide scalar for compatibility, but
    /// heightfields and future terrain backends can override it for ice
    /// patches, wet bands, dirt/asphalt transitions, etc.
    fn friction_at(&self, _x: f64, _z: f64) -> f64 {
        self.friction()
    }
    /// Local surface velocity at `(x, z)` in world coordinates.
    ///
    /// Static terrain returns zero; conveyors, deformable soil flow, or moving
    /// terrain patches override this so tire slip is relative to the ground.
    fn surface_velocity(&self, _x: f64, _z: f64) -> [f64; 3] {
        [0.0; 3]
    }
    /// Water/film depth at `(x,z)` in metres.
    ///
    /// Defaults to dry terrain. Heightfields can override this for spatial
    /// wet patches without collapsing the tire model into a single global
    /// friction coefficient.
    fn water_film_depth(&self, _x: f64, _z: f64) -> f64 {
        0.0
    }
    /// Signed distance and closest-surface normal for a world point.
    ///
    /// Positive means outside/above the terrain solid; negative means inside
    /// terrain and should be projected along the returned normal. Heightfield
    /// style terrain defaults to the existing vertical height/normal query,
    /// while SDF terrain overrides this with true closest-surface distance.
    fn signed_distance_and_normal(&self, p: [f64; 3]) -> (f64, [f64; 3]) {
        let normal = self.normal(p[0], p[2]);
        let surface_y = self.height(p[0], p[2]);
        // The heightfield's closest-surface proxy point sits vertically
        // below/above `p` at `(p.x, surface_y, p.z)`, so the offset
        // `p − q` is purely vertical and `(p − q)·n` reduces to the
        // `dy · n_y` term.
        ((p[1] - surface_y) * normal[1], normal)
    }
    /// Contact manifold for a spherical tread/contact sample.
    ///
    /// Most terrains collapse this to one local signed-distance patch.
    /// Triangle meshes override it with a real multi-triangle manifold so tire
    /// contact near seams/curbs/plates does not collapse to one closest point.
    fn contact_manifold(
        &self,
        center: [f64; 3],
        sphere_radius: f64,
        contact_margin: f64,
        max_contacts: usize,
    ) -> Vec<TriangleMeshContactPatch> {
        if !sphere_radius.is_finite() || sphere_radius < 0.0 || max_contacts == 0 {
            return Vec::new();
        }
        let (signed_distance, normal) = self.signed_distance_and_normal(center);
        if signed_distance > contact_margin.max(0.0) {
            return Vec::new();
        }
        let n = normalize3(normal);
        vec![TriangleMeshContactPatch {
            triangle_id: usize::MAX,
            point: [
                center[0] - n[0] * signed_distance,
                center[1] - n[1] * signed_distance,
                center[2] - n[2] * signed_distance,
            ],
            normal: n,
            signed_distance,
            penetration: (-signed_distance).max(0.0),
            friction: self.friction_at(center[0], center[2]).max(0.0),
        }]
    }
}

/// An infinite flat plane at a fixed height.
#[derive(Clone, Copy, Debug)]
pub struct FlatTerrain {
    /// World `y` of the plane.
    pub height: f64,
    /// Surface friction coefficient.
    pub friction: f64,
}

impl FlatTerrain {
    /// A flat plane at `height` with friction `friction`.
    pub fn new(height: f64, friction: f64) -> Self {
        Self { height, friction }
    }
}

impl Terrain for FlatTerrain {
    fn height(&self, _x: f64, _z: f64) -> f64 {
        self.height
    }
    fn normal(&self, _x: f64, _z: f64) -> [f64; 3] {
        [0.0, 1.0, 0.0]
    }
    fn friction(&self) -> f64 {
        self.friction
    }
}

/// A regular-grid heightfield. Heights are sampled on an `nx × nz`
/// grid; `height` bilinearly interpolates and `normal` is taken from
/// the finite-difference gradient of the interpolated surface.
#[derive(Clone, Debug)]
pub struct HeightfieldTerrain {
    /// World `(x, z)` of sample `(0, 0)`.
    pub origin: [f64; 2],
    /// Grid spacing.
    pub spacing: f64,
    /// Samples per row (`x`).
    pub nx: usize,
    /// Number of rows (`z`).
    pub nz: usize,
    /// Row-major heights, `heights[iz * nx + ix]`.
    pub heights: Vec<f64>,
    /// Surface friction coefficient.
    pub friction: f64,
    /// Optional row-major friction samples. When present, sampled like height.
    pub friction_map: Option<Vec<f64>>,
    /// Optional row-major water/film depth samples in metres.
    pub water_film_map: Option<Vec<f64>>,
    /// Optional row-major surface velocity samples, in world m/s.
    pub surface_velocity_map: Option<Vec<[f64; 3]>>,
}

impl HeightfieldTerrain {
    /// Build a heightfield. `heights.len()` must equal `nx * nz`.
    pub fn new(
        origin: [f64; 2],
        spacing: f64,
        nx: usize,
        nz: usize,
        heights: Vec<f64>,
        friction: f64,
    ) -> Self {
        assert_eq!(heights.len(), nx * nz, "heights length must be nx * nz");
        Self {
            origin,
            spacing,
            nx,
            nz,
            heights,
            friction,
            friction_map: None,
            water_film_map: None,
            surface_velocity_map: None,
        }
    }

    /// Build a heightfield with a per-cell/per-sample friction map. The map is
    /// row-major and bilinearly interpolated at tire contact points.
    pub fn with_friction_map(
        origin: [f64; 2],
        spacing: f64,
        nx: usize,
        nz: usize,
        heights: Vec<f64>,
        friction: f64,
        friction_map: Vec<f64>,
    ) -> Self {
        assert_eq!(heights.len(), nx * nz, "heights length must be nx * nz");
        assert_eq!(
            friction_map.len(),
            nx * nz,
            "friction map length must be nx * nz"
        );
        Self {
            origin,
            spacing,
            nx,
            nz,
            heights,
            friction,
            friction_map: Some(friction_map),
            water_film_map: None,
            surface_velocity_map: None,
        }
    }

    /// Attach a row-major water-film map, in metres, sampled like height.
    pub fn with_water_film_map(mut self, water_film_map: Vec<f64>) -> Self {
        assert_eq!(
            water_film_map.len(),
            self.nx * self.nz,
            "water film map length must be nx * nz"
        );
        self.water_film_map = Some(water_film_map);
        self
    }

    /// Attach a row-major local surface-velocity map, in world m/s, sampled
    /// like height/friction/film. This is the CPU terrain counterpart to the
    /// GPU FEM `surface_velocity_map`: moving belts, flowing soil, or DEM/MPM
    /// terrain exports can vary relative ground motion across the patch
    /// without a scene-local slip multiplier.
    pub fn with_surface_velocity_map(mut self, surface_velocity_map: Vec<[f64; 3]>) -> Self {
        assert_eq!(
            surface_velocity_map.len(),
            self.nx * self.nz,
            "surface velocity map length must be nx * nz"
        );
        self.surface_velocity_map = Some(surface_velocity_map);
        self
    }

    /// Clamped grid sample.
    fn sample(&self, ix: i64, iz: i64) -> f64 {
        let ix = ix.clamp(0, self.nx as i64 - 1) as usize;
        let iz = iz.clamp(0, self.nz as i64 - 1) as usize;
        self.heights[iz * self.nx + ix]
    }

    fn sample_from(&self, values: &[f64], x: f64, z: f64) -> f64 {
        let fx = (x - self.origin[0]) / self.spacing;
        let fz = (z - self.origin[1]) / self.spacing;
        let ix = fx.floor() as i64;
        let iz = fz.floor() as i64;
        let tx = fx - ix as f64;
        let tz = fz - iz as f64;
        let sample = |ix: i64, iz: i64| -> f64 {
            let ix = ix.clamp(0, self.nx as i64 - 1) as usize;
            let iz = iz.clamp(0, self.nz as i64 - 1) as usize;
            values[iz * self.nx + ix]
        };
        let v00 = sample(ix, iz);
        let v10 = sample(ix + 1, iz);
        let v01 = sample(ix, iz + 1);
        let v11 = sample(ix + 1, iz + 1);
        let v0 = v00 * (1.0 - tx) + v10 * tx;
        let v1 = v01 * (1.0 - tx) + v11 * tx;
        v0 * (1.0 - tz) + v1 * tz
    }

    fn sample_vec3_from(&self, values: &[[f64; 3]], x: f64, z: f64) -> [f64; 3] {
        [
            self.sample_from_component(values, x, z, 0),
            self.sample_from_component(values, x, z, 1),
            self.sample_from_component(values, x, z, 2),
        ]
    }

    fn sample_from_component(&self, values: &[[f64; 3]], x: f64, z: f64, component: usize) -> f64 {
        let fx = (x - self.origin[0]) / self.spacing;
        let fz = (z - self.origin[1]) / self.spacing;
        let ix = fx.floor() as i64;
        let iz = fz.floor() as i64;
        let tx = fx - ix as f64;
        let tz = fz - iz as f64;
        let sample = |ix: i64, iz: i64| -> f64 {
            let ix = ix.clamp(0, self.nx as i64 - 1) as usize;
            let iz = iz.clamp(0, self.nz as i64 - 1) as usize;
            values[iz * self.nx + ix][component]
        };
        let v00 = sample(ix, iz);
        let v10 = sample(ix + 1, iz);
        let v01 = sample(ix, iz + 1);
        let v11 = sample(ix + 1, iz + 1);
        let v0 = v00 * (1.0 - tx) + v10 * tx;
        let v1 = v01 * (1.0 - tx) + v11 * tx;
        v0 * (1.0 - tz) + v1 * tz
    }
}

impl Terrain for HeightfieldTerrain {
    fn height(&self, x: f64, z: f64) -> f64 {
        let fx = (x - self.origin[0]) / self.spacing;
        let fz = (z - self.origin[1]) / self.spacing;
        let ix = fx.floor() as i64;
        let iz = fz.floor() as i64;
        let tx = fx - ix as f64;
        let tz = fz - iz as f64;
        let h00 = self.sample(ix, iz);
        let h10 = self.sample(ix + 1, iz);
        let h01 = self.sample(ix, iz + 1);
        let h11 = self.sample(ix + 1, iz + 1);
        let h0 = h00 * (1.0 - tx) + h10 * tx;
        let h1 = h01 * (1.0 - tx) + h11 * tx;
        h0 * (1.0 - tz) + h1 * tz
    }

    fn normal(&self, x: f64, z: f64) -> [f64; 3] {
        let d = self.spacing * 0.5;
        let dhdx = (self.height(x + d, z) - self.height(x - d, z)) / (2.0 * d);
        let dhdz = (self.height(x, z + d) - self.height(x, z - d)) / (2.0 * d);
        let n = [-dhdx, 1.0, -dhdz];
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        [n[0] / len, n[1] / len, n[2] / len]
    }

    fn friction(&self) -> f64 {
        self.friction
    }

    fn friction_at(&self, x: f64, z: f64) -> f64 {
        self.friction_map
            .as_ref()
            .map(|map| self.sample_from(map, x, z).max(0.0))
            .unwrap_or(self.friction)
    }

    fn water_film_depth(&self, x: f64, z: f64) -> f64 {
        self.water_film_map
            .as_ref()
            .map(|map| self.sample_from(map, x, z).max(0.0))
            .unwrap_or(0.0)
    }

    fn surface_velocity(&self, x: f64, z: f64) -> [f64; 3] {
        self.surface_velocity_map
            .as_ref()
            .map(|map| self.sample_vec3_from(map, x, z))
            .unwrap_or([0.0; 3])
    }
}

/// A triangle-mesh drivable surface queried by vertical projection and, for
/// FEM contact, finite-triangle closest-surface distance.
///
/// This is the first non-heightfield terrain backend for the tire stack. It is
/// still exposed through the height/normal/friction `Terrain` contract: for a
/// world `(x, z)` it finds all triangles whose XZ projection contains the
/// point, returns the highest interpolated `y`, and exposes that triangle's
/// upward normal and optional friction sample. It is intended for ramps, curbs,
/// bridge plates, and other static road/obstacle meshes. `height`/`normal`
/// stay compatible with the terrain API, while
/// [`Terrain::signed_distance_and_normal`] uses the closest finite triangle so
/// tire-FEM nodes near mesh edges/plates do not fall back to a vertical
/// heightfield-style projection.
#[derive(Clone, Debug)]
pub struct TriangleMeshTerrain {
    pub vertices: Vec<[f64; 3]>,
    pub triangles: Vec<[usize; 3]>,
    pub friction: f64,
    pub triangle_friction: Option<Vec<f64>>,
    pub fallback_height: f64,
    /// Per-triangle bounds used as a broadphase for height and closest-surface
    /// tire contact queries. This keeps large static meshes from requiring a
    /// full closest-point solve against every triangle for every tread node.
    pub triangle_bounds: Vec<TriangleMeshTriangleBounds>,
    /// Binary BVH over `triangle_bounds` for larger static road/obstacle
    /// meshes. Leaves store ranges into `triangle_bvh_indices`.
    pub triangle_bvh: Vec<TriangleMeshBvhNode>,
    pub triangle_bvh_indices: Vec<usize>,
}

#[derive(Clone, Copy, Debug)]
pub struct TriangleMeshTriangleBounds {
    pub min: [f64; 3],
    pub max: [f64; 3],
}

#[derive(Clone, Copy, Debug)]
pub struct TriangleMeshBvhNode {
    pub bounds: TriangleMeshTriangleBounds,
    pub left: Option<usize>,
    pub right: Option<usize>,
    pub start: usize,
    pub count: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct TriangleMeshContactPatch {
    pub triangle_id: usize,
    pub point: [f64; 3],
    pub normal: [f64; 3],
    pub signed_distance: f64,
    pub penetration: f64,
    pub friction: f64,
}

impl TriangleMeshBvhNode {
    pub fn is_leaf(&self) -> bool {
        self.count > 0
    }
}

impl TriangleMeshTriangleBounds {
    pub fn from_triangle(a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> Self {
        Self {
            min: [
                a[0].min(b[0]).min(c[0]),
                a[1].min(b[1]).min(c[1]),
                a[2].min(b[2]).min(c[2]),
            ],
            max: [
                a[0].max(b[0]).max(c[0]),
                a[1].max(b[1]).max(c[1]),
                a[2].max(b[2]).max(c[2]),
            ],
        }
    }

    pub fn contains_xz(self, x: f64, z: f64, eps: f64) -> bool {
        x >= self.min[0] - eps
            && x <= self.max[0] + eps
            && z >= self.min[2] - eps
            && z <= self.max[2] + eps
    }

    pub fn distance2_to_point(self, p: [f64; 3]) -> f64 {
        let dx = if p[0] < self.min[0] {
            self.min[0] - p[0]
        } else if p[0] > self.max[0] {
            p[0] - self.max[0]
        } else {
            0.0
        };
        let dy = if p[1] < self.min[1] {
            self.min[1] - p[1]
        } else if p[1] > self.max[1] {
            p[1] - self.max[1]
        } else {
            0.0
        };
        let dz = if p[2] < self.min[2] {
            self.min[2] - p[2]
        } else if p[2] > self.max[2] {
            p[2] - self.max[2]
        } else {
            0.0
        };
        dx * dx + dy * dy + dz * dz
    }

    pub fn union(self, other: Self) -> Self {
        Self {
            min: [
                self.min[0].min(other.min[0]),
                self.min[1].min(other.min[1]),
                self.min[2].min(other.min[2]),
            ],
            max: [
                self.max[0].max(other.max[0]),
                self.max[1].max(other.max[1]),
                self.max[2].max(other.max[2]),
            ],
        }
    }

    pub fn centroid(self) -> [f64; 3] {
        [
            0.5 * (self.min[0] + self.max[0]),
            0.5 * (self.min[1] + self.max[1]),
            0.5 * (self.min[2] + self.max[2]),
        ]
    }
}

impl TriangleMeshTerrain {
    pub fn new(vertices: Vec<[f64; 3]>, triangles: Vec<[usize; 3]>, friction: f64) -> Self {
        Self::with_fallback_height(vertices, triangles, friction, 0.0)
    }

    pub fn with_fallback_height(
        vertices: Vec<[f64; 3]>,
        triangles: Vec<[usize; 3]>,
        friction: f64,
        fallback_height: f64,
    ) -> Self {
        for tri in &triangles {
            for &id in tri {
                assert!(id < vertices.len(), "triangle index out of bounds");
            }
        }
        let triangle_bounds = build_triangle_bounds(&vertices, &triangles);
        let (triangle_bvh, triangle_bvh_indices) = build_triangle_bvh(&triangle_bounds);
        Self {
            vertices,
            triangles,
            friction,
            triangle_friction: None,
            fallback_height,
            triangle_bounds,
            triangle_bvh,
            triangle_bvh_indices,
        }
    }

    pub fn with_triangle_friction(mut self, triangle_friction: Vec<f64>) -> Self {
        assert_eq!(
            triangle_friction.len(),
            self.triangles.len(),
            "triangle friction length must match triangle count"
        );
        self.triangle_friction = Some(triangle_friction);
        self
    }

    /// Debug/telemetry helper: `(height_xz_candidates, total_triangles)` for
    /// the triangle AABB broadphase at a query point.
    pub fn broadphase_candidate_counts(&self, p: [f64; 3]) -> (usize, usize) {
        let mut xz = 0;
        self.for_each_xz_candidate(p[0], p[2], 1.0e-9, |_| {
            xz += 1;
        });
        (xz, self.triangles.len())
    }

    fn for_each_xz_candidate<F>(&self, x: f64, z: f64, eps: f64, mut f: F)
    where
        F: FnMut(usize),
    {
        if self.triangle_bvh.is_empty() {
            return;
        }
        let mut stack = vec![0usize];
        while let Some(node_id) = stack.pop() {
            let node = self.triangle_bvh[node_id];
            if !node.bounds.contains_xz(x, z, eps) {
                continue;
            }
            if node.is_leaf() {
                let end = node.start + node.count;
                for &ti in &self.triangle_bvh_indices[node.start..end] {
                    if self
                        .triangle_bounds
                        .get(ti)
                        .is_none_or(|bounds| bounds.contains_xz(x, z, eps))
                    {
                        f(ti);
                    }
                }
            } else {
                if let Some(right) = node.right {
                    stack.push(right);
                }
                if let Some(left) = node.left {
                    stack.push(left);
                }
            }
        }
    }

    fn for_each_closest_candidate<F>(&self, p: [f64; 3], max_dist2: f64, mut f: F)
    where
        F: FnMut(usize, &mut f64),
    {
        if self.triangle_bvh.is_empty() {
            return;
        }
        let mut best_dist2 = max_dist2;
        let mut stack = vec![0usize];
        while let Some(node_id) = stack.pop() {
            let node = self.triangle_bvh[node_id];
            if node.bounds.distance2_to_point(p) > best_dist2 {
                continue;
            }
            if node.is_leaf() {
                let end = node.start + node.count;
                for &ti in &self.triangle_bvh_indices[node.start..end] {
                    if self
                        .triangle_bounds
                        .get(ti)
                        .is_some_and(|bounds| bounds.distance2_to_point(p) <= best_dist2)
                    {
                        f(ti, &mut best_dist2);
                    }
                }
            } else {
                if let Some(right) = node.right {
                    stack.push(right);
                }
                if let Some(left) = node.left {
                    stack.push(left);
                }
            }
        }
    }

    fn for_each_radius_candidate<F>(&self, p: [f64; 3], radius: f64, mut f: F)
    where
        F: FnMut(usize),
    {
        if self.triangle_bvh.is_empty() {
            return;
        }
        let radius2 = radius.max(0.0) * radius.max(0.0);
        let mut stack = vec![0usize];
        while let Some(node_id) = stack.pop() {
            let node = self.triangle_bvh[node_id];
            if node.bounds.distance2_to_point(p) > radius2 {
                continue;
            }
            if node.is_leaf() {
                let end = node.start + node.count;
                for &ti in &self.triangle_bvh_indices[node.start..end] {
                    if self
                        .triangle_bounds
                        .get(ti)
                        .is_some_and(|bounds| bounds.distance2_to_point(p) <= radius2)
                    {
                        f(ti);
                    }
                }
            } else {
                if let Some(right) = node.right {
                    stack.push(right);
                }
                if let Some(left) = node.left {
                    stack.push(left);
                }
            }
        }
    }

    fn query(&self, x: f64, z: f64) -> Option<(f64, [f64; 3], f64)> {
        let mut best: Option<(f64, [f64; 3], f64)> = None;
        self.for_each_xz_candidate(x, z, 1.0e-9, |ti| {
            let tri = &self.triangles[ti];
            let a = self.vertices[tri[0]];
            let b = self.vertices[tri[1]];
            let c = self.vertices[tri[2]];
            let denom = (b[2] - c[2]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[2] - c[2]);
            if denom.abs() < 1.0e-12 {
                return;
            }
            let l0 = ((b[2] - c[2]) * (x - c[0]) + (c[0] - b[0]) * (z - c[2])) / denom;
            let l1 = ((c[2] - a[2]) * (x - c[0]) + (a[0] - c[0]) * (z - c[2])) / denom;
            let l2 = 1.0 - l0 - l1;
            let eps = 1.0e-9;
            if l0 < -eps || l1 < -eps || l2 < -eps {
                return;
            }
            let y = l0 * a[1] + l1 * b[1] + l2 * c[1];
            let normal = triangle_normal(a, b, c);
            let friction = self
                .triangle_friction
                .as_ref()
                .and_then(|m| m.get(ti).copied())
                .unwrap_or(self.friction)
                .max(0.0);
            if best.as_ref().is_none_or(|(best_y, _, _)| y > *best_y) {
                best = Some((y, normal, friction));
            }
        });
        best
    }

    fn closest_surface(&self, p: [f64; 3]) -> Option<(f64, [f64; 3], f64)> {
        let mut best: Option<(f64, [f64; 3], f64)> = None;
        self.for_each_closest_candidate(p, f64::INFINITY, |ti, best_dist2| {
            let tri = &self.triangles[ti];
            let a = self.vertices[tri[0]];
            let b = self.vertices[tri[1]];
            let c = self.vertices[tri[2]];
            let normal = triangle_normal(a, b, c);
            let closest = closest_point_on_triangle(p, a, b, c);
            let delta = [p[0] - closest[0], p[1] - closest[1], p[2] - closest[2]];
            let dist2 = dot3(delta, delta);
            if !dist2.is_finite() {
                return;
            }
            let plane_side = dot3(delta, normal);
            let signed_distance = if plane_side >= 0.0 {
                dist2.sqrt()
            } else {
                -dist2.sqrt()
            };
            if dist2 < *best_dist2 {
                *best_dist2 = dist2;
                best = Some((signed_distance, normal, dist2));
            }
        });
        best
    }

    /// Return a multi-triangle contact manifold for a spherical tread/contact
    /// sample against the finite mesh.
    ///
    /// Unlike [`Terrain::signed_distance_and_normal`], this does not collapse
    /// the answer to the single closest triangle. It walks the same triangle
    /// BVH and returns every triangle whose closest finite point lies within
    /// `sphere_radius + contact_margin`, sorted deepest-first and optionally
    /// capped by `max_contacts`. This is the reusable CPU foundation for tire
    /// tread/mesh surface-to-surface contact patches on curbs, seams, plates
    /// and coarse triangle terrain.
    pub fn contact_manifold(
        &self,
        center: [f64; 3],
        sphere_radius: f64,
        contact_margin: f64,
        max_contacts: usize,
    ) -> Vec<TriangleMeshContactPatch> {
        if !sphere_radius.is_finite() || sphere_radius < 0.0 || max_contacts == 0 {
            return Vec::new();
        }
        let query_radius = sphere_radius + contact_margin.max(0.0);
        let mut contacts = Vec::new();
        self.for_each_radius_candidate(center, query_radius, |ti| {
            let tri = &self.triangles[ti];
            let a = self.vertices[tri[0]];
            let b = self.vertices[tri[1]];
            let c = self.vertices[tri[2]];
            let normal = triangle_normal(a, b, c);
            let point = closest_point_on_triangle(center, a, b, c);
            let delta = [
                center[0] - point[0],
                center[1] - point[1],
                center[2] - point[2],
            ];
            let dist2 = dot3(delta, delta);
            if !dist2.is_finite() {
                return;
            }
            let unsigned = dist2.sqrt();
            if unsigned > query_radius {
                return;
            }
            let signed_distance = if dot3(delta, normal) >= 0.0 {
                unsigned
            } else {
                -unsigned
            };
            let penetration = (sphere_radius - signed_distance).max(0.0);
            if penetration <= 0.0 && unsigned > query_radius {
                return;
            }
            let friction = self
                .triangle_friction
                .as_ref()
                .and_then(|m| m.get(ti).copied())
                .unwrap_or(self.friction)
                .max(0.0);
            contacts.push(TriangleMeshContactPatch {
                triangle_id: ti,
                point,
                normal,
                signed_distance,
                penetration,
                friction,
            });
        });
        contacts.sort_by(|a, b| {
            b.penetration
                .partial_cmp(&a.penetration)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    a.signed_distance
                        .partial_cmp(&b.signed_distance)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
        });
        contacts.truncate(max_contacts);
        contacts
    }
}

fn build_triangle_bounds(
    vertices: &[[f64; 3]],
    triangles: &[[usize; 3]],
) -> Vec<TriangleMeshTriangleBounds> {
    triangles
        .iter()
        .map(|tri| {
            TriangleMeshTriangleBounds::from_triangle(
                vertices[tri[0]],
                vertices[tri[1]],
                vertices[tri[2]],
            )
        })
        .collect()
}

fn build_triangle_bvh(
    bounds: &[TriangleMeshTriangleBounds],
) -> (Vec<TriangleMeshBvhNode>, Vec<usize>) {
    let mut indices: Vec<usize> = (0..bounds.len()).collect();
    let mut nodes = Vec::new();
    if !indices.is_empty() {
        build_triangle_bvh_node(bounds, &mut indices, 0, bounds.len(), &mut nodes);
    }
    (nodes, indices)
}

fn build_triangle_bvh_node(
    bounds: &[TriangleMeshTriangleBounds],
    indices: &mut [usize],
    start: usize,
    end: usize,
    nodes: &mut Vec<TriangleMeshBvhNode>,
) -> usize {
    const LEAF_SIZE: usize = 8;
    let mut node_bounds = bounds[indices[start]];
    for &ti in &indices[start + 1..end] {
        node_bounds = node_bounds.union(bounds[ti]);
    }
    let node_id = nodes.len();
    nodes.push(TriangleMeshBvhNode {
        bounds: node_bounds,
        left: None,
        right: None,
        start,
        count: end - start,
    });
    if end - start <= LEAF_SIZE {
        return node_id;
    }

    let mut centroid_min = bounds[indices[start]].centroid();
    let mut centroid_max = centroid_min;
    for &ti in &indices[start + 1..end] {
        let c = bounds[ti].centroid();
        for axis in 0..3 {
            centroid_min[axis] = centroid_min[axis].min(c[axis]);
            centroid_max[axis] = centroid_max[axis].max(c[axis]);
        }
    }
    let extents = [
        centroid_max[0] - centroid_min[0],
        centroid_max[1] - centroid_min[1],
        centroid_max[2] - centroid_min[2],
    ];
    let axis = if extents[0] >= extents[1] && extents[0] >= extents[2] {
        0
    } else if extents[2] >= extents[1] {
        2
    } else {
        1
    };
    if extents[axis] <= 1.0e-12 {
        return node_id;
    }
    indices[start..end].sort_by(|&a, &b| {
        bounds[a].centroid()[axis]
            .partial_cmp(&bounds[b].centroid()[axis])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mid = start + (end - start) / 2;
    let left = build_triangle_bvh_node(bounds, indices, start, mid, nodes);
    let right = build_triangle_bvh_node(bounds, indices, mid, end, nodes);
    nodes[node_id].left = Some(left);
    nodes[node_id].right = Some(right);
    nodes[node_id].count = 0;
    nodes[node_id].start = 0;
    nodes[node_id].bounds = nodes[left].bounds.union(nodes[right].bounds);
    node_id
}

impl Terrain for TriangleMeshTerrain {
    fn height(&self, x: f64, z: f64) -> f64 {
        self.query(x, z)
            .map(|(y, _, _)| y)
            .unwrap_or(self.fallback_height)
    }

    fn normal(&self, x: f64, z: f64) -> [f64; 3] {
        self.query(x, z)
            .map(|(_, n, _)| n)
            .unwrap_or([0.0, 1.0, 0.0])
    }

    fn friction(&self) -> f64 {
        self.friction
    }

    fn friction_at(&self, x: f64, z: f64) -> f64 {
        self.query(x, z)
            .map(|(_, _, friction)| friction)
            .unwrap_or(self.friction)
    }

    fn signed_distance_and_normal(&self, p: [f64; 3]) -> (f64, [f64; 3]) {
        let fallback_distance = p[1] - self.fallback_height;
        if let Some((mesh_distance, mesh_normal, mesh_dist2)) = self.closest_surface(p) {
            if mesh_dist2.sqrt() < fallback_distance.abs() {
                return (mesh_distance, mesh_normal);
            }
        }
        (fallback_distance, [0.0, 1.0, 0.0])
    }

    fn contact_manifold(
        &self,
        center: [f64; 3],
        sphere_radius: f64,
        contact_margin: f64,
        max_contacts: usize,
    ) -> Vec<TriangleMeshContactPatch> {
        TriangleMeshTerrain::contact_manifold(
            self,
            center,
            sphere_radius,
            contact_margin,
            max_contacts,
        )
    }
}

/// Simple signed-distance primitive for [`SdfTerrain`].
///
/// These are native Rust terrain primitives, not importer formats. They give
/// tire contact a reusable SDF query path for rocks/curbs/rounded obstacles
/// while deeper full surface-to-surface SDF contact is built.
#[derive(Clone, Copy, Debug)]
pub enum SdfTerrainPrimitive {
    Sphere {
        center: [f64; 3],
        radius: f64,
        friction: f64,
    },
    Box {
        min: [f64; 3],
        max: [f64; 3],
        friction: f64,
    },
}

impl SdfTerrainPrimitive {
    pub fn sphere(center: [f64; 3], radius: f64, friction: f64) -> Self {
        Self::Sphere {
            center,
            radius: radius.max(1.0e-9),
            friction,
        }
    }

    pub fn aabb(min: [f64; 3], max: [f64; 3], friction: f64) -> Self {
        Self::Box { min, max, friction }
    }

    fn sdf(&self, p: [f64; 3]) -> f64 {
        match *self {
            Self::Sphere { center, radius, .. } => {
                let dx = p[0] - center[0];
                let dy = p[1] - center[1];
                let dz = p[2] - center[2];
                (dx * dx + dy * dy + dz * dz).sqrt() - radius
            }
            Self::Box { min, max, .. } => {
                let c = [
                    0.5 * (min[0] + max[0]),
                    0.5 * (min[1] + max[1]),
                    0.5 * (min[2] + max[2]),
                ];
                let h = [
                    0.5 * (max[0] - min[0]).abs(),
                    0.5 * (max[1] - min[1]).abs(),
                    0.5 * (max[2] - min[2]).abs(),
                ];
                let q = [
                    (p[0] - c[0]).abs() - h[0],
                    (p[1] - c[1]).abs() - h[1],
                    (p[2] - c[2]).abs() - h[2],
                ];
                let outside = [q[0].max(0.0), q[1].max(0.0), q[2].max(0.0)];
                let outside_len =
                    (outside[0] * outside[0] + outside[1] * outside[1] + outside[2] * outside[2])
                        .sqrt();
                let inside = q[0].max(q[1]).max(q[2]).min(0.0);
                outside_len + inside
            }
        }
    }

    fn friction(&self) -> f64 {
        match *self {
            Self::Sphere { friction, .. } | Self::Box { friction, .. } => friction,
        }
    }
}

/// SDF terrain queried by vertical top-surface projection.
///
/// `height(x,z)` scans down from `vertical_max` to `vertical_min` and returns
/// the top-most zero crossing of the union of all SDF primitives, or the flat
/// fallback ground height when no primitive is hit. This keeps compatibility
/// with the existing `Terrain` height/normal API while giving the tire stack
/// true SDF-shaped obstacles without file formats or FFI.
#[derive(Clone, Debug)]
pub struct SdfTerrain {
    pub primitives: Vec<SdfTerrainPrimitive>,
    pub fallback_height: f64,
    pub friction: f64,
    pub vertical_min: f64,
    pub vertical_max: f64,
    pub scan_steps: usize,
}

impl SdfTerrain {
    pub fn new(fallback_height: f64, friction: f64) -> Self {
        Self {
            primitives: Vec::new(),
            fallback_height,
            friction,
            vertical_min: fallback_height - 1.0,
            vertical_max: fallback_height + 2.0,
            scan_steps: 96,
        }
    }

    pub fn with_vertical_range(mut self, vertical_min: f64, vertical_max: f64) -> Self {
        self.vertical_min = vertical_min.min(vertical_max);
        self.vertical_max = vertical_min.max(vertical_max);
        self
    }

    pub fn with_scan_steps(mut self, scan_steps: usize) -> Self {
        self.scan_steps = scan_steps.max(8);
        self
    }

    pub fn add_primitive(&mut self, primitive: SdfTerrainPrimitive) {
        self.primitives.push(primitive);
    }

    pub fn with_primitive(mut self, primitive: SdfTerrainPrimitive) -> Self {
        self.add_primitive(primitive);
        self
    }

    fn sdf(&self, p: [f64; 3]) -> f64 {
        self.primitives
            .iter()
            .map(|prim| prim.sdf(p))
            .fold(f64::INFINITY, f64::min)
    }

    fn sdf_gradient(&self, p: [f64; 3]) -> [f64; 3] {
        let eps = ((self.vertical_max - self.vertical_min).abs() * 1.0e-5).clamp(1.0e-5, 1.0e-3);
        let gx = self.sdf([p[0] + eps, p[1], p[2]]) - self.sdf([p[0] - eps, p[1], p[2]]);
        let gy = self.sdf([p[0], p[1] + eps, p[2]]) - self.sdf([p[0], p[1] - eps, p[2]]);
        let gz = self.sdf([p[0], p[1], p[2] + eps]) - self.sdf([p[0], p[1], p[2] - eps]);
        let len = (gx * gx + gy * gy + gz * gz).sqrt();
        if len > 1.0e-12 {
            [gx / len, gy / len, gz / len]
        } else {
            [0.0, 1.0, 0.0]
        }
    }

    fn top_hit(&self, x: f64, z: f64) -> Option<(f64, f64)> {
        if self.primitives.is_empty() {
            return None;
        }
        let steps = self.scan_steps.max(8);
        let dy = (self.vertical_max - self.vertical_min) / steps as f64;
        if dy <= 0.0 {
            return None;
        }
        let mut y_hi = self.vertical_max;
        let mut sdf_hi = self.sdf([x, y_hi, z]);
        for _ in 0..steps {
            let y_lo = y_hi - dy;
            let sdf_lo = self.sdf([x, y_lo, z]);
            if sdf_hi > 0.0 && sdf_lo <= 0.0 {
                let mut lo = y_lo;
                let mut hi = y_hi;
                for _ in 0..32 {
                    let mid = 0.5 * (lo + hi);
                    if self.sdf([x, mid, z]) <= 0.0 {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                let y = 0.5 * (lo + hi);
                return Some((y, self.primitive_friction_at([x, y, z])));
            }
            y_hi = y_lo;
            sdf_hi = sdf_lo;
        }
        None
    }

    fn primitive_friction_at(&self, p: [f64; 3]) -> f64 {
        let mut best = (f64::INFINITY, self.friction);
        for prim in &self.primitives {
            let d = prim.sdf(p).abs();
            if d < best.0 {
                best = (d, prim.friction().max(0.0));
            }
        }
        best.1
    }
}

impl Terrain for SdfTerrain {
    fn height(&self, x: f64, z: f64) -> f64 {
        self.top_hit(x, z)
            .map(|(y, _)| y)
            .unwrap_or(self.fallback_height)
    }

    fn normal(&self, x: f64, z: f64) -> [f64; 3] {
        if self.top_hit(x, z).is_none() {
            return [0.0, 1.0, 0.0];
        }
        let y = self.height(x, z);
        let eps = ((self.vertical_max - self.vertical_min).abs() * 1.0e-5).clamp(1.0e-5, 1.0e-3);
        let gx = self.sdf([x + eps, y, z]) - self.sdf([x - eps, y, z]);
        let gy = self.sdf([x, y + eps, z]) - self.sdf([x, y - eps, z]);
        let gz = self.sdf([x, y, z + eps]) - self.sdf([x, y, z - eps]);
        let len = (gx * gx + gy * gy + gz * gz).sqrt();
        if len > 1.0e-12 {
            let mut n = [gx / len, gy / len, gz / len];
            if n[1] < 0.0 {
                n = [-n[0], -n[1], -n[2]];
            }
            n
        } else {
            [0.0, 1.0, 0.0]
        }
    }

    fn friction(&self) -> f64 {
        self.friction
    }

    fn friction_at(&self, x: f64, z: f64) -> f64 {
        self.top_hit(x, z)
            .map(|(_, friction)| friction)
            .unwrap_or(self.friction)
    }

    fn signed_distance_and_normal(&self, p: [f64; 3]) -> (f64, [f64; 3]) {
        let plane_distance = p[1] - self.fallback_height;
        let primitive_distance = if self.primitives.is_empty() {
            f64::INFINITY
        } else {
            self.sdf(p)
        };
        if primitive_distance < plane_distance {
            (primitive_distance, self.sdf_gradient(p))
        } else {
            (plane_distance, [0.0, 1.0, 0.0])
        }
    }
}

fn triangle_normal(a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> [f64; 3] {
    let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let mut n = [
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    ];
    if n[1] < 0.0 {
        n = [-n[0], -n[1], -n[2]];
    }
    let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    if len > 1.0e-12 {
        [n[0] / len, n[1] / len, n[2] / len]
    } else {
        [0.0, 1.0, 0.0]
    }
}

fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn normalize3(v: [f64; 3]) -> [f64; 3] {
    let len = dot3(v, v).sqrt();
    if len > 1.0e-12 && len.is_finite() {
        [v[0] / len, v[1] / len, v[2] / len]
    } else {
        [0.0, 1.0, 0.0]
    }
}

fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn add3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn scale3(v: [f64; 3], s: f64) -> [f64; 3] {
    [v[0] * s, v[1] * s, v[2] * s]
}

fn closest_point_on_triangle(p: [f64; 3], a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> [f64; 3] {
    let ab = sub3(b, a);
    let ac = sub3(c, a);
    let ap = sub3(p, a);
    let d1 = dot3(ab, ap);
    let d2 = dot3(ac, ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }

    let bp = sub3(p, b);
    let d3 = dot3(ab, bp);
    let d4 = dot3(ac, bp);
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }

    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3).max(1.0e-12);
        return add3(a, scale3(ab, v));
    }

    let cp = sub3(p, c);
    let d5 = dot3(ab, cp);
    let d6 = dot3(ac, cp);
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }

    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6).max(1.0e-12);
        return add3(a, scale3(ac, w));
    }

    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6)).max(1.0e-12);
        return add3(b, scale3(sub3(c, b), w));
    }

    let denom = (va + vb + vc).max(1.0e-12);
    let v = vb / denom;
    let w = vc / denom;
    add3(a, add3(scale3(ab, v), scale3(ac, w)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_terrain_is_level() {
        let t = FlatTerrain::new(0.5, 0.9);
        assert_eq!(t.height(3.0, -2.0), 0.5);
        assert_eq!(t.normal(3.0, -2.0), [0.0, 1.0, 0.0]);
        assert_eq!(t.friction(), 0.9);
        assert_eq!(t.friction_at(10.0, -7.0), 0.9);
        assert_eq!(t.surface_velocity(0.0, 0.0), [0.0; 3]);
    }

    #[test]
    fn heightfield_interpolates_and_clamps() {
        // A 3×3 grid that ramps in x: h = 2·x (spacing 1).
        let heights = vec![0.0, 2.0, 4.0, 0.0, 2.0, 4.0, 0.0, 2.0, 4.0];
        let t = HeightfieldTerrain::new([0.0, 0.0], 1.0, 3, 3, heights, 0.8);
        // Exact on a sample, bilinear between samples.
        assert!((t.height(1.0, 1.0) - 2.0).abs() < 1e-9);
        assert!((t.height(0.5, 0.5) - 1.0).abs() < 1e-9);
        // Out-of-range clamps to the edge sample.
        assert!((t.height(-5.0, 0.0) - 0.0).abs() < 1e-9);
        assert!((t.height(99.0, 0.0) - 4.0).abs() < 1e-9);
    }

    #[test]
    fn heightfield_normal_tilts_with_the_slope() {
        // A ramp h = 0.5·x → ∂h/∂x = 0.5 → normal ∝ (-0.5, 1, 0).
        let mut heights = Vec::new();
        for _iz in 0..5 {
            for ix in 0..5 {
                heights.push(0.5 * ix as f64);
            }
        }
        let t = HeightfieldTerrain::new([0.0, 0.0], 1.0, 5, 5, heights, 0.8);
        let n = t.normal(2.0, 2.0);
        let expect = {
            let v = [-0.5_f64, 1.0, 0.0];
            let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            [v[0] / l, v[1] / l, v[2] / l]
        };
        for k in 0..3 {
            assert!(
                (n[k] - expect[k]).abs() < 1e-9,
                "normal[{k}] {} vs {}",
                n[k],
                expect[k]
            );
        }
    }

    #[test]
    fn heightfield_friction_map_interpolates_and_clamps() {
        let heights = vec![0.0; 9];
        let friction = vec![1.0, 0.5, 0.1, 1.0, 0.5, 0.1, 1.0, 0.5, 0.1];
        let t =
            HeightfieldTerrain::with_friction_map([0.0, 0.0], 1.0, 3, 3, heights, 0.9, friction);
        assert!((t.friction_at(0.0, 1.0) - 1.0).abs() < 1e-9);
        assert!((t.friction_at(1.0, 1.0) - 0.5).abs() < 1e-9);
        assert!((t.friction_at(0.5, 1.0) - 0.75).abs() < 1e-9);
        assert!((t.friction_at(99.0, 1.0) - 0.1).abs() < 1e-9);
    }

    #[test]
    fn heightfield_surface_velocity_map_interpolates_and_clamps() {
        let heights = vec![0.0; 9];
        let velocities = vec![
            [0.0, 0.0, 0.0],
            [-2.0, 0.0, 0.0],
            [-4.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [-2.0, 0.0, 0.0],
            [-4.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [-2.0, 0.0, 0.0],
            [-4.0, 0.0, 0.0],
        ];
        let terrain = HeightfieldTerrain::new([0.0, 0.0], 1.0, 3, 3, heights, 0.9)
            .with_surface_velocity_map(velocities);
        assert_eq!(terrain.surface_velocity(0.0, 1.0), [0.0, 0.0, 0.0]);
        assert_eq!(terrain.surface_velocity(1.0, 1.0), [-2.0, 0.0, 0.0]);
        let mid = terrain.surface_velocity(0.5, 1.0);
        assert!((mid[0] + 1.0).abs() < 1.0e-9, "mid velocity {mid:?}");
        assert_eq!(terrain.surface_velocity(99.0, 1.0), [-4.0, 0.0, 0.0]);
    }

    #[test]
    fn triangle_mesh_terrain_queries_ramp_height_normal_and_friction() {
        // Square ramp y=x, split into two triangles.
        let terrain = TriangleMeshTerrain::new(
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            vec![[0, 1, 2], [1, 3, 2]],
            0.9,
        )
        .with_triangle_friction(vec![0.4, 1.1]);
        assert!((terrain.height(0.25, 0.25) - 0.25).abs() < 1.0e-9);
        assert!((terrain.height(0.75, 0.75) - 0.75).abs() < 1.0e-9);
        let n = terrain.normal(0.25, 0.25);
        let expect = {
            let v = [-1.0_f64, 1.0, 0.0];
            let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            [v[0] / l, v[1] / l, v[2] / l]
        };
        for k in 0..3 {
            assert!(
                (n[k] - expect[k]).abs() < 1.0e-9,
                "normal[{k}] {} vs {}",
                n[k],
                expect[k]
            );
        }
        assert!((terrain.friction_at(0.25, 0.25) - 0.4).abs() < 1.0e-9);
        assert!((terrain.friction_at(0.75, 0.75) - 1.1).abs() < 1.0e-9);
        assert_eq!(terrain.height(5.0, 5.0), 0.0);
        assert_eq!(terrain.normal(5.0, 5.0), [0.0, 1.0, 0.0]);
    }

    #[test]
    fn triangle_mesh_signed_distance_uses_closest_finite_triangle_near_edges() {
        // Square ramp y=x, split into two triangles. The query point is just
        // outside the XZ footprint, near the front edge and below the ramp
        // surface; vertical height projection would fall back to the flat
        // ground and miss the mesh plate entirely.
        let terrain = TriangleMeshTerrain::new(
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            vec![[0, 1, 2], [1, 3, 2]],
            0.9,
        );
        let (distance, normal) = terrain.signed_distance_and_normal([0.52, 0.45, -0.02]);
        assert!(
            distance < 0.0 && distance.abs() < 0.10,
            "closest finite triangle should catch below-ramp edge contact instead of fallback ground: distance={distance} normal={normal:?}"
        );
        assert!(
            normal[0] < -0.65 && normal[1] > 0.65 && normal[2].abs() < 1.0e-9,
            "triangle closest-surface normal should be the ramp normal, not fallback up: {normal:?}"
        );

        let (fallback_distance, fallback_normal) =
            terrain.signed_distance_and_normal([5.0, -0.02, 0.0]);
        assert!((fallback_distance + 0.02).abs() < 1.0e-12);
        assert_eq!(fallback_normal, [0.0, 1.0, 0.0]);
    }

    #[test]
    fn triangle_mesh_broadphase_reduces_height_candidates_without_changing_query() {
        let mut vertices = Vec::new();
        let mut triangles = Vec::new();
        for ix in 0..16 {
            for iz in 0..16 {
                let x = ix as f64 * 2.0;
                let z = iz as f64 * 2.0;
                let base = vertices.len();
                vertices.push([x, 0.0, z]);
                vertices.push([x + 1.0, 0.1 * ix as f64, z]);
                vertices.push([x, 0.0, z + 1.0]);
                vertices.push([x + 1.0, 0.1 * ix as f64, z + 1.0]);
                triangles.push([base, base + 1, base + 2]);
                triangles.push([base + 1, base + 3, base + 2]);
            }
        }
        let terrain = TriangleMeshTerrain::new(vertices, triangles, 0.8);
        assert_eq!(terrain.triangle_bounds.len(), terrain.triangles.len());
        assert!(
            terrain.triangle_bvh.len() > 1,
            "large static triangle meshes should build a reusable BVH"
        );
        assert_eq!(terrain.triangle_bvh_indices.len(), terrain.triangles.len());
        assert!(
            !terrain.triangle_bvh[0].is_leaf(),
            "root BVH node should split this grid"
        );
        let (candidates, total) = terrain.broadphase_candidate_counts([2.25, 0.0, 0.25]);
        assert!(
            candidates <= 2 && total > 400,
            "BVH/AABB broadphase should narrow height query candidates: candidates={candidates} total={total}"
        );
        assert!(
            (terrain.height(2.25, 0.25) - 0.025).abs() < 1.0e-9,
            "broadphase must preserve triangle height query"
        );
        let (miss_candidates, _) = terrain.broadphase_candidate_counts([100.0, 0.0, 100.0]);
        assert_eq!(
            miss_candidates, 0,
            "far-away height queries should reject all triangle AABBs"
        );
        assert_eq!(terrain.height(100.0, 100.0), 0.0);
    }

    #[test]
    fn triangle_mesh_bvh_preserves_closest_surface_near_sparse_plate() {
        let mut vertices = Vec::new();
        let mut triangles = Vec::new();
        for ix in 0..18 {
            for iz in 0..18 {
                let x = ix as f64 * 4.0;
                let z = iz as f64 * 4.0;
                let base = vertices.len();
                vertices.push([x, 0.2, z]);
                vertices.push([x + 1.0, 0.2, z]);
                vertices.push([x, 0.2, z + 1.0]);
                vertices.push([x + 1.0, 0.2, z + 1.0]);
                triangles.push([base, base + 1, base + 2]);
                triangles.push([base + 1, base + 3, base + 2]);
            }
        }
        let terrain = TriangleMeshTerrain::new(vertices, triangles, 0.7);
        assert!(terrain.triangle_bvh.len() > 8);
        let p = [20.25, 0.17, 20.25];
        let (distance, normal) = terrain.signed_distance_and_normal(p);
        assert!(
            distance < 0.0 && (distance + 0.03).abs() < 1.0e-9,
            "BVH closest-surface traversal should preserve finite plate distance, got {distance}"
        );
        assert_eq!(normal, [0.0, 1.0, 0.0]);
        let (candidates, total) = terrain.broadphase_candidate_counts(p);
        assert!(
            candidates <= 2 && total > 600,
            "BVH height traversal should only visit the local plate pair: candidates={candidates} total={total}"
        );
    }

    #[test]
    fn triangle_mesh_contact_manifold_returns_multiple_finite_patches() {
        // Two adjacent plates with different friction. A spherical tread sample
        // straddling the seam should not collapse to one closest triangle.
        let terrain = TriangleMeshTerrain::new(
            vec![
                [-1.0, 0.0, -1.0],
                [0.0, 0.0, -1.0],
                [-1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, -1.0],
                [1.0, 0.0, 1.0],
            ],
            vec![[0, 1, 2], [1, 3, 2], [1, 4, 3], [4, 5, 3]],
            0.8,
        )
        .with_triangle_friction(vec![0.4, 0.5, 1.1, 1.2]);
        let contacts = terrain.contact_manifold([0.0, 0.08, 0.0], 0.12, 0.02, 8);
        assert!(
            contacts.len() >= 2,
            "seam contact should return a multi-triangle manifold: {contacts:?}"
        );
        assert!(
            contacts.iter().any(|c| c.triangle_id <= 1)
                && contacts.iter().any(|c| c.triangle_id >= 2),
            "manifold should include both sides of the seam: {contacts:?}"
        );
        assert!(
            contacts.iter().all(|c| c.penetration > 0.0),
            "sphere below radius should produce positive penetration contacts: {contacts:?}"
        );
        assert!(
            contacts.iter().any(|c| c.friction < 0.6) && contacts.iter().any(|c| c.friction > 1.0),
            "per-triangle material should survive manifold construction: {contacts:?}"
        );
        let capped = terrain.contact_manifold([0.0, 0.08, 0.0], 0.12, 0.02, 1);
        assert_eq!(capped.len(), 1);
        assert!(capped[0].penetration >= contacts.last().unwrap().penetration);
    }

    #[test]
    fn sdf_terrain_queries_sphere_cap_height_normal_and_friction() {
        let terrain = SdfTerrain::new(0.0, 0.9)
            .with_vertical_range(-1.0, 2.0)
            .with_scan_steps(160)
            .with_primitive(SdfTerrainPrimitive::sphere([0.0, 0.5, 0.0], 0.5, 0.35));
        assert!((terrain.height(0.0, 0.0) - 1.0).abs() < 1.0e-5);
        assert_eq!(terrain.normal(0.0, 0.0), [0.0, 1.0, 0.0]);
        assert!((terrain.friction_at(0.0, 0.0) - 0.35).abs() < 1.0e-9);
        let x = 0.3;
        let expected_y = 0.5 + (0.5_f64 * 0.5 - x * x).sqrt();
        assert!((terrain.height(x, 0.0) - expected_y).abs() < 1.0e-5);
        let n = terrain.normal(x, 0.0);
        assert!(
            n[0] > 0.55 && n[1] > 0.75,
            "sphere cap normal should tilt outward/upward: {n:?}"
        );
        assert_eq!(terrain.height(2.0, 0.0), 0.0);
        assert_eq!(terrain.normal(2.0, 0.0), [0.0, 1.0, 0.0]);
        assert_eq!(terrain.friction_at(2.0, 0.0), 0.9);
    }

    #[test]
    fn sdf_terrain_signed_distance_uses_closest_surface_not_vertical_top() {
        let terrain = SdfTerrain::new(0.0, 0.9)
            .with_vertical_range(-1.0, 2.0)
            .with_scan_steps(160)
            .with_primitive(SdfTerrainPrimitive::sphere([0.0, 0.5, 0.0], 0.5, 0.35));
        let p = [0.49, 0.5, 0.0];
        let (distance, normal) = terrain.signed_distance_and_normal(p);
        assert!(
            (distance + 0.01).abs() < 1.0e-6,
            "signed distance should be radial SDF, got {distance}"
        );
        assert!(
            normal[0] > 0.999 && normal[1].abs() < 1.0e-6,
            "closest SDF normal should point sideways, not upward top-projection normal: {normal:?}"
        );
        let (ground_distance, ground_normal) =
            terrain.signed_distance_and_normal([2.0, -0.02, 0.0]);
        assert!((ground_distance + 0.02).abs() < 1.0e-12);
        assert_eq!(ground_normal, [0.0, 1.0, 0.0]);
    }

    #[test]
    fn sdf_terrain_queries_box_top() {
        let terrain = SdfTerrain::new(0.0, 0.8)
            .with_vertical_range(-0.5, 1.5)
            .with_primitive(SdfTerrainPrimitive::aabb(
                [-0.5, 0.0, -0.5],
                [0.5, 0.4, 0.5],
                1.2,
            ));
        assert!((terrain.height(0.0, 0.0) - 0.4).abs() < 1.0e-5);
        assert_eq!(terrain.normal(0.0, 0.0), [0.0, 1.0, 0.0]);
        assert!((terrain.friction_at(0.0, 0.0) - 1.2).abs() < 1.0e-9);
        assert_eq!(terrain.height(0.7, 0.0), 0.0);
    }
}
