use super::{BarrierObstacle, BarrierShell, Error, Real, Result, Vec3};
use glam::DVec2 as Vec2;
use std::collections::{BTreeMap, BTreeSet};

fn invalid(name: &str) -> Error {
    Error::Solver(format!("barrier: invalid {name}"))
}

pub(super) fn finite_scalars(values: &[Real], stage: &str) -> Result<()> {
    if values.iter().any(|v| !v.is_finite()) {
        return Err(invalid(stage));
    }
    Ok(())
}

pub(super) fn finite_vectors(values: &[Vec3], stage: &str) -> Result<()> {
    if values.iter().any(|v| !v.is_finite()) {
        return Err(invalid(stage));
    }
    Ok(())
}

impl BarrierShell {
    pub(super) fn validate_inputs(&self) -> Result<Vec<Real>> {
        let n = self.positions.len();
        if self.velocities.len() != n
            || self.masses.len() != n
            || self.pinned.len() != n
            || self.proxy_of.len() != n
            || self.vertex_ksum.len() != n
            || self.tets.len() != self.tet_v0.len()
            || self.edge_run.len() != self.collision_edges.len()
        {
            return Err(invalid("state or topology lengths"));
        }
        if self
            .positions
            .iter()
            .chain(&self.velocities)
            .any(|v| !v.is_finite())
            || self.masses.iter().any(|m| !m.is_finite() || *m <= 0.0)
        {
            return Err(invalid("particle state"));
        }
        let ids_valid = |ids: &[u32]| {
            ids.iter()
                .enumerate()
                .all(|(i, &v)| (v as usize) < n && !ids[..i].contains(&v))
        };
        let mut stiffness = vec![0.0; n];
        for s in &self.springs {
            if !ids_valid(&[s.a, s.b])
                || !s.rest.is_finite()
                || s.rest < 0.0
                || !s.k.is_finite()
                || s.k < 0.0
                || (self.params.strain_limit.is_some() && s.rest == 0.0)
            {
                return Err(invalid("spring or strain limit"));
            }
            stiffness[s.a as usize] += s.k;
            stiffness[s.b as usize] += s.k;
        }
        if stiffness.iter().any(|v: &Real| !v.is_finite()) {
            return Err(invalid("incident stiffness"));
        }
        let mut expected_edges = BTreeSet::new();
        for t in &self.triangles {
            if !ids_valid(t) {
                return Err(invalid("triangle indices"));
            }
            for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                expected_edges.insert((a.min(b), a.max(b)));
            }
        }
        let mut actual_edges = BTreeSet::new();
        let mut runs = BTreeMap::new();
        for (e, provenance) in self.collision_edges.iter().zip(&self.edge_run) {
            if !ids_valid(e) {
                return Err(invalid("collision edge"));
            }
            if let Some((run, index)) = provenance {
                if runs.insert((*run, *index), *e).is_some() {
                    return Err(invalid("edge run"));
                }
            } else {
                actual_edges.insert((e[0].min(e[1]), e[0].max(e[1])));
            }
        }
        if actual_edges != expected_edges {
            return Err(invalid("derived triangle edges; use with_triangles"));
        }
        for ((run, index), e) in &runs {
            if *index > 0
                && !runs
                    .get(&(*run, index - 1))
                    .is_some_and(|prev| e.iter().any(|i| prev.contains(i)))
            {
                return Err(invalid("edge run adjacency"));
            }
        }
        for (tet, rest) in self.tets.iter().zip(&self.tet_v0) {
            if !ids_valid(&tet.ids)
                || !tet.kv.is_finite()
                || tet.kv < 0.0
                || !rest.is_finite()
                || *rest == 0.0
            {
                return Err(invalid("tet or rest volume"));
            }
        }
        for o in &self.obstacles {
            let valid = match *o {
                BarrierObstacle::HalfSpace { n, offset } => {
                    n.is_finite()
                        && (n.length_squared() - 1.0).abs() <= 1.0e-8
                        && offset.is_finite()
                }
                BarrierObstacle::Sphere { center, radius } => {
                    center.is_finite() && radius.is_finite() && radius > 0.0
                }
            };
            if !valid {
                return Err(invalid("obstacle"));
            }
        }
        let mut expected_proxy = vec![None; n];
        for (i, body) in self.affine_bodies.iter().enumerate() {
            let end = body
                .proxy_start
                .checked_add(body.proxy_count)
                .filter(|&end| end <= n)
                .ok_or_else(|| invalid("affine proxy range"))?;
            if body.proxy_count == 0
                || body.rest.len() != body.proxy_count
                || body.rest.iter().any(|p| !p.is_finite())
                || !body.p.is_finite()
                || !body.a_mat.is_finite()
                || !body.v_p.is_finite()
                || !body.v_a.is_finite()
                || !body.mass.is_finite()
                || body.mass <= 0.0
                || !body.kappa_ortho.is_finite()
                || body.kappa_ortho < 0.0
                || !body.j_mat.is_finite()
                || !body.j_mat.determinant().is_finite()
                || body.j_mat.determinant() <= 0.0
            {
                return Err(invalid("affine state or inertia"));
            }
            let j = body.j_mat;
            let scale = j
                .to_cols_array()
                .iter()
                .map(|v| v.abs())
                .fold(0.0, Real::max);
            let asymmetry = (j - j.transpose())
                .to_cols_array()
                .iter()
                .map(|v| v.abs())
                .fold(0.0, Real::max);
            let minor = j.x_axis.x * j.y_axis.y - j.x_axis.y * j.y_axis.x;
            if asymmetry > 64.0 * Real::EPSILON * scale
                || j.x_axis.x <= 0.0
                || !minor.is_finite()
                || minor <= 0.0
                || !j.inverse().is_finite()
            {
                return Err(invalid("affine positive-definite inertia"));
            }
            for owner in &mut expected_proxy[body.proxy_start..end] {
                if owner.replace(i as u32).is_some() {
                    return Err(invalid("overlapping affine proxies"));
                }
            }
        }
        if expected_proxy != self.proxy_of {
            return Err(invalid("affine proxy ownership"));
        }
        Ok(stiffness)
    }

    pub(super) fn validate_geometry(&self, x: &[Vec3], stage: &str, contacts: bool) -> Result<()> {
        let fail =
            |what: &str| Error::Solver(format!("barrier: {stage} geometry rejected: {what}"));
        if x.len() != self.positions.len() || x.iter().any(|p| !p.is_finite()) {
            return Err(fail("nonfinite state"));
        }
        for element in &self.elastic_elements {
            element.validate(x)?;
        }
        for t in &self.triangles {
            let area2 = (x[t[1] as usize] - x[t[0] as usize])
                .cross(x[t[2] as usize] - x[t[0] as usize])
                .length_squared();
            if !area2.is_finite() || area2 <= 0.0 {
                return Err(fail("degenerate triangle"));
            }
        }
        for (t, &rest) in self.tets.iter().zip(&self.tet_v0) {
            let volume = Self::tet_volume(x, t.ids);
            if !volume.is_finite() || volume == 0.0 || (volume < 0.0) != (rest < 0.0) {
                return Err(fail("degenerate or inverted tet"));
            }
        }
        if let Some(limit) = self.params.strain_limit {
            for s in &self.springs {
                let ratio = (x[s.a as usize] - x[s.b as usize]).length() / s.rest;
                if !ratio.is_finite() || ratio >= limit {
                    return Err(fail("strain limit"));
                }
            }
        }
        if contacts {
            for (i, &p) in x.iter().enumerate() {
                if self.pinned[i] {
                    continue;
                }
                for o in &self.obstacles {
                    let gap = o.distance(p).0;
                    if !gap.is_finite() || gap <= 0.0 {
                        return Err(fail("obstacle INTERSECTS or nonfinite distance"));
                    }
                }
            }
            if self.has_intersections_at(x) {
                return Err(fail("mesh INTERSECTS"));
            }
            if self.min_self_gap(x) <= 0.0 {
                return Err(fail("self-contact INTERSECTS"));
            }
        }
        Ok(())
    }

    pub(super) fn candidate_velocities(&self, x: &[Vec3], h: Real) -> Result<Vec<Vec3>> {
        let velocities: Vec<_> = x
            .iter()
            .enumerate()
            .map(|(i, &p)| {
                if self.pinned[i] {
                    Vec3::ZERO
                } else {
                    (p - self.positions[i]) / h
                }
            })
            .collect();
        if velocities.iter().any(|v| !v.is_finite()) {
            return Err(invalid("accepted velocity"));
        }
        Ok(velocities)
    }

    pub(super) fn has_intersections_at(&self, x: &[Vec3]) -> bool {
        for t in &self.triangles {
            let [a, b, c] = t.map(|i| x[i as usize]);
            let lo = a.min(b).min(c);
            let hi = a.max(b).max(c);
            let eps = 32.0 * Real::EPSILON * lo.abs().max(hi.abs()).max_element().max(1.0e-12);
            let overlaps = |p: Vec3, q: Vec3| {
                p.min(q).cmple(hi + Vec3::splat(eps)).all()
                    && p.max(q).cmpge(lo - Vec3::splat(eps)).all()
            };
            for &e in &self.collision_edges {
                if self.same_body(e[0], t[0]) {
                    continue;
                }
                let p = x[e[0] as usize];
                let q = x[e[1] as usize];
                if overlaps(p, q)
                    && segment_triangle(p, q, a, b, c, [t.contains(&e[0]), t.contains(&e[1])])
                {
                    return true;
                }
            }
            for (i, &p) in x.iter().enumerate() {
                if t.contains(&(i as u32)) || self.same_body(i as u32, t[0]) {
                    continue;
                }
                if overlaps(p, p) && segment_triangle(p, p, a, b, c, [false; 2]) {
                    return true;
                }
            }
        }
        false
    }
}

fn segment_triangle(p0: Vec3, q0: Vec3, a: Vec3, b0: Vec3, c0: Vec3, shared: [bool; 2]) -> bool {
    if shared[0] && shared[1] {
        return false;
    }
    let scale = [p0 - a, q0 - a, b0 - a, c0 - a]
        .iter()
        .map(|v| v.abs().max_element())
        .fold(0.0, Real::max);
    if !scale.is_finite() || scale <= 0.0 {
        return true;
    }
    let [p, q, b, c] = [p0 - a, q0 - a, b0 - a, c0 - a].map(|v| v / scale);
    let normal = b.cross(c);
    if !normal.is_finite() || normal.length_squared() == 0.0 {
        return true;
    }
    let magnitude = [p0, q0, a, b0, c0]
        .iter()
        .map(|v| v.abs().max_element())
        .fold(scale, Real::max);
    let eps = 32.0 * Real::EPSILON * (magnitude / scale).max(1.0);
    let plane_eps = eps * normal.abs().element_sum();
    let d0 = normal.dot(p);
    let d1 = normal.dot(q);
    if (d0 > plane_eps && d1 > plane_eps) || (d0 < -plane_eps && d1 < -plane_eps) {
        return false;
    }
    let abs = normal.abs();
    let axis = if abs.x >= abs.y && abs.x >= abs.z {
        0
    } else if abs.y >= abs.z {
        1
    } else {
        2
    };
    let project = |p: Vec3| match axis {
        0 => Vec2::new(p.y, p.z),
        1 => Vec2::new(p.x, p.z),
        _ => Vec2::new(p.x, p.y),
    };
    let corners = [Vec2::ZERO, project(b), project(c)];
    let cross = |a: Vec2, b: Vec2| a.x * b.y - a.y * b.x;
    let orientation = cross(corners[1], corners[2]).signum();
    let inside = |p: Vec2| {
        (0..3).all(|i| {
            let e = corners[(i + 1) % 3] - corners[i];
            orientation * cross(e, p - corners[i]) >= -eps * e.length()
        })
    };
    let end_eps = 4.0 * eps / (q - p).length().max(eps);
    let retained = |t: Real| {
        t >= 0.0 && t <= 1.0 && !(shared[0] && t <= end_eps) && !(shared[1] && t >= 1.0 - end_eps)
    };
    if d0.abs() > plane_eps || d1.abs() > plane_eps {
        let t = d0 / (d0 - d1);
        return !t.is_finite() || (retained(t) && inside(project(p + (q - p) * t)));
    }
    let p = project(p);
    let q = project(q);
    if (!shared[0] && inside(p)) || (!shared[1] && inside(q)) {
        return true;
    }
    let d = q - p;
    for i in 0..3 {
        let start = corners[i];
        let e = corners[(i + 1) % 3] - start;
        let denominator = cross(d, e);
        if denominator.abs() > eps * d.length() * e.length() {
            let t = cross(start - p, e) / denominator;
            let s = cross(start - p, d) / denominator;
            if retained(t) && (-eps..=1.0 + eps).contains(&s) {
                return true;
            }
        } else if d.length_squared() > 0.0 && cross(start - p, d).abs() <= eps * d.length() {
            let t0 = (start - p).dot(d) / d.length_squared();
            let t1 = (start + e - p).dot(d) / d.length_squared();
            let lo = t0.min(t1).max(0.0);
            let hi = t0.max(t1).min(1.0);
            if lo <= hi && (retained(lo) || retained(hi) || retained((lo + hi) * 0.5)) {
                return true;
            }
        }
    }
    false
}
