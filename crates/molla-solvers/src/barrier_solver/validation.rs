use molla_core::{Error, Result};
use molla_math::{SpatialVector, Transform};
use molla_sim::{Model, ShapeGeometry, State};

fn invalid(name: &str) -> Error {
    Error::Solver(format!("barrier solver: invalid {name}"))
}

fn length(actual: usize, expected: usize, name: &str) -> Result<()> {
    if actual != expected {
        return Err(invalid(name));
    }
    Ok(())
}

fn transform(q: &Transform) -> bool {
    q.position.is_finite()
        && q.rotation.is_finite()
        && (q.rotation.length_squared() - 1.0).abs() <= 1.0e-8
}

pub(super) fn motion(q: &[Transform], qd: &[SpatialVector]) -> Result<()> {
    if q.iter().any(|q| !transform(q))
        || qd
            .iter()
            .any(|v| !v.linear.is_finite() || !v.angular.is_finite())
    {
        return Err(invalid("body motion"));
    }
    Ok(())
}

pub(super) fn preflight(model: &Model, input: &State, output: &State) -> Result<()> {
    let np = model.particle_count;
    let nb = model.body_count;
    length(
        input.particle_q.host()?.len(),
        np,
        "input particle positions length",
    )?;
    length(
        input.particle_qd.host()?.len(),
        np,
        "input particle velocities length",
    )?;
    length(input.body_q.host()?.len(), nb, "input body poses length")?;
    length(
        input.body_qd.host()?.len(),
        nb,
        "input body velocities length",
    )?;
    length(
        output.particle_q.host()?.len(),
        np,
        "output particle positions length",
    )?;
    length(
        output.particle_qd.host()?.len(),
        np,
        "output particle velocities length",
    )?;
    length(output.body_q.host()?.len(), nb, "output body poses length")?;
    length(
        output.body_qd.host()?.len(),
        nb,
        "output body velocities length",
    )?;
    motion(input.body_q.host()?, input.body_qd.host()?)?;
    let inv_mass = model.body_inv_mass.host()?;
    let inv_inertia = model.body_inv_inertia.host()?;
    length(inv_mass.len(), nb, "body inverse masses length")?;
    length(inv_inertia.len(), nb, "body inverse inertias length")?;
    if inv_mass.iter().any(|v| !v.is_finite() || *v < 0.0)
        || inv_inertia.iter().any(|v| !v.is_finite())
    {
        return Err(invalid("body inverse mass or inertia"));
    }
    Ok(())
}

pub(super) fn model(model: &Model, gpu: bool, rod_edges: bool) -> Result<()> {
    let np = model.particle_count;
    let mass = model.particle_mass.host()?;
    let inv_mass = model.particle_inv_mass.host()?;
    length(mass.len(), np, "particle masses length")?;
    length(inv_mass.len(), np, "particle inverse masses length")?;
    if mass.iter().zip(inv_mass).any(|(&m, &w)| {
        !m.is_finite() || !w.is_finite() || m < 0.0 || w < 0.0 || (w > 0.0 && m == 0.0)
    }) {
        return Err(invalid("particle mass"));
    }
    let ns = model.spring_count;
    let a = model.spring_a.host()?;
    let b = model.spring_b.host()?;
    let rest = model.spring_rest_length.host()?;
    let compliance = model.spring_compliance.host()?;
    for len in [
        a.len(),
        b.len(),
        rest.len(),
        compliance.len(),
        model.spring_tension_only.host()?.len(),
    ] {
        length(len, ns, "spring arrays length")?;
    }
    for i in 0..ns {
        if a[i] == b[i]
            || a[i] as usize >= np
            || b[i] as usize >= np
            || !rest[i].is_finite()
            || rest[i] < 0.0
            || !compliance[i].is_finite()
            || compliance[i] < 0.0
        {
            return Err(invalid("spring"));
        }
    }
    for t in &model.soft_surface_tris {
        if t.iter()
            .enumerate()
            .any(|(i, &v)| v as usize >= np || t[..i].contains(&v))
        {
            return Err(invalid("surface triangle"));
        }
    }
    for s in &model.soft_surfaces {
        if s.vertex_offset
            .checked_add(s.vertex_count)
            .is_none_or(|end| end as usize > np)
        {
            return Err(invalid("surface vertex range"));
        }
    }
    let nb = model.body_count;
    let mass = model.body_mass.host()?;
    length(mass.len(), nb, "body masses length")?;
    if mass
        .iter()
        .zip(model.body_inv_mass.host()?)
        .any(|(&m, &w)| !m.is_finite() || m < 0.0 || (w > 0.0 && m == 0.0))
    {
        return Err(invalid("body mass"));
    }
    let ns = model.shape_count;
    let body = model.shape_body.host()?;
    let xf = model.shape_transform.host()?;
    for len in [body.len(), xf.len(), model.shape_geometry.len()] {
        length(len, ns, "shape arrays length")?;
    }
    for s in 0..ns {
        if body[s] < -1 || (body[s] >= 0 && body[s] as usize >= nb) || !transform(&xf[s]) {
            return Err(invalid("shape body or transform"));
        }
        if gpu && body[s] >= 0 {
            return Err(Error::Solver(
                "barrier gpu: body-attached shapes are not supported".into(),
            ));
        }
        let valid = match model.shape_geometry[s] {
            ShapeGeometry::Plane { normal } => {
                normal.is_finite()
                    && normal.length_squared().is_finite()
                    && normal.length_squared() > 0.0
            }
            ShapeGeometry::Sphere { radius } => radius.is_finite() && radius > 0.0,
            ShapeGeometry::Box { half_extents }
                if !gpu && body[s] >= 0 && model.body_inv_mass.host()?[body[s] as usize] > 0.0 =>
            {
                half_extents.is_finite() && half_extents.min_element() > 0.0
            }
            _ => {
                return Err(Error::Solver(
                    "barrier solver: unsupported collider geometry".into(),
                ));
            }
        };
        if !valid {
            return Err(invalid("shape geometry"));
        }
    }
    if model.tet_count > 0 {
        return Err(Error::Solver(
            "barrier solver: Model tet materials are not integrated".into(),
        ));
    }
    if gpu && rod_edges {
        return Err(Error::Solver(
            "barrier gpu: explicit rod collision edges are not integrated".into(),
        ));
    }
    Ok(())
}
