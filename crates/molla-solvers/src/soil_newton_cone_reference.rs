use super::*;

fn cone_fischer(
    n: DVec3,
    a: DVec3,
    b: DVec3,
    da: DVec3,
    db: DVec3,
    epsilon: f64,
) -> (DVec3, DVec3) {
    let scale = a.length().max(b.length()).max(epsilon);
    if scale == 0. {
        return (
            DVec3::ZERO,
            (1. - std::f64::consts::FRAC_1_SQRT_2) * (da + db),
        );
    }
    let x = a / scale;
    let y = b / scale;
    let e = epsilon / scale;
    let xn = n.dot(x);
    let yn = n.dot(y);
    let xt = x - n * xn;
    let yt = y - n * yn;
    let tangent = 2. * (xn * xt + yn * yt);
    let length = tangent.length();
    let unit = if length > 0. {
        tangent / length
    } else {
        DVec3::ZERO
    };
    let xp = unit.dot(xt);
    let yp = unit.dot(yt);
    let xperp = xt - unit * xp;
    let yperp = yt - unit * yp;
    let perpendicular = xperp.length_squared() + yperp.length_squared() + 2. * e * e;
    let plus = ((xn + xp).powi(2) + (yn + yp).powi(2) + perpendicular).sqrt();
    let minus = ((xn - xp).powi(2) + (yn - yp).powi(2) + perpendicular).sqrt();
    let zn = 0.5 * (plus + minus);
    let z = n * zn + tangent / (2. * zn);
    let dan = n.dot(da);
    let dbn = n.dot(db);
    let dat = da - n * dan;
    let dbt = db - n * dbn;
    let dap = unit.dot(dat);
    let dbp = unit.dot(dbt);
    let cross = xperp.dot(dat) + yperp.dot(dbt);
    let positive = if plus > 0. {
        ((xn + xp) * (dan + dap) + (yn + yp) * (dbn + dbp) + cross) / (2. * plus)
    } else {
        0.
    };
    let negative = if minus > 0. {
        ((xn - xp) * (dan - dap) + (yn - yp) * (dbn - dbp) + cross) / (2. * minus)
    } else {
        0.
    };
    let perpendicular_rhs =
        xn * (dat - unit * dap) + dan * xperp + yn * (dbt - unit * dbp) + dbn * yperp;
    let dz = (n + unit) * positive + (n - unit) * negative + perpendicular_rhs / zn;
    (a + b - scale * z, da + db - dz)
}

pub(super) fn contact(
    c: &Contact,
    k: DVec3,
    relative: DVec3,
    dk: DVec3,
    dr: DVec3,
) -> (DVec3, DVec3) {
    let n = c.normal;
    let lambda = n.dot(k);
    let tangent = k - n * lambda;
    let slip = relative - n * n.dot(relative);
    let dslip = dr - n * n.dot(dr);
    let speed = slip.length();
    let dspeed = if speed > 0. {
        slip.dot(dslip) / speed
    } else {
        0.
    };
    let a = n * (c.mu * lambda) + tangent;
    let b = c.rho * (n * (n.dot(relative) + c.gap + c.mu * speed) + c.mu * slip);
    let da = dk + n * ((c.mu - 1.) * n.dot(dk));
    let db = c.rho * (n * (n.dot(dr) + c.mu * dspeed) + c.mu * dslip);
    cone_fischer(n, a, b, da, db, c.smoothing)
}

#[test]
fn global_newton_cone_complementarity() {
    let n = DVec3::Y;
    for (a, b) in [
        (DVec3::ZERO, DVec3::Y),
        (DVec3::Y, DVec3::ZERO),
        (DVec3::new(1., 1., 0.), DVec3::new(-2., 2., 0.)),
        (DVec3::ZERO, DVec3::ZERO),
    ] {
        assert!(
            cone_fischer(n, a, b, DVec3::ZERO, DVec3::ZERO, 0.)
                .0
                .length()
                < 1e-14
        );
    }
    assert!(
        cone_fischer(n, -n, n, DVec3::ZERO, DVec3::ZERO, 0.)
            .0
            .length()
            > 1.
    );
    assert!(
        cone_fischer(n, n, n, DVec3::ZERO, DVec3::ZERO, 0.)
            .0
            .length()
            > 0.1
    );
}

#[test]
fn global_newton_cone_derivative() {
    let n = DVec3::new(0.2, -0.9, 0.3).normalize();
    for epsilon in [0., 0.01] {
        for i in 0..32 {
            let t = i as f64 + 0.3;
            let a = DVec3::new(t.sin(), (3. * t).cos(), (2. * t).sin());
            let b = DVec3::new((2. * t).cos(), t.cos(), (3. * t).sin());
            let da = DVec3::new(0.3, -0.4, 0.6);
            let db = DVec3::new(-0.7, 0.2, 0.5);
            let analytic = cone_fischer(n, a, b, da, db, epsilon).1;
            let h = 1e-6;
            let plus = cone_fischer(n, a + h * da, b + h * db, DVec3::ZERO, DVec3::ZERO, epsilon).0;
            let minus =
                cone_fischer(n, a - h * da, b - h * db, DVec3::ZERO, DVec3::ZERO, epsilon).0;
            assert!((analytic - (plus - minus) / (2. * h)).length() < 1e-8);
        }
    }
}
