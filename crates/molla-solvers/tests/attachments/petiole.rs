use molla_math::Vec3;
use molla_solvers::attachments::{
    Anchor, Attachment, BladeFrame, Config, LeafBody, Particle, Petiole, step_cpu_blade,
};

const MASS: f64 = 0.0001;
const DT: f64 = 1.0 / 60.0;

fn rest() -> BladeFrame {
    BladeFrame {
        direction: Vec3::X,
        normal: Vec3::Y,
        angular_velocity: Vec3::ZERO,
    }
}

fn leaf() -> LeafBody {
    LeafBody {
        frame: rest(),
        length: 0.1,
        width: 0.06,
        area: MASS * 10.0,
        camber: 0.004,
    }
}

fn case(strength: f64) -> (Attachment, Particle, Anchor) {
    let petiole = Petiole::from_frequency(MASS, leaf(), 0.008, 4.5, 4.0, 0.12);
    let a = Attachment {
        mass: MASS,
        stiffness: MASS * (std::f64::consts::TAU * 7.0).powi(2),
        damping: 0.0,
        strength,
        drag: 0.0002,
        radius: 0.002,
        petiole: Some(petiole),
    };
    let anchor = Anchor {
        position: Vec3::Y,
        velocity: Vec3::ZERO,
        valid: true,
    };
    let sag = Vec3::Y * (MASS * 9.81 / a.stiffness);
    let mut p = Particle::attached(anchor.position + Vec3::X * (0.008 + 0.05) - sag);
    p.leaf = Some(leaf());
    (a, p, anchor)
}

fn config(wind: Vec3) -> Config {
    Config {
        dt: DT,
        wind,
        ..Default::default()
    }
}

/// Mean angle between the blade and its rest direction over the last `average` steps.
fn mean_bend(wind: Vec3, steps: usize, average: usize) -> f64 {
    let (a, mut p, anchor) = case(1.0);
    let mut sum = 0.0;
    for step in 0..steps {
        step_cpu_blade(a, &mut p, anchor, rest(), config(wind)).unwrap();
        if step >= steps - average {
            sum += p.leaf.unwrap().frame.direction.angle_between(Vec3::X);
        }
    }
    sum / average as f64
}

#[test]
fn hinged_blade_at_rest_holds_its_twig_pose_and_weight() {
    let (a, mut p, anchor) = case(0.01);
    let start = p.position;
    for _ in 0..240 {
        step_cpu_blade(a, &mut p, anchor, rest(), config(Vec3::ZERO)).unwrap();
    }
    assert!(p.attached);
    assert!(p.position.distance(start) < 1e-9, "{:?}", p.position - start);
    assert!((p.force - Vec3::Y * (MASS * 9.81)).length() < 1e-9 * MASS * 9.81 + 1e-12);
}

#[test]
fn downwash_folds_the_blade_downstream_and_sheds_broadside_drag() {
    let wind = Vec3::new(0.0, -6.0, 0.0);
    let (a, mut p, anchor) = case(1.0);
    for _ in 0..180 {
        step_cpu_blade(a, &mut p, anchor, rest(), config(wind)).unwrap();
    }
    let broadside = 0.5 * 1.225 * leaf().area * 2.4 * 36.0;
    assert!(p.attached);
    assert!(p.leaf.unwrap().frame.direction.y < -0.5, "{:?}", p.leaf);
    assert!(p.air_force.length() < 0.5 * broadside, "{} vs {broadside}", p.air_force.length());
    let joint = p.force;
    assert!(joint.is_finite() && joint.length() > MASS * 9.81);
}

#[test]
fn stronger_air_bends_leaves_further() {
    let bends: Vec<f64> = [0.5, 2.0, 6.0]
        .map(|speed| mean_bend(Vec3::new(0.0, -speed, 0.0), 240, 60))
        .to_vec();
    assert!(bends[0] < bends[1] && bends[1] < bends[2], "{bends:?}");
    assert!(bends[0] < 0.2 && bends[2] > 0.8, "{bends:?}");
}

#[test]
fn shaking_twig_swings_the_blade_through_pivot_acceleration() {
    let (a, mut p, mut anchor) = case(1.0);
    let mut largest: f64 = 0.0;
    for step in 0..180 {
        let t = step as f64 * DT;
        let w = std::f64::consts::TAU * 4.5;
        anchor.position = Vec3::Y + Vec3::Y * (0.01 * (w * t).sin());
        anchor.velocity = Vec3::Y * (0.01 * w * (w * t).cos());
        step_cpu_blade(a, &mut p, anchor, rest(), config(Vec3::ZERO)).unwrap();
        largest = largest.max(p.leaf.unwrap().frame.direction.angle_between(Vec3::X));
        let arm = p.position - anchor.position;
        assert!((arm.length() - 0.058).abs() < 0.02, "{}", arm.length());
    }
    assert!(largest > 0.15, "{largest}");
}

#[test]
fn canopy_sheltering_scales_the_blade_air_force() {
    let wind = Vec3::new(1.0, -4.0, 0.5);
    let (open, start, anchor) = case(1.0);
    let mut sheltered = open;
    sheltered.petiole.as_mut().unwrap().sheltering = 0.3;
    let force = |a: Attachment| {
        let mut p = start;
        let short = Config {
            dt: 1e-4,
            ..config(wind)
        };
        step_cpu_blade(a, &mut p, anchor, rest(), short).unwrap();
        p.air_force
    };
    let (full, reduced) = (force(open), force(sheltered));
    assert!((reduced - full * 0.3).length() < 0.02 * full.length(), "{full:?} {reduced:?}");
    let mut invalid = open;
    invalid.petiole.as_mut().unwrap().sheltering = 1.5;
    let mut p = start;
    assert!(step_cpu_blade(invalid, &mut p, anchor, rest(), config(wind)).is_err());
}

#[test]
fn weak_petiole_releases_into_free_flight_continuously() {
    let wind = Vec3::new(0.0, -8.0, 0.0);
    let (a, mut p, anchor) = case(MASS * 9.81 * 4.0);
    let mut released = None;
    for step in 0..120 {
        let before = p;
        step_cpu_blade(a, &mut p, anchor, rest(), config(wind)).unwrap();
        if !p.attached {
            released = Some((step, before, p));
            break;
        }
    }
    let (_, before, after) = released.expect("leaf sheds in strong downwash");
    assert!(after.position.distance(before.position) < 0.2);
    assert!(after.force == Vec3::ZERO && after.leaf.is_some());
    let mut flying = after;
    step_cpu_blade(a, &mut flying, anchor, rest(), config(wind)).unwrap();
    assert!(!flying.attached && flying.position.y < after.position.y);
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires Vulkan"]
fn gpu_hinged_blades_match_cpu() {
    use molla_compute::Engine;
    use molla_solvers::attachments::GpuAttachments;
    let engine = Engine::new_gpu().unwrap();
    let (a, p, anchor) = case(1.0);
    let weak = Attachment {
        strength: MASS * 9.81 * 4.0,
        ..a
    };
    let mut sheltered = a;
    sheltered.petiole.as_mut().unwrap().sheltering = 0.3;
    let coefficients = vec![a, sheltered, a, weak];
    let mut cpu = vec![p; 4];
    let winds = vec![
        Vec3::ZERO,
        Vec3::new(1.5, -3.0, 0.7),
        Vec3::new(0.0, -1.0, 0.0),
        Vec3::new(0.0, -8.0, 0.0),
    ];
    let mut gpu = GpuAttachments::new(&engine, coefficients.clone(), &cpu).unwrap();
    let frames = vec![rest(); 4];
    for step in 0..90 {
        let t = step as f64 * DT;
        let mut anchors = vec![anchor; 4];
        anchors[2].velocity = Vec3::new(0.0, 0.0, 0.3 * (9.0 * t).cos());
        anchors[2].position = anchor.position + Vec3::Z * (0.3 / 9.0 * (9.0 * t).sin());
        for ((p, a), (anchor, wind)) in cpu
            .iter_mut()
            .zip(&coefficients)
            .zip(anchors.iter().zip(&winds))
        {
            step_cpu_blade(*a, p, *anchor, rest(), config(*wind)).unwrap();
        }
        gpu.step_spatial_blades(&engine, &anchors, config(Vec3::ZERO), &winds, &frames)
            .unwrap();
    }
    let actual = gpu.snapshot(&engine).unwrap();
    for (i, (g, c)) in actual.iter().zip(&cpu).enumerate() {
        assert_eq!(g.attached, c.attached, "leaf {i}");
        let (gl, cl) = (g.leaf.unwrap().frame, c.leaf.unwrap().frame);
        let scale = 1.0 + c.velocity.length();
        assert!(g.position.distance(c.position) < 2e-3 * scale, "leaf {i}: {g:?} {c:?}");
        assert!(gl.direction.distance(cl.direction) < 0.03, "leaf {i}");
        assert!(gl.normal.distance(cl.normal) < 0.03, "leaf {i}");
        if c.attached {
            assert!((g.force - c.force).length() < 0.05 * c.force.length() + 1e-5, "leaf {i}");
        }
    }
    assert!(!actual[3].attached && actual[0].attached);
}
