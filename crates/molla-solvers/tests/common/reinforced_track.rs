use molla_core::{BodyId, JointId, WorldId};
use molla_kernels::soft_rigid_contact::SoftRigidShapeGpu;
use molla_math::{Mat3, Quat, Real, SpatialInertia, Transform, Vec3};
use molla_sim::soft_belt::{
    RubberBeltMesh, RubberBeltParams, RubberCordParams, RubberLugParams,
    path::{BeltPath, BeltPulley},
};
use molla_sim::{BodyParams, ModelBuilder};

pub struct TrackRig {
    pub mesh: RubberBeltMesh,
    pub path: BeltPath,
    pub params: RubberBeltParams,
    pub pulleys: [BeltPulley; 3],
    pub lugs: usize,
    pub nodes: Vec<u32>,
    pub drive_joint: JointId,
    pub shapes: Vec<SoftRigidShapeGpu>,
    pub center: Vec3,
    pub installed: Vec<Vec3>,
}

pub fn primitive_gap(shape: &SoftRigidShapeGpu, pose: Transform, q: Vec3) -> Real {
    let rotation = Quat::from_xyzw(
        shape.rotation[0] as Real,
        shape.rotation[1] as Real,
        shape.rotation[2] as Real,
        shape.rotation[3] as Real,
    )
    .normalize();
    let center = Vec3::new(
        shape.position[0] as Real,
        shape.position[1] as Real,
        shape.position[2] as Real,
    );
    let data = Vec3::new(
        shape.data[0] as Real,
        shape.data[1] as Real,
        shape.data[2] as Real,
    );
    let p = rotation.conjugate() * (pose.rotation.conjugate() * (q - pose.position) - center);
    match shape.ids[0] {
        0 => p.length() - data.x,
        1 => {
            let delta = p.abs() - data;
            delta.max(Vec3::ZERO).length() + delta.max_element().min(0.0)
        }
        2 => {
            let radial = p.x.hypot(p.z) - data.x;
            let axial = p.y.abs() - data.y;
            radial.max(0.0).hypot(axial.max(0.0)) + radial.max(axial).min(0.0)
        }
        _ => panic!("unsupported track primitive"),
    }
}

fn cylinder(body: BodyId, x: Real, radius: Real, half_width: Real) -> SoftRigidShapeGpu {
    SoftRigidShapeGpu {
        position: [x as f32, 0.0, 0.0, 0.0],
        rotation: Quat::from_rotation_z(std::f64::consts::FRAC_PI_2)
            .to_array()
            .map(|v| v as f32),
        data: [radius as f32, half_width as f32, 0.0, 0.8],
        ids: [2, body.index() as u32, 0, 0],
    }
}

pub fn build_track(
    soft: &mut ModelBuilder,
    rigid: &mut ModelBuilder,
    carrier: BodyId,
    center: Vec3,
    segments: usize,
    width_cells: usize,
    prestrain: Real,
) -> TrackRig {
    let mut pulleys = [
        BeltPulley {
            center: Vec3::new(0.0, -0.52, -0.11),
            radius: 0.132,
        },
        BeltPulley {
            center: Vec3::new(0.0, 0.54, -0.11),
            radius: 0.132,
        },
        BeltPulley {
            center: Vec3::new(0.0, 0.405, 0.194),
            radius: 0.111,
        },
    ];
    let teeth = 12;
    let lugs = 56;
    for _ in 0..16 {
        let path = BeltPath::around_pulleys(&pulleys).unwrap();
        pulleys[2].radius = path.length() * teeth as Real / (std::f64::consts::TAU * lugs as Real);
    }
    let path = BeltPath::around_pulleys(&pulleys).unwrap();
    let params = RubberBeltParams {
        radius: path.length() / std::f64::consts::TAU,
        segments,
        width_cells,
        thickness_cells: 2,
        ..Default::default()
    };
    let mesh = RubberBeltMesh::circular(params)
        .unwrap()
        .with_drive_lugs(RubberLugParams {
            count: lugs,
            segment_span: segments / lugs / 2,
            width_start: width_cells / 4,
            width_cells: width_cells / 2,
            depth: 0.012,
            depth_cells: 2,
        })
        .unwrap();

    let nodes = mesh
        .append_reinforced_to(
            soft,
            WorldId(0),
            center,
            RubberCordParams {
                prestrain,
                ..Default::default()
            },
        )
        .unwrap()
        .nodes;
    let installed = mesh.installed_positions(&path, 0.0, center).unwrap();
    let mut shapes = Vec::new();
    let mut drive = BodyId::NONE;
    let mut drive_joint = None;
    for (i, pulley) in pulleys.iter().enumerate() {
        let wheel = rigid.add_body(
            BodyParams::new()
                .with_pose(Transform::from_translation(center + pulley.center))
                .with_spatial_inertia(SpatialInertia {
                    mass: 6.0,
                    com: Vec3::ZERO,
                    inertia: Mat3::from_diagonal(Vec3::splat(0.05)),
                }),
        );
        let joint = rigid
            .add_joint_revolute(
                carrier,
                wheel,
                Transform::from_translation(center + pulley.center),
                Transform::IDENTITY,
                Vec3::X,
            )
            .unwrap();
        if i == 2 {
            drive = wheel;
            drive_joint = Some(joint);
            let mut core = cylinder(wheel, 0.0, pulley.radius - 0.024, 0.044);
            core.data[3] = 0.0;
            shapes.push(core);
        } else {
            shapes.push(cylinder(
                wheel,
                0.0,
                pulley.radius - params.thickness * 0.5 - 0.012,
                0.05,
            ));
            for x in [-0.075, 0.075] {
                shapes.push(cylinder(
                    wheel,
                    x,
                    pulley.radius - params.thickness * 0.5,
                    0.025,
                ));
            }
        }
    }
    for y in [-0.28, 0.005, 0.29] {
        let wheel_center = center + Vec3::new(0.0, y, -0.18);
        let wheel = rigid.add_body(
            BodyParams::new()
                .with_pose(Transform::from_translation(wheel_center))
                .with_spatial_inertia(SpatialInertia {
                    mass: 3.0,
                    com: Vec3::ZERO,
                    inertia: Mat3::from_diagonal(Vec3::splat(0.01)),
                }),
        );
        rigid
            .add_joint_revolute(
                carrier,
                wheel,
                Transform::from_translation(wheel_center),
                Transform::IDENTITY,
                Vec3::X,
            )
            .unwrap();
        for x in [-0.075, 0.075] {
            shapes.push(cylinder(wheel, x, 0.052, 0.025));
        }
        shapes.push(cylinder(wheel, 0.0, 0.040, 0.05));
    }
    let mut gap_angle = 0.0;
    let mut nearest = Real::INFINITY;
    for lug in 0..lugs {
        let sample = path
            .sample((lug as Real + 0.75) * path.length() / lugs as Real)
            .unwrap();
        let radial = sample.position - pulleys[2].center;
        let error = (radial.length() - pulleys[2].radius).abs();
        if error < nearest {
            nearest = error;
            gap_angle = radial.z.atan2(radial.y);
        }
    }
    for t in 0..teeth {
        let angle = gap_angle + std::f64::consts::TAU * t as Real / teeth as Real;
        let radius = pulleys[2].radius - 0.0181;
        shapes.push(SoftRigidShapeGpu {
            position: [
                0.0,
                (radius * angle.cos()) as f32,
                (radius * angle.sin()) as f32,
                0.0,
            ],
            rotation: Quat::from_rotation_x(angle).to_array().map(|v| v as f32),
            data: [0.043, 0.0079, 0.006, 0.0],
            ids: [1, drive.index() as u32, 0, 0],
        });
    }

    TrackRig {
        mesh,
        path,
        params,
        pulleys,
        lugs,
        nodes,
        drive_joint: drive_joint.unwrap(),
        shapes,
        center,
        installed,
    }
}
