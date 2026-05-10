//! URDF importer.
//!
//! Supports links (with inertial) and joints (revolute, continuous,
//! prismatic, fixed). Visual + collision blocks are parsed but
//! ignored — wiring them into the narrow-phase pipeline is a
//! follow-up. Multi-DOF joints (planar, floating) are mapped to the
//! closest molla equivalent (`Free` for floating); `planar` returns
//! an error with a clear message.
//!
//! Conventions assumed:
//! - URDF `joint/origin` maps to the joint's `x_p` (joint frame in
//!   parent body). The child anchor `x_c` is identity — URDF
//!   anchors the child link's origin at the joint origin by spec.
//! - Continuous joints (`continuous` URDF type) are revolute with
//!   no joint limits.
//! - Inertial origin in the link's local frame becomes
//!   `BodyParams.com`. Inertia in the link's local frame becomes
//!   `BodyParams.inertia`.

use std::collections::HashMap;
use std::path::Path;

use urdf_rs::{JointType as UrdfJointType, Robot};

use crate::core::{BodyId, Error, JointId, Result};
use crate::math::{Mat3, Quat, Real, SpatialInertia, Transform, Vec3};
use crate::sim::{BodyParams, ModelBuilder};

/// Output of [`load_urdf`] / [`build_urdf`] — name → id maps so the
/// caller can look up bodies / joints by their URDF name string after
/// import.
#[derive(Debug, Default, Clone)]
pub struct UrdfBodies {
    /// `link_name → BodyId`.
    pub links: HashMap<String, BodyId>,
    /// `joint_name → JointId`.
    pub joints: HashMap<String, JointId>,
}

/// Parse a URDF file at `path` and append the robot to `builder`.
/// Returns name lookup tables.
pub fn load_urdf(path: &Path, builder: &mut ModelBuilder) -> Result<UrdfBodies> {
    let robot = urdf_rs::read_file(path)
        .map_err(|e| Error::Build(format!("urdf parse {}: {e}", path.display())))?;
    build_urdf(&robot, builder)
}

/// Parse a URDF source string and append to `builder`.
pub fn parse_urdf(xml: &str, builder: &mut ModelBuilder) -> Result<UrdfBodies> {
    let robot = urdf_rs::read_from_string(xml)
        .map_err(|e| Error::Build(format!("urdf parse: {e}")))?;
    build_urdf(&robot, builder)
}

/// Convert an already-parsed URDF [`Robot`] into bodies + joints on
/// the given builder.
pub fn build_urdf(robot: &Robot, builder: &mut ModelBuilder) -> Result<UrdfBodies> {
    let mut out = UrdfBodies::default();

    // Pass 1: links → bodies. URDF doesn't constrain the order, but
    // we want every link present before joints reference them.
    for link in &robot.links {
        let inertial = &link.inertial;
        let mass = inertial.mass.value as Real;
        let com = pose_to_transform(&inertial.origin).position;
        let inertia = inertia_from_urdf(&inertial.inertia);

        let id = builder.add_body(
            BodyParams::new()
                .with_spatial_inertia(SpatialInertia { mass, com, inertia })
                .with_pose(Transform::IDENTITY),
        );
        out.links.insert(link.name.clone(), id);
    }

    // Pass 2: joints. URDF guarantees `parent` and `child` are
    // valid link names; we resolve them via the map we just built.
    for joint in &robot.joints {
        let parent = out
            .links
            .get(&joint.parent.link)
            .copied()
            .ok_or_else(|| {
                Error::Build(format!(
                    "urdf joint {:?}: parent link {:?} not found",
                    joint.name, joint.parent.link
                ))
            })?;
        let child = out
            .links
            .get(&joint.child.link)
            .copied()
            .ok_or_else(|| {
                Error::Build(format!(
                    "urdf joint {:?}: child link {:?} not found",
                    joint.name, joint.child.link
                ))
            })?;

        let x_p = pose_to_transform(&joint.origin);
        let x_c = Transform::IDENTITY; // URDF anchors child origin at joint origin
        let axis = Vec3::new(joint.axis.xyz[0], joint.axis.xyz[1], joint.axis.xyz[2]);

        let id = match joint.joint_type {
            UrdfJointType::Revolute | UrdfJointType::Continuous => {
                builder.add_joint_revolute(parent, child, x_p, x_c, axis)?
            }
            UrdfJointType::Prismatic => {
                builder.add_joint_prismatic(parent, child, x_p, x_c, axis)?
            }
            UrdfJointType::Fixed => builder.add_joint_fixed(parent, child, x_p, x_c)?,
            UrdfJointType::Floating => builder.add_joint_free(parent, child, x_p, x_c)?,
            UrdfJointType::Planar => {
                return Err(Error::Build(format!(
                    "urdf joint {:?}: planar joints have no molla equivalent",
                    joint.name
                )));
            }
            UrdfJointType::Spherical => {
                return Err(Error::Build(format!(
                    "urdf joint {:?}: spherical (ball) joints not yet routed \
                     through the URDF importer (use ModelBuilder::add_joint_ball directly)",
                    joint.name
                )));
            }
        };
        out.joints.insert(joint.name.clone(), id);
    }

    Ok(out)
}

fn pose_to_transform(p: &urdf_rs::Pose) -> Transform {
    let pos = Vec3::new(p.xyz[0] as Real, p.xyz[1] as Real, p.xyz[2] as Real);
    // URDF rpy = roll/pitch/yaw about fixed axes (X, Y, Z).
    let r = p.rpy[0] as Real;
    let p_ = p.rpy[1] as Real;
    let y = p.rpy[2] as Real;
    let rot = Quat::from_euler(glam::EulerRot::XYZ, r, p_, y);
    Transform::new(pos, rot)
}

fn inertia_from_urdf(i: &urdf_rs::Inertia) -> Mat3 {
    // URDF inertia is the *upper triangular* of a symmetric 3×3.
    let ixx = i.ixx as Real;
    let ixy = i.ixy as Real;
    let ixz = i.ixz as Real;
    let iyy = i.iyy as Real;
    let iyz = i.iyz as Real;
    let izz = i.izz as Real;
    Mat3::from_cols(
        Vec3::new(ixx, ixy, ixz),
        Vec3::new(ixy, iyy, iyz),
        Vec3::new(ixz, iyz, izz),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two-link pendulum URDF, all revolute. Should produce 2
    /// bodies + 1 joint (between them).
    #[test]
    fn loads_two_link_revolute_chain() {
        let xml = r#"<?xml version="1.0"?>
<robot name="double_pendulum">
  <link name="base">
    <inertial>
      <mass value="0.5"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>
  <link name="link1">
    <inertial>
      <mass value="1.0"/>
      <origin xyz="0 -0.5 0" rpy="0 0 0"/>
      <inertia ixx="0.083" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.083"/>
    </inertial>
  </link>
  <joint name="hinge" type="revolute">
    <parent link="base"/>
    <child link="link1"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="10" velocity="3"/>
  </joint>
</robot>"#;
        let mut b = ModelBuilder::new();
        let bodies = parse_urdf(xml, &mut b).unwrap();
        assert_eq!(bodies.links.len(), 2);
        assert_eq!(bodies.joints.len(), 1);
        assert!(bodies.links.contains_key("base"));
        assert!(bodies.links.contains_key("link1"));
        assert!(bodies.joints.contains_key("hinge"));

        let model = b.build().unwrap();
        assert_eq!(model.body_count, 2);
        assert_eq!(model.joint_count, 1);
        assert_eq!(model.joint_coord_total, 1); // 1 revolute = 1 coord
        assert_eq!(model.joint_dof_total, 1);

        // Mass / inertia readback
        let masses = model.body_mass.host().unwrap();
        let base = bodies.links["base"].index();
        let link1 = bodies.links["link1"].index();
        assert!((masses[base] - 0.5).abs() < 1e-12);
        assert!((masses[link1] - 1.0).abs() < 1e-12);
    }

    /// Mixed joint types — fixed + revolute + prismatic.
    #[test]
    fn loads_mixed_joint_types() {
        let xml = r#"<?xml version="1.0"?>
<robot name="mix">
  <link name="a">
    <inertial><mass value="1"/><origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/></inertial>
  </link>
  <link name="b">
    <inertial><mass value="1"/><origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/></inertial>
  </link>
  <link name="c">
    <inertial><mass value="1"/><origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/></inertial>
  </link>
  <link name="d">
    <inertial><mass value="1"/><origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/></inertial>
  </link>
  <joint name="j_fix" type="fixed">
    <parent link="a"/><child link="b"/><origin xyz="1 0 0" rpy="0 0 0"/>
  </joint>
  <joint name="j_rev" type="continuous">
    <parent link="b"/><child link="c"/><origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
  </joint>
  <joint name="j_prs" type="prismatic">
    <parent link="c"/><child link="d"/><origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-1" upper="1" effort="0" velocity="0"/>
  </joint>
</robot>"#;
        let mut b = ModelBuilder::new();
        let _ = parse_urdf(xml, &mut b).unwrap();
        let model = b.build().unwrap();
        assert_eq!(model.joint_count, 3);
        // 1 fixed (0 dof) + 1 revolute (1 dof) + 1 prismatic (1 dof) = 2 dof total.
        assert_eq!(model.joint_dof_total, 2);
        assert_eq!(model.joint_coord_total, 2);
    }
}
