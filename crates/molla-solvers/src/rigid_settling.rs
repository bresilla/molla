use std::collections::{BTreeMap, BTreeSet};

use molla_core::{Error, Result};
use molla_math::{Real, SpatialVector};
use molla_sim::joint_control::JointAxis;
use molla_sim::runtime::{BodyActivationStamp, BodyHandle, RigidScene};

use super::RigidWorld;
use crate::rigid_islands::RigidIsland;
use crate::wheel_forces::WheelForces;

/// CPU automatic sleep thresholds in metres/second, radians/second and seconds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SleepSettings {
    pub linear_speed: Real,
    pub angular_speed: Real,
    pub quiet_time: Real,
}

impl Default for SleepSettings {
    fn default() -> Self {
        Self {
            linear_speed: 1.0e-4,
            angular_speed: 1.0e-4,
            quiet_time: 1.0,
        }
    }
}

struct QuietWindow {
    island: RigidIsland,
    elapsed: Real,
    watched: BTreeMap<BodyHandle, BodyActivationStamp>,
    wheels: (std::sync::Arc<()>, u64),
}

#[derive(Default)]
pub(super) struct Settling {
    settings: Option<SleepSettings>,
    windows: BTreeMap<BodyHandle, QuietWindow>,
}

impl Settling {
    pub(super) fn clear_windows(&mut self) {
        self.windows.clear();
    }

    pub(super) fn reset_bodies(&mut self, bodies: &[BodyHandle]) {
        self.windows
            .retain(|_, window| !bodies.iter().any(|body| window.watched.contains_key(body)));
    }

    pub(super) fn retain_sleep_eligibility(
        &mut self,
        scene: &RigidScene,
        wheels: &WheelForces,
        island: &RigidIsland,
    ) {
        let Some(settings) = self.settings else {
            return;
        };
        let watched = island
            .bodies
            .iter()
            .chain(&island.boundaries)
            .map(|&body| {
                (
                    body,
                    scene.activation_stamp(body).expect("wake island body"),
                )
            })
            .collect();
        self.windows.insert(
            island.bodies[0],
            QuietWindow {
                island: island.clone(),
                elapsed: settings.quiet_time,
                watched,
                wheels: wheels.activation_stamp(),
            },
        );
    }

    pub(super) fn changed_inputs(
        &self,
        scene: &RigidScene,
        wheels: &WheelForces,
    ) -> BTreeSet<BodyHandle> {
        self.windows
            .iter()
            .filter_map(|(&root, window)| {
                (!wheels.matches_activation(&window.wheels)
                    || window
                        .watched
                        .iter()
                        .any(|(&body, stamp)| scene.activation_stamp(body).as_ref() != Some(stamp)))
                .then_some(root)
            })
            .collect()
    }
}

impl RigidWorld {
    /// Automatic sleep is disabled by default. Disabling it retains explicit sleepers.
    pub fn set_sleep_settings(&mut self, settings: Option<SleepSettings>) -> Result<()> {
        if settings.is_some_and(|s| {
            !s.linear_speed.is_finite()
                || s.linear_speed < 0.0
                || !s.angular_speed.is_finite()
                || s.angular_speed < 0.0
                || !s.quiet_time.is_finite()
                || s.quiet_time <= 0.0
        }) {
            return Err(Error::Solver("invalid automatic sleep settings".into()));
        }
        self.settling.settings = settings;
        self.settling.clear_windows();
        Ok(())
    }

    pub fn sleep_settings(&self) -> Option<SleepSettings> {
        self.settling.settings
    }

    pub(super) fn finish_settling(&mut self, dt: Real, changed: &BTreeSet<BodyHandle>) {
        if let Some(settings) = self.settling.settings
            && let Err(error) = self.settle(dt, changed, settings)
        {
            self.settling.clear_windows();
            tracing::warn!(%error, "automatic sleep skipped");
        }
    }

    fn settle(
        &mut self,
        dt: Real,
        changed: &BTreeSet<BodyHandle>,
        settings: SleepSettings,
    ) -> Result<()> {
        let islands = self.islands()?;
        let driven = driven_bodies(&self.scene)?;
        let velocities = self.scene.state().body_qd.host()?;
        let forces = self.scene.state().body_f.host()?;
        let external = self.scene.control().body_external_f.host()?;
        let mut windows = BTreeMap::new();
        let mut ready = Vec::new();
        for island in islands.groups() {
            let root = island.bodies[0];
            if self.is_sleeping(root) {
                continue;
            }
            let quiet = island.bodies.iter().all(|&body| {
                let index = self.scene.body_index(body).unwrap().index();
                let v = velocities[index];
                v.linear.length() <= settings.linear_speed
                    && v.angular.length() <= settings.angular_speed
                    && forces[index] == SpatialVector::ZERO
                    && external[index] == SpatialVector::ZERO
                    && !driven.contains(&body)
                    && !self.wheels.transitioning(body)
            }) && island.boundaries.iter().all(|&body| {
                self.scene
                    .body_velocity(body)
                    .is_ok_and(|v| v == SpatialVector::ZERO)
            });
            if !quiet {
                continue;
            }
            let previous = self
                .settling
                .windows
                .get(&root)
                .filter(|window| window.island == *island && !changed.contains(&root))
                .map_or(0.0, |window| window.elapsed);
            let elapsed = previous + dt;
            if elapsed >= settings.quiet_time {
                ready.push(root);
            }
            windows.insert(
                root,
                QuietWindow {
                    island: island.clone(),
                    elapsed,
                    watched: BTreeMap::new(),
                    wheels: self.wheels.activation_stamp(),
                },
            );
        }
        if !ready.is_empty() {
            self.sleep_bodies(&ready)?;
        }
        windows.retain(|&root, window| {
            if self.is_sleeping(root) {
                return false;
            }
            window.watched = window
                .island
                .bodies
                .iter()
                .chain(&window.island.boundaries)
                .map(|&body| (body, self.scene.activation_stamp(body).unwrap()))
                .collect();
            true
        });
        self.settling.windows = windows;
        Ok(())
    }
}

fn commanded(mode: u32, velocity: Real, position: Real, damping: Real, cap: Real) -> bool {
    cap > 0.0
        && ((matches!(mode, 2 | 3) && velocity != 0.0 && damping > 0.0)
            || (mode == 4 && position != 0.0))
}

fn driven_bodies(scene: &RigidScene) -> Result<BTreeSet<BodyHandle>> {
    let model = scene.model();
    let control = scene.control();
    let offsets = model.joint_dof_offset.host()?;
    let children = model.joint_child.host()?;
    let modes = control.joint_target_mode.host()?;
    let velocities = control.joint_target_vel.host()?;
    let positions = control.joint_target_pos.host()?;
    let damping = control.joint_target_kd.host()?;
    let caps = control.joint_motor_max_force.host()?;
    let effort = control.joint_force.host()?;
    let mut bodies = BTreeSet::new();
    for (joint, range) in offsets.windows(2).enumerate() {
        if (range[0] as usize..range[1] as usize).any(|i| {
            effort[i] != 0.0
                || commanded(modes[i], velocities[i], positions[i], damping[i], caps[i])
        }) {
            if let Some(body) = scene.body_handle(molla_core::BodyId(children[joint] as u32)) {
                bodies.insert(body);
            }
        }
    }
    for (id, joint) in scene.loop_joints() {
        if !scene.joint_settings(id).is_some_and(|s| s.enabled) {
            continue;
        }
        let driven = JointAxis::ALL.iter().any(|&axis| {
            scene
                .joint_axis_settings(id, axis)
                .ok()
                .and_then(|s| s.motor)
                .is_some_and(|motor| {
                    commanded(
                        motor.mode as u32,
                        motor.target_velocity,
                        motor.target_position,
                        motor.damping,
                        motor.max_force,
                    )
                })
        });
        if driven {
            bodies.insert(joint.child);
            bodies.extend(joint.parent);
        }
    }
    Ok(bodies)
}
