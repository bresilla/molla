//! Emitters and receivers: packets carried between sensor frames.
//!
//! A [`RadioMedium`] queues packets sent by emitters and hands each one to the
//! receivers that can hear it when [`RadioMedium::deliver`] is called with the
//! current poses. Radio reaches every receiver in range; infra-red also needs
//! each end inside the other's aperture and a clear line of sight, which the
//! host answers (e.g. with a ray query); serial ignores distance, as a cable.

use molla_math::{Real, Transform, Vec3};

/// How a link carries packets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RadioKind {
    #[default]
    Radio,
    Infrared,
    Serial,
}

/// Channel that every emitter and receiver shares.
pub const ALL_CHANNELS: i32 = -1;

/// Emitter or receiver settings. `range` is in metres (infinite for none),
/// `aperture` the full cone angle around the frame's +X (infra-red only).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RadioEndpoint {
    pub kind: RadioKind,
    pub channel: i32,
    pub range: Real,
    pub aperture: Real,
}

impl Default for RadioEndpoint {
    fn default() -> Self {
        Self {
            kind: RadioKind::Radio,
            channel: 0,
            range: Real::INFINITY,
            aperture: std::f64::consts::TAU,
        }
    }
}

/// One packet handed to a receiver.
#[derive(Debug, Clone, PartialEq)]
pub struct Delivery {
    pub receiver: usize,
    pub emitter: usize,
    pub channel: i32,
    pub data: Vec<u8>,
    /// Inverse-square strength, 1 at one metre.
    pub signal_strength: Real,
    /// Unit direction towards the emitter in receiver axes.
    pub direction: Vec3,
}

/// Queued packets between emitter and receiver endpoints.
#[derive(Debug, Clone, Default)]
pub struct RadioMedium {
    pub emitters: Vec<RadioEndpoint>,
    pub receivers: Vec<RadioEndpoint>,
    queue: Vec<(usize, i32, Vec<u8>)>,
}

fn channels_match(a: i32, b: i32) -> bool {
    a == ALL_CHANNELS || b == ALL_CHANNELS || a == b
}

fn within_aperture(pose: &Transform, towards: Vec3, aperture: Real) -> bool {
    aperture >= std::f64::consts::TAU
        || (pose.rotation * Vec3::X).angle_between(towards) <= 0.5 * aperture
}

impl RadioMedium {
    pub fn new(emitters: Vec<RadioEndpoint>, receivers: Vec<RadioEndpoint>) -> Self {
        Self {
            emitters,
            receivers,
            queue: Vec::new(),
        }
    }

    /// Queue a packet from `emitter` on its current channel.
    pub fn send(&mut self, emitter: usize, data: Vec<u8>) {
        if let Some(endpoint) = self.emitters.get(emitter) {
            self.queue.push((emitter, endpoint.channel, data));
        }
    }

    pub fn pending(&self) -> usize {
        self.queue.len()
    }

    /// Deliver every queued packet to the receivers that hear it, given the
    /// endpoints' world poses and a line-of-sight test for infra-red.
    pub fn deliver(
        &mut self,
        emitter_poses: &[Transform],
        receiver_poses: &[Transform],
        line_of_sight: &mut dyn FnMut(Vec3, Vec3) -> bool,
    ) -> Vec<Delivery> {
        let mut out = Vec::new();
        for (emitter, channel, data) in self.queue.drain(..) {
            let (Some(tx), Some(tx_pose)) =
                (self.emitters.get(emitter), emitter_poses.get(emitter))
            else {
                continue;
            };
            for (receiver, (rx, rx_pose)) in self.receivers.iter().zip(receiver_poses).enumerate() {
                if rx.kind != tx.kind || !channels_match(channel, rx.channel) {
                    continue;
                }
                let offset = tx_pose.position - rx_pose.position;
                let distance = offset.length();
                if tx.kind != RadioKind::Serial && distance > tx.range.min(rx.range) {
                    continue;
                }
                let towards = if distance > 0.0 {
                    offset / distance
                } else {
                    Vec3::X
                };
                if tx.kind == RadioKind::Infrared
                    && (!within_aperture(tx_pose, -towards, tx.aperture)
                        || !within_aperture(rx_pose, towards, rx.aperture)
                        || !line_of_sight(tx_pose.position, rx_pose.position))
                {
                    continue;
                }
                out.push(Delivery {
                    receiver,
                    emitter,
                    channel,
                    data: data.clone(),
                    signal_strength: 1.0 / distance.max(1e-3).powi(2),
                    direction: rx_pose.rotation.inverse() * towards,
                });
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(x: Real) -> Transform {
        Transform::new(Vec3::new(x, 0.0, 0.0), molla_math::Quat::IDENTITY)
    }

    #[test]
    fn radio_reaches_receivers_on_its_channel_within_range() {
        let tx = RadioEndpoint {
            range: 50.0,
            ..Default::default()
        };
        let receivers = vec![
            RadioEndpoint::default(),
            RadioEndpoint {
                channel: 2,
                ..Default::default()
            },
            RadioEndpoint {
                channel: ALL_CHANNELS,
                ..Default::default()
            },
            RadioEndpoint::default(),
        ];
        let mut medium = RadioMedium::new(vec![tx], receivers);
        medium.send(0, b"hello".to_vec());
        let poses = [at(10.0), at(10.0), at(20.0), at(100.0)];
        let out = medium.deliver(&[at(0.0)], &poses, &mut |_, _| true);
        let heard: Vec<usize> = out.iter().map(|d| d.receiver).collect();
        assert_eq!(heard, [0, 2]);
        assert!((out[0].signal_strength - 0.01).abs() < 1e-12);
        assert!((out[0].direction - Vec3::NEG_X).length() < 1e-12);
        assert_eq!(out[0].data, b"hello");
        assert_eq!(medium.pending(), 0);
    }

    #[test]
    fn infrared_needs_aperture_and_line_of_sight_and_serial_ignores_range() {
        let ir = RadioEndpoint {
            kind: RadioKind::Infrared,
            range: 10.0,
            aperture: 0.5,
            ..Default::default()
        };
        let facing_back = Transform::new(
            Vec3::new(5.0, 0.0, 0.0),
            molla_math::Quat::from_rotation_z(std::f64::consts::PI),
        );
        let mut medium = RadioMedium::new(vec![ir], vec![ir, ir]);
        medium.send(0, vec![1]);
        let out = medium.deliver(&[at(0.0)], &[facing_back, at(5.0)], &mut |_, _| true);
        assert_eq!(out.iter().map(|d| d.receiver).collect::<Vec<_>>(), [0]);
        medium.send(0, vec![1]);
        assert!(
            medium
                .deliver(&[at(0.0)], &[facing_back, at(5.0)], &mut |_, _| false)
                .is_empty()
        );

        let serial = RadioEndpoint {
            kind: RadioKind::Serial,
            range: 1.0,
            ..Default::default()
        };
        let mut wired = RadioMedium::new(vec![serial], vec![serial, RadioEndpoint::default()]);
        wired.send(0, vec![7]);
        let out = wired.deliver(&[at(0.0)], &[at(1000.0), at(1.0)], &mut |_, _| false);
        assert_eq!(out.iter().map(|d| d.receiver).collect::<Vec<_>>(), [0]);
    }
}
