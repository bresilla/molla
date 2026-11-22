use crate::{
    apple_tree_sim::{self, DT, Tree},
    leafy_model::{self, Leaf, random},
    orchard_model,
};
use molla_compute::Engine;
use molla_core::Result;
use molla_math::Vec3;
use molla_solvers::{
    airflow::{
        AirConfig, AirForces, AirState, CylinderDrag, Diagnostics, FlightReference, FlightSegment,
        GpuAir, Grid, MotionLimits, Quadrotor, QuadrotorConfig,
    },
    attachments::{
        Anchor, Attachment, BladeFrame, Config, GpuAttachments, LeafBody, Particle, Petiole,
    },
    strands::GpuStrands,
};

pub const AIR_DT: f64 = DT * 4.0;
pub const PHYSICS_HZ: usize = 60;
pub const DOWNWASH_LEVEL: f64 = 60.0;
#[cfg(test)]
pub const DRONE_MASS_KG: f64 = 2.0 + (8.0 - 2.0) * DOWNWASH_LEVEL / 100.0;
/// Abscission strength scale; reproduces the previous mission shed count with petiole leaves.
pub const LEAF_STRENGTH_SCALE: f64 = 1.25;
/// Range of petiole bending frequencies of attached leaves, in hertz.
pub const PETIOLE_BENDING_HZ: [f64; 2] = [2.5, 4.0];
/// Abscission strength scale while the unloaded model settles into its loaded pose.
const SETTLING_STRENGTH_SCALE: f64 = 1e9;
/// Longest settling time before initialization fails, in seconds.
const SETTLING_LIMIT_S: f64 = 24.0;
/// Fraction of the isolated-plate air force on a leaf inside the canopy.
pub const PETIOLE_SHELTERING: f64 = 0.3;
/// Petiole bending-to-torsion stiffness ratio EI/GJ.
pub const PETIOLE_TWIST_TO_BEND: f64 = 4.0;
/// Structural damping of the petiole as a fraction of critical.
pub const PETIOLE_DAMPING_RATIO: f64 = 0.12;
pub const EXTERIOR_WAKE_LENGTH: f64 = 3.0;
pub const TURBULENCE_STRENGTH: f64 = 2.2;
pub const FLIGHT_SECONDS: f64 = 78.0;
const LEG_SECONDS: f64 = 30.0;

/// Blade bodies, elastic petioles and statically loaded start poses for every leaf.
fn hold_leaves(canopy: &mut leafy_model::Canopy) {
    for (i, ((p, leaf), a)) in canopy
        .particles
        .iter_mut()
        .zip(&canopy.leaves)
        .zip(&mut canopy.attachments)
        .enumerate()
    {
        let direction = leaf.released_direction.normalize();
        let normal =
            (leaf.released_normal - direction * leaf.released_normal.dot(direction)).normalize();
        let body = LeafBody {
            frame: BladeFrame {
                direction,
                normal,
                angular_velocity: Vec3::ZERO,
            },
            length: leaf.length,
            width: leaf.length * 0.6,
            area: a.mass * (8.0 + 5.0 * random(i, 1901)),
            camber: leaf.length * (0.03 + 0.05 * random(i, 1902)),
        };
        let [low, high] = PETIOLE_BENDING_HZ;
        a.petiole = Some(Petiole {
            sheltering: PETIOLE_SHELTERING,
            ..Petiole::from_frequency(
                a.mass,
                body,
                leaf.offset.length() - 0.5 * leaf.length,
                low + (high - low) * random(i, 1903),
                PETIOLE_TWIST_TO_BEND,
                PETIOLE_DAMPING_RATIO,
            )
        });
        p.position += Config::default().gravity * (a.mass / a.stiffness);
        p.leaf = Some(body);
    }
}

fn air_grid() -> Result<Grid> {
    let z_cells = ((2.0 * (orchard_model::ROW_HALF_LENGTH + 3.6) / 0.15).ceil() as usize)
        .next_multiple_of(16);
    Grid::new(
        [64, 32, z_cells],
        0.15,
        Vec3::new(-4.8, 0.0, -(z_cells as f64) * 0.075),
    )
}

pub struct LeafTrack {
    pub index: usize,
    pub samples: Vec<[f64; 7]>,
}

pub struct Orchard {
    pub spray: Option<crate::orchard_spray::Spray>,
    pub crosswind: Option<crate::crosswind::Crosswind>,
    pub tree: Tree,
    pub drone: Quadrotor,
    pub leaves: Vec<Leaf>,
    pub particles: Vec<Particle>,
    pub coefficients: Vec<Attachment>,
    pub coupling: bool,
    pub motors: bool,
    pub leaf_strength_scale: f64,
    pub downwash_level: f64,
    pub wood_flexibility: f64,
    pub autopilot: bool,
    pub target: Vec3,
    pub boundary_impulse: Vec3,
    pub peak_air_force: f64,
    pub peak_branch_displacement: f64,
    pub peak_trunk_displacement: f64,
    pub peak_limb_displacement: f64,
    pub tracer_points: Vec<Vec3>,
    pub tracer_velocities: Vec<Vec3>,
    pub air_paths: Vec<Vec<Vec3>>,
    pub nearest_rotor_leaf: f64,
    pub peak_shedding_wind: f64,
    pub air_diagnostics: Option<Diagnostics>,
    pub frames: usize,
    pub stage_seconds: [f64; 11],
    pub physics_substeps: usize,
    pub settling_seconds: f64,
    pub settling_speed: f64,
    pub ambient_wind: bool,
    pub visited_canopy_sides: [bool; 2],
    pub tree_overflight_distances: Vec<f64>,
    pub flight_trace: Vec<[f64; 4]>,
    pub flight_dynamics: Vec<[f64; 12]>,
    pub peak_leaf_spin: f64,
    pub maximum_leaf_travel: f64,
    pub boundary_angular_impulse: Vec3,
    pub leaf_tracks: Vec<LeafTrack>,
    pub leaf_relaunches: usize,
    pub airborne_rising_steps: usize,
    pub ground_rolling_steps: usize,
    pub relaunched_leaves: std::collections::BTreeSet<usize>,
    grounded_since_launch: Vec<bool>,
    release_positions: Vec<Option<Vec3>>,
    reference_centers: std::collections::BTreeMap<u64, Vec3>,
    trunk_cells: std::collections::BTreeSet<u64>,
    limb_cells: std::collections::BTreeSet<u64>,
    flight_height: f64,
    air: GpuAir,
    grid: Grid,
    forces: AirForces,
    attachments: GpuAttachments,
}

impl Orchard {
    pub fn configured(
        engine: &Engine,
        ambient_wind: bool,
        downwash_level: f64,
        wood_flexibility: f64,
    ) -> Result<Self> {
        if !downwash_level.is_finite() || !(0.0..=100.0).contains(&downwash_level) {
            return Err(molla_core::Error::Build(
                "downwash level must be between 0 and 100".into(),
            ));
        }
        let mut canopy = orchard_model::build_with_flexibility(wood_flexibility)?;
        let trunk_cells = canopy.tree.trunk_cells.iter().copied().collect();
        let limb_cells = canopy.tree.limb_cells.iter().flatten().copied().collect();
        hold_leaves(&mut canopy);
        let mut attachments =
            GpuAttachments::new(engine, canopy.attachments.clone(), &canopy.particles)?;
        let cells = canopy.tree.model.fibers().len();
        let mut tree = Tree::from_built_with_limits(
            engine,
            canopy.tree,
            molla_solvers::strands::GpuStrandLimits {
                contact_pairs: cells * (cells - 1) / 2,
                ..apple_tree_sim::gpu_limits()
            },
        )?;
        tree.retained_subdivisions = 64;
        tree.coarsen_batches = 120;
        tree.config.dt = 1.0 / PHYSICS_HZ as f64;
        let grid = air_grid()?;
        let mut config = AirConfig::new(grid);
        config.exterior_decay_length = EXTERIOR_WAKE_LENGTH;
        if ambient_wind {
            config.ambient = Vec3::new(1.2, 0.0, 0.35);
            config.gust_strength = 0.65;
            config.turbulence_strength = TURBULENCE_STRENGTH;
        }
        let air = GpuAir::new(engine, &AirState::new(config)?)?;
        attachments.couple_air(engine, &air.coupling())?;
        let flight_height = canopy
            .particles
            .iter()
            .map(|p| p.position.y)
            .fold(0.0, f64::max)
            + 0.55;
        let target = flight_target(0.0, flight_height);
        let mut drone = Quadrotor::new(
            target,
            QuadrotorConfig {
                mass: 2.0 + (8.0 - 2.0) * downwash_level / 100.0,
                ..Default::default()
            },
        )?;
        let mut floor =
            molla_sim::runtime::ColliderDesc::new(molla_sim::runtime::ColliderGeometry::Cuboid {
                half_extents: Vec3::new(100.0, 0.1, 100.0),
            });
        floor.pose.position.y = -0.1;
        drone.world.scene.insert_collider(floor)?;
        let rotor_frames = drone.rotor_frames()?;
        let tracer_points: Vec<_> = (0..12)
            .map(|i| {
                rotor_frames[i % 4].position
                    + Vec3::new(
                        (random(i, 801) - 0.5) * 0.35,
                        0.06 + 0.15 * random(i, 802),
                        (random(i, 803) - 0.5) * 0.35,
                    )
            })
            .collect();
        let reference_centers = tree
            .model
            .fibers()
            .iter()
            .map(|f| (f.id, f.center()))
            .collect();
        let leaf_count = canopy.particles.len();
        Ok(Self {
            spray: None,
            crosswind: None,
            tree,
            drone,
            release_positions: vec![None; canopy.particles.len()],
            leaves: canopy.leaves,
            particles: canopy.particles,
            coefficients: canopy.attachments,
            coupling: true,
            motors: true,
            leaf_strength_scale: LEAF_STRENGTH_SCALE,
            downwash_level,
            wood_flexibility,
            autopilot: true,
            target,
            boundary_impulse: Vec3::ZERO,
            peak_air_force: 0.0,
            peak_branch_displacement: 0.0,
            peak_trunk_displacement: 0.0,
            peak_limb_displacement: 0.0,
            tracer_velocities: vec![Vec3::ZERO; tracer_points.len()],
            air_paths: vec![Vec::new(); tracer_points.len()],
            tracer_points,
            nearest_rotor_leaf: f64::INFINITY,
            peak_shedding_wind: 0.0,
            air_diagnostics: None,
            frames: 0,
            stage_seconds: [0.0; 11],
            physics_substeps: PHYSICS_HZ / 60,
            settling_seconds: 0.0,
            settling_speed: 0.0,
            ambient_wind,
            visited_canopy_sides: [false; 2],
            tree_overflight_distances: vec![f64::INFINITY; orchard_model::TREE_COUNT],
            flight_trace: Vec::new(),
            flight_dynamics: Vec::new(),
            peak_leaf_spin: 0.0,
            maximum_leaf_travel: 0.0,
            boundary_angular_impulse: Vec3::ZERO,
            leaf_tracks: Vec::new(),
            leaf_relaunches: 0,
            airborne_rising_steps: 0,
            ground_rolling_steps: 0,
            relaunched_leaves: std::collections::BTreeSet::new(),
            grounded_since_launch: vec![false; leaf_count],
            reference_centers,
            trunk_cells,
            limb_cells,
            flight_height,
            air,
            grid,
            forces: AirForces::new(grid),
            attachments,
        })
    }

    pub fn attached(&self) -> usize {
        self.particles.iter().filter(|p| p.attached).count()
    }

    pub fn enable_spraying(&mut self) -> Result<()> {
        self.spray = Some(crate::orchard_spray::Spray::new(
            self.drone.pose()?.position,
            self.flight_height + 0.65,
            self.particles.len(),
        ));
        Ok(())
    }

    /// Blow a `speed` m/s crosswind once the spray mission reaches the second row.
    pub fn enable_crosswind(&mut self, speed: f64) -> Result<()> {
        let (breeze, gusts) = self.air.background();
        self.crosswind = Some(crate::crosswind::Crosswind::new(breeze, gusts, speed)?);
        Ok(())
    }

    pub fn time(&self) -> f64 {
        self.tree.time
    }

    pub fn escaped_leaves(&self) -> usize {
        self.particles
            .iter()
            .filter(|p| {
                !p.attached
                    && (!p.position.cmpge(self.grid.origin()).all()
                        || !p.position.cmple(self.grid.upper()).all())
            })
            .count()
    }

    pub fn set_physics_hz(&mut self, hz: usize) -> Result<()> {
        if ![60, 120, 240, 480].contains(&hz)
            || self.frames != 0
            || self.time() != 0.0
            || self.settling_seconds != 0.0
        {
            return Err(molla_core::Error::Build(
                "physics rate must be 60, 120, 240 or 480 Hz and set before stepping".into(),
            ));
        }
        self.physics_substeps = hz / 60;
        self.tree.config.dt = 1.0 / hz as f64;
        Ok(())
    }

    fn anchors(&mut self) -> Vec<Anchor> {
        self.leaves
            .iter_mut()
            .zip(&self.particles)
            .map(|(leaf, p)| {
                if self.tree.model.fiber(leaf.owner).is_none()
                    && let Some(f) = self
                        .tree
                        .model
                        .fibers()
                        .iter()
                        .find(|f| f.rest == leaf.rest && f.contains_local(leaf.anchor))
                {
                    leaf.owner = f.id;
                }
                if let Some(f) = self.tree.model.fiber(leaf.owner) {
                    let position = f.motion.pose.transform_point(leaf.anchor + leaf.offset);
                    if p.attached {
                        leaf.released_direction = f.motion.pose.rotation * leaf.direction;
                        leaf.released_normal = f.motion.pose.rotation * leaf.normal;
                    }
                    Anchor {
                        position,
                        velocity: f.motion.linear_velocity
                            + f.motion.angular_velocity.cross(position - f.center()),
                        valid: true,
                    }
                } else {
                    Anchor {
                        position: p.position,
                        velocity: p.velocity,
                        valid: false,
                    }
                }
            })
            .collect()
    }

    fn react_to_air(&mut self, point: Vec3, force: Vec3) -> Result<()> {
        let inner = self.grid.origin() + Vec3::splat(self.grid.spacing());
        let outer = self.grid.upper() - Vec3::splat(self.grid.spacing());
        if point.cmpge(inner).all() && point.cmple(outer).all() {
            self.forces
                .add(point, force * (self.tree.config.dt / AIR_DT))?;
        } else {
            self.boundary_impulse += force * self.tree.config.dt;
        }
        Ok(())
    }

    fn microstep(&mut self, engine: &Engine, flying: bool) -> Result<()> {
        let mut stamp = std::time::Instant::now();
        let dt = self.tree.config.dt;
        let anchors = self.anchors();
        let mut wood_points = Vec::new();
        for f in self.tree.model.fibers() {
            let lo = f
                .shape
                .vertices()
                .iter()
                .map(|p| p.z)
                .fold(f64::INFINITY, f64::min);
            let hi = f
                .shape
                .vertices()
                .iter()
                .map(|p| p.z)
                .fold(f64::NEG_INFINITY, f64::max);
            let radius = f
                .shape
                .vertices()
                .iter()
                .map(|p| p.truncate().length())
                .fold(0.0, f64::max);
            let axis = f.motion.pose.rotation * Vec3::Z;
            for side in [-1.0, 1.0] {
                let point = f.center() + axis * ((hi - lo) / 12.0f64.sqrt() * side);
                let velocity =
                    f.motion.linear_velocity + f.motion.angular_velocity.cross(point - f.center());
                wood_points.push((
                    f.id,
                    point,
                    velocity,
                    axis,
                    radius,
                    hi - lo,
                    f.mass.mass,
                    f.pin.is_some(),
                ));
            }
        }
        let mut queries = Vec::with_capacity(wood_points.len() + 1);
        queries.push(self.drone.pose()?.position);
        queries.extend(wood_points.iter().map(|p| p.1));
        let spray_start = queries.len();
        if let Some(spray) = &self.spray {
            queries.extend(spray.parcels.iter().map(|p| p.drop.position));
        }
        self.stage_seconds[0] += stamp.elapsed().as_secs_f64();
        stamp = std::time::Instant::now();
        let wind = if flying {
            self.air.sample(engine, &queries)?
        } else {
            vec![Vec3::ZERO; queries.len()]
        };
        self.stage_seconds[1] += stamp.elapsed().as_secs_f64();
        stamp = std::time::Instant::now();
        if flying {
            let spray_reference = if let Some(spray) = &mut self.spray {
                let r = spray.mission.update(
                    dt,
                    self.drone.pose()?,
                    self.drone.velocity()?,
                    self.autopilot && self.motors,
                );
                Some(r)
            } else {
                None
            };
            if self.motors {
                if self.autopilot {
                    if let Some(r) = spray_reference {
                        self.target = r.position;
                        self.drone.track(r, Vec3::Z)?;
                    } else {
                        let t = self.time() % FLIGHT_SECONDS;
                        self.drone
                            .track(flight_reference(t, self.flight_height), flight_heading(t))?;
                    }
                } else {
                    self.drone.hold(self.target, 0.65)?;
                }
            } else {
                self.drone.stop_motors()?;
            }
            self.drone.step(dt, wind[0])?;
            let x = self.drone.pose()?.position.x;
            self.visited_canopy_sides[0] |= x < -0.65;
            self.visited_canopy_sides[1] |= x > 0.65;
            for load in self.drone.air_loads()? {
                if load.force.length_squared() + load.torque.length_squared() > 1e-18 {
                    load.deposit(&mut self.forces, dt / AIR_DT)?;
                }
            }
            let (point, force) = self.drone.hull_air_load()?;
            self.react_to_air(point, force)?;
        }
        let before = self.tree.model.clone();
        self.stage_seconds[2] += stamp.elapsed().as_secs_f64();
        stamp = std::time::Instant::now();
        let blade_frames: Vec<_> = self
            .leaves
            .iter()
            .zip(&self.particles)
            .map(|(leaf, p)| {
                if p.attached
                    && let Some(f) = self.tree.model.fiber(leaf.owner)
                {
                    let rotation = f.motion.pose.rotation;
                    let direction = (rotation * leaf.direction).normalize();
                    let normal = rotation * leaf.normal;
                    let normal = (normal - direction * normal.dot(direction)).normalize();
                    BladeFrame {
                        direction,
                        normal,
                        angular_velocity: f.motion.angular_velocity,
                    }
                } else {
                    p.leaf.unwrap().frame
                }
            })
            .collect();
        let leaf_config = Config {
            dt,
            strength_scale: self.leaf_strength_scale,
            ..Default::default()
        };
        if flying && self.coupling {
            self.attachments.step_coupled_blades(
                engine,
                &anchors,
                leaf_config,
                &blade_frames,
                Some(dt / AIR_DT),
            )?;
        } else {
            let still = vec![Vec3::ZERO; self.particles.len()];
            self.attachments
                .step_spatial_blades(engine, &anchors, leaf_config, &still, &blade_frames)?;
        }
        let next = self.attachments.snapshot(engine)?;
        if flying {
            for (i, (old, new)) in self.particles.iter().zip(&next).enumerate() {
                if old.attached && !new.attached {
                    self.peak_shedding_wind = self.peak_shedding_wind.max(new.wind_speed);
                    self.release_positions[i] = Some(old.position);
                    if self.leaf_tracks.len() < 8 {
                        self.leaf_tracks.push(LeafTrack {
                            index: i,
                            samples: Vec::new(),
                        });
                    }
                }
                if let Some(start) = self.release_positions[i] {
                    let blade = new.leaf.unwrap();
                    let clearance = blade
                        .contact_offsets()
                        .iter()
                        .map(|r| new.position.y + r.y)
                        .fold(f64::INFINITY, f64::min);
                    self.grounded_since_launch[i] |= clearance < 0.008;
                    if clearance > 0.03 && new.velocity.y > 0.1 {
                        self.airborne_rising_steps += 1;
                        if self.grounded_since_launch[i] {
                            self.leaf_relaunches += 1;
                            self.relaunched_leaves.insert(i);
                            self.grounded_since_launch[i] = false;
                        }
                    }
                    if clearance < 0.03 && blade.frame.angular_velocity.length() > 1.0 {
                        self.ground_rolling_steps += 1;
                    }
                    self.maximum_leaf_travel = self
                        .maximum_leaf_travel
                        .max(((new.position - start) * Vec3::new(1.0, 0.0, 1.0)).length());
                    self.peak_leaf_spin = self
                        .peak_leaf_spin
                        .max(new.leaf.unwrap().frame.angular_velocity.length());
                }
            }
        }
        self.stage_seconds[3] += stamp.elapsed().as_secs_f64();
        stamp = std::time::Instant::now();
        let mut impulses = Vec::new();
        let mut reactions = Vec::new();
        for (p, leaf) in next.iter().zip(&self.leaves) {
            self.peak_air_force = self.peak_air_force.max(p.air_force.length());
            if p.attached
                && let Some(f) = self.tree.model.fiber(leaf.owner)
            {
                impulses.push((
                    leaf.owner,
                    f.motion.pose.transform_point(leaf.anchor),
                    -p.force * dt,
                ));
            }
        }
        if self.coupling {
            for (element, air) in wood_points.iter().zip(&wind[1..]) {
                let &(id, point, velocity, axis, radius, length, mass, pinned) = element;
                let force = CylinderDrag {
                    radius,
                    length,
                    density: 1.225,
                    transverse_coefficient: 1.1,
                    axial_coefficient: 0.8,
                }
                .sample_force(
                    *air - velocity,
                    axis,
                    if pinned { 0.0 } else { mass.recip() },
                    dt,
                    0.5,
                )?;
                if !pinned {
                    impulses.push((id, point, force * dt));
                }
                reactions.push((point, -force));
            }
        }
        let result = (|| {
            self.stage_seconds[4] += stamp.elapsed().as_secs_f64();
            stamp = std::time::Instant::now();
            if !impulses.is_empty() {
                self.tree.model =
                    self.tree
                        .gpu
                        .impulses_snapshot_cached(engine, &self.tree.model, &impulses)?;
            }
            self.stage_seconds[5] += stamp.elapsed().as_secs_f64();
            stamp = std::time::Instant::now();
            let result = self.tree.advance(engine, 1);
            self.stage_seconds[6] += stamp.elapsed().as_secs_f64();
            stamp = std::time::Instant::now();
            result
        })();
        if let Err(error) = result {
            self.attachments.restore(engine, &self.particles)?;
            self.tree.gpu = GpuStrands::with_limits(engine, before.clone(), self.tree.limits)?;
            self.tree.model = before;
            return Err(error);
        }
        self.particles = next;
        if flying && let Some(spray) = &mut self.spray {
            spray.advance(dt, &wind[spray_start..], &self.particles)?;
            let velocity = self.drone.world.scene.body_velocity(self.drone.body)?;
            spray
                .mission
                .commit_spray(dt, self.drone.pose()?, velocity.linear);
            spray.emit(dt, self.drone.pose()?, velocity.linear, velocity.angular)?;
        }
        if flying && self.frames % 12 == 0 {
            for track in &mut self.leaf_tracks {
                if track.samples.len() < 600 {
                    let p = self.particles[track.index];
                    let n = p.leaf.unwrap().frame.normal;
                    track.samples.push([
                        self.tree.time,
                        p.position.x,
                        p.position.y,
                        p.position.z,
                        n.x,
                        n.y,
                        n.z,
                    ]);
                }
            }
        }
        if flying {
            for (point, force) in reactions {
                self.react_to_air(point, force)?;
            }
        }
        for f in self.tree.model.fibers() {
            let displacement = f.center().distance(
                *self
                    .reference_centers
                    .get(&f.id)
                    .unwrap_or(&f.rest.transform_point(f.mass.local_com)),
            );
            self.peak_branch_displacement = self.peak_branch_displacement.max(displacement);
            if self.trunk_cells.contains(&f.id) {
                self.peak_trunk_displacement = self.peak_trunk_displacement.max(displacement);
            }
            if self.limb_cells.contains(&f.id) {
                self.peak_limb_displacement = self.peak_limb_displacement.max(displacement);
            }
        }
        self.stage_seconds[7] += stamp.elapsed().as_secs_f64();
        Ok(())
    }

    pub fn advance(&mut self, engine: &Engine) -> Result<()> {
        if self.autopilot && self.spray.is_none() {
            let t = self.time() % FLIGHT_SECONDS;
            self.target = flight_target(t, self.flight_height);
        }
        let time = self.time();
        if let Some(wind) = &mut self.crosswind {
            let station = self.spray.as_ref().map_or(0, |s| s.mission.station);
            if station >= orchard_model::TREES_PER_ROW {
                wind.announce(time);
            }
            let (ambient, gusts) = wind.background(time);
            self.air.set_background(ambient, gusts)?;
        }
        self.forces.clear();
        for _ in 0..self.physics_substeps {
            self.microstep(engine, true)?;
        }
        let position = self.drone.pose()?.position;
        if self.frames % 6 == 0 && self.flight_dynamics.len() < 1800 {
            let pose = self.drone.pose()?;
            let v = self.drone.velocity()?;
            let axis = pose.rotation * Vec3::Z;
            let front = pose.rotation * Vec3::X;
            let force = self
                .drone
                .air_loads()?
                .iter()
                .map(|r| r.force)
                .sum::<Vec3>();
            self.flight_dynamics.push([
                self.time(),
                v.x,
                v.y,
                v.z,
                axis.x,
                axis.y,
                axis.z,
                force.x,
                force.y,
                force.z,
                front.z.atan2(front.x),
                axis.y.clamp(-1.0, 1.0).acos(),
            ]);
        }
        for (i, distance) in self.tree_overflight_distances.iter_mut().enumerate() {
            let x = orchard_model::ROW_X[i / orchard_model::TREES_PER_ROW];
            let z = (i % orchard_model::TREES_PER_ROW) as f64 * orchard_model::TREE_SPACING
                - orchard_model::ROW_HALF_LENGTH;
            *distance = distance.min((position.x - x).hypot(position.z - z));
        }
        if self.frames % 60 == 0 && self.flight_trace.len() <= 180 {
            self.flight_trace
                .push([self.time(), position.x, position.y, position.z]);
        }
        let stamp = std::time::Instant::now();
        self.air.step(engine, AIR_DT, &self.forces)?;
        self.stage_seconds[9] += stamp.elapsed().as_secs_f64();
        let sampling_stamp = std::time::Instant::now();
        self.tracer_velocities = self.air.sample(engine, &self.tracer_points)?;
        self.stage_seconds[10] += sampling_stamp.elapsed().as_secs_f64();
        let rotor_frames = self.drone.rotor_frames()?;
        if self.frames % 6 == 0 {
            for (leaf, particle) in self
                .leaves
                .iter()
                .zip(&self.particles)
                .filter(|(_, p)| p.attached)
            {
                for frame in &rotor_frames {
                    let delta = particle.position - frame.position;
                    let axial = delta.dot(frame.rotation * Vec3::X);
                    let radial = (delta.length_squared() - axial * axial).max(0.0).sqrt();
                    let gap = axial.hypot((radial - self.drone.config.rotor_radius).max(0.0))
                        - leaf.length * 0.5;
                    self.nearest_rotor_leaf = self.nearest_rotor_leaf.min(gap);
                }
            }
        }
        for (i, (p, v)) in self
            .tracer_points
            .iter_mut()
            .zip(&self.tracer_velocities)
            .enumerate()
        {
            *p += *v * AIR_DT;
            let path = &mut self.air_paths[i];
            if p.y < 0.03
                || p.x.abs() > 4.5
                || p.z < self.grid.origin().z + 0.3
                || p.z > self.grid.upper().z - 0.3
                || p.y > 4.5
                || (self.frames + i * 3) % 120 == 0
            {
                *p = rotor_frames[i % 4].position
                    + Vec3::new(
                        (random(i + self.frames, 811) - 0.5) * 0.35,
                        0.06 + 0.15 * random(i + self.frames, 812),
                        (random(i + self.frames, 813) - 0.5) * 0.35,
                    );
                path.clear();
            } else if v.length() > 0.15 {
                path.push(*p);
                if path.len() > 40 {
                    path.remove(0);
                }
            }
        }
        self.frames += 1;
        self.stage_seconds[8] += stamp.elapsed().as_secs_f64();
        Ok(())
    }

    /// Add the boundary impulses of reactions that leaves recorded on the device.
    pub fn collect_air_boundary(&mut self, engine: &Engine) -> Result<()> {
        let (linear, angular) = self.air.take_boundary_impulses(engine)?;
        self.boundary_impulse += linear;
        self.boundary_angular_impulse += angular;
        Ok(())
    }

    pub fn diagnose(&mut self, engine: &Engine) -> Result<Diagnostics> {
        let d = self.air.snapshot(engine)?.diagnostics();
        self.air_diagnostics = Some(d);
        Ok(d)
    }

    pub fn settle(&mut self, engine: &Engine) -> Result<()> {
        if self.frames != 0 || self.time() != 0.0 || self.settling_seconds != 0.0 {
            return Err(molla_core::Error::Build(
                "settling requires a fresh orchard".into(),
            ));
        }
        let flight_dt = self.tree.config.dt;
        let strength = self.leaf_strength_scale;
        self.tree.config.dt = AIR_DT;
        self.leaf_strength_scale = SETTLING_STRENGTH_SCALE;
        let result = (|| {
            let mut quiet = 0usize;
            let mut next_report = 2.0;
            let mut final_speed = 0.0;
            while self.tree.time < SETTLING_LIMIT_S {
                self.microstep(engine, false)?;
                let speed = self
                    .tree
                    .model
                    .fibers()
                    .iter()
                    .map(|f| {
                        f.motion.linear_velocity.length() + f.motion.angular_velocity.length() * 0.3
                    })
                    .chain(self.particles.iter().map(|p| p.velocity.length()))
                    .fold(0.0, f64::max);
                final_speed = speed;
                if self.tree.time >= next_report {
                    eprintln!(
                        "ORCHARD settling t={:.1} speed={speed:.6} shed={} wood_failures={}",
                        self.tree.time,
                        self.particles.len() - self.attached(),
                        self.tree.stress_breaks
                    );
                    next_report += 2.0;
                }
                quiet = if speed < 0.005 && self.tree.time > 1.0 {
                    quiet + 1
                } else {
                    0
                };
                if quiet as f64 * self.tree.config.dt >= 0.2 {
                    self.settling_speed = speed;
                    return Ok(());
                }
            }
            Err(molla_core::Error::Solver(format!(
                "vegetation did not settle within {SETTLING_LIMIT_S} seconds; speed={final_speed:.6}"
            )))
        })();
        self.tree.config.dt = flight_dt;
        self.leaf_strength_scale = strength;
        result?;
        if self.attached() != self.particles.len() || self.tree.stress_breaks != 0 {
            return Err(molla_core::Error::Solver(
                "vegetation failed during static initialization".into(),
            ));
        }
        self.settling_seconds = self.tree.time;
        self.tree.time = 0.0;
        self.reference_centers = self
            .tree
            .model
            .fibers()
            .iter()
            .map(|f| (f.id, f.center()))
            .collect();
        self.stage_seconds = [0.0; 11];
        self.peak_air_force = 0.0;
        self.peak_branch_displacement = 0.0;
        self.peak_trunk_displacement = 0.0;
        self.peak_limb_displacement = 0.0;
        eprintln!(
            "ORCHARD settled {:.3} seconds, speed {:.6} m/s, all {} leaves attached",
            self.settling_seconds,
            self.settling_speed,
            self.attached()
        );
        Ok(())
    }
}

fn flight_target(t: f64, height: f64) -> Vec3 {
    flight_reference(t, height).position
}

fn flight_heading(t: f64) -> Vec3 {
    if t < 32.0 {
        Vec3::Z
    } else if t < 39.0 {
        Vec3::X
    } else if t < 69.0 {
        -Vec3::Z
    } else {
        -Vec3::X
    }
}

fn flight_reference(t: f64, height: f64) -> FlightReference {
    let end = orchard_model::ROW_HALF_LENGTH;
    let [left, right] = orchard_model::ROW_X;
    let points = [
        Vec3::new(left, height, -end),
        Vec3::new(left, height, end),
        Vec3::new(right, height, end),
        Vec3::new(right, height, -end),
    ];
    if t < 2.0 {
        return FlightReference::hold(points[0]);
    }
    let (leg, begin, duration, speed) = if t < 32.0 {
        (0, 2.0, LEG_SECONDS - 1.0, 0.7)
    } else if t < 39.0 {
        (1, 32.0, 6.0, 1.1)
    } else if t < 69.0 {
        (2, 39.0, LEG_SECONDS - 1.0, 0.7)
    } else {
        (3, 69.0, 6.0, 1.1)
    };
    FlightSegment::new(
        points[leg],
        points[(leg + 1) % 4],
        MotionLimits {
            speed,
            acceleration: 2.0,
            jerk: 6.0,
        },
    )
    .unwrap()
    .with_duration(duration)
    .unwrap()
    .sample((t - begin).max(0.0))
    .unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Steps every held leaf on its fixed twig in still air; returns sheds and peak load / weight.
    fn still_twig_leaves(seconds: f64) -> (usize, f64) {
        let mut canopy = orchard_model::build().unwrap();
        hold_leaves(&mut canopy);
        let config = Config {
            dt: AIR_DT,
            strength_scale: LEAF_STRENGTH_SCALE,
            ..Default::default()
        };
        let (mut shed, mut peak) = (0, 0.0f64);
        for (leaf, (p, a)) in canopy
            .leaves
            .iter()
            .zip(canopy.particles.iter_mut().zip(&canopy.attachments))
        {
            let f = canopy.tree.model.fiber(leaf.owner).unwrap();
            let anchor = Anchor {
                position: f.motion.pose.transform_point(leaf.anchor),
                velocity: Vec3::ZERO,
                valid: true,
            };
            let direction = f.motion.pose.rotation * leaf.direction;
            let normal = f.motion.pose.rotation * leaf.normal;
            let rest = BladeFrame {
                direction,
                normal: (normal - direction * normal.dot(direction)).normalize(),
                angular_velocity: Vec3::ZERO,
            };
            for _ in 0..(seconds / AIR_DT) as usize {
                molla_solvers::attachments::step_cpu_blade(*a, p, anchor, rest, config).unwrap();
            }
            shed += usize::from(!p.attached);
            peak = peak.max(p.peak_force / (a.mass * 9.81));
        }
        (shed, peak)
    }

    #[test]
    #[ignore = "requires Vulkan"]
    fn gpu_held_leaves_rest_on_still_twigs_without_shedding() {
        let engine = Engine::new_gpu().unwrap();
        let mut canopy = orchard_model::build().unwrap();
        hold_leaves(&mut canopy);
        let mut gpu =
            GpuAttachments::new(&engine, canopy.attachments.clone(), &canopy.particles).unwrap();
        let (anchors, frames): (Vec<_>, Vec<_>) = canopy
            .leaves
            .iter()
            .map(|leaf| {
                let f = canopy.tree.model.fiber(leaf.owner).unwrap();
                let direction = f.motion.pose.rotation * leaf.direction;
                let normal = f.motion.pose.rotation * leaf.normal;
                (
                    Anchor {
                        position: f.motion.pose.transform_point(leaf.anchor),
                        velocity: Vec3::ZERO,
                        valid: true,
                    },
                    BladeFrame {
                        direction,
                        normal: (normal - direction * normal.dot(direction)).normalize(),
                        angular_velocity: Vec3::ZERO,
                    },
                )
            })
            .unzip();
        let config = Config {
            dt: AIR_DT,
            strength_scale: LEAF_STRENGTH_SCALE,
            ..Default::default()
        };
        let wind = vec![Vec3::ZERO; anchors.len()];
        for _ in 0..120 {
            gpu.step_spatial_blades(&engine, &anchors, config, &wind, &frames)
                .unwrap();
        }
        let after = gpu.snapshot(&engine).unwrap();
        let shed = after.iter().filter(|p| !p.attached).count();
        let peak = after
            .iter()
            .zip(&canopy.attachments)
            .map(|(p, a)| p.peak_force / (a.mass * 9.81))
            .fold(0.0, f64::max);
        assert_eq!(shed, 0, "peak load {peak} weights");
        assert!(peak < 1.05, "peak load {peak} weights");
    }

    #[test]
    fn held_leaves_rest_on_still_twigs_without_shedding() {
        let (shed, peak) = still_twig_leaves(2.0);
        assert_eq!(shed, 0, "peak load {peak} weights");
        assert!(peak < 1.05, "peak load {peak} weights");
    }

    #[test]
    fn orchard_downwash_interpolates_between_gentle_and_strong_hover() {
        let target = Vec3::Y * 3.0;
        let mut reactions = Vec::new();
        for mass in [2.0, DRONE_MASS_KG, 8.0] {
            let mut drone = Quadrotor::new(
                target,
                QuadrotorConfig {
                    mass,
                    ..Default::default()
                },
            )
            .unwrap();
            for _ in 0..360 {
                drone.hold(target, 0.65).unwrap();
                drone.step(AIR_DT, Vec3::ZERO).unwrap();
            }
            assert!(drone.pose().unwrap().position.distance(target) < 0.03);
            assert!(drone.velocity().unwrap().length() < 0.03);
            let reaction = drone
                .air_loads()
                .unwrap()
                .iter()
                .map(|load| load.force)
                .sum::<Vec3>();
            assert!((reaction.y + mass * 9.81).abs() < 0.1);
            reactions.push(-reaction.y);
        }
        let level = (reactions[1] - reactions[0]) / (reactions[2] - reactions[0]);
        assert!((level - 0.60).abs() < 0.005);
        assert!((DRONE_MASS_KG - 5.6).abs() < 1e-12);
    }
    #[test]
    fn flight_visits_both_row_centers_and_returns() {
        assert_eq!(
            flight_target(0.0, 3.0),
            flight_target(FLIGHT_SECONDS - 1.0, 3.0)
        );
        assert_eq!(flight_target(0.0, 3.0).x, orchard_model::ROW_X[0]);
        assert_eq!(flight_target(5.0, 3.0).x, orchard_model::ROW_X[0]);
        assert_eq!(
            flight_target(8.5 + LEG_SECONDS, 3.0).x,
            orchard_model::ROW_X[1]
        );
        assert_eq!(flight_target(31.5, 3.0).z, orchard_model::ROW_HALF_LENGTH);
        assert!(flight_reference(2.4, 3.0).acceleration.z > 0.0);
        assert!(flight_reference(30.6, 3.0).acceleration.z < 0.0);
        assert!(LEG_SECONDS * 0.65 > 2.0 * orchard_model::ROW_HALF_LENGTH + 2.0);
        let grid = air_grid().unwrap();
        for i in 0..FLIGHT_SECONDS as usize {
            let p = flight_target(i as f64, 3.0);
            assert!(p.cmpgt(grid.origin() + Vec3::ONE).all());
            assert!(p.cmplt(grid.upper() - Vec3::ONE).all());
        }
    }

    #[test]
    fn dynamic_controller_flies_over_every_tree_in_both_rows() {
        let start = flight_target(0.0, 3.0);
        let mut drone = Quadrotor::new(
            start,
            QuadrotorConfig {
                mass: DRONE_MASS_KG,
                ..Default::default()
            },
        )
        .unwrap();
        let mut gaps = [f64::INFINITY; orchard_model::TREE_COUNT];
        for step in 0..(FLIGHT_SECONDS / AIR_DT) as usize {
            let t = step as f64 * AIR_DT;
            drone
                .track(flight_reference(t, 3.0), flight_heading(t))
                .unwrap();
            drone.step(AIR_DT, Vec3::ZERO).unwrap();
            let p = drone.pose().unwrap().position;
            for (i, gap) in gaps.iter_mut().enumerate() {
                let x = orchard_model::ROW_X[i / orchard_model::TREES_PER_ROW];
                let z = (i % orchard_model::TREES_PER_ROW) as f64 * orchard_model::TREE_SPACING
                    - orchard_model::ROW_HALF_LENGTH;
                *gap = gap.min((p.x - x).hypot(p.z - z));
            }
        }
        assert!(gaps.iter().all(|gap| *gap < 0.15), "{gaps:?}");
        println!(
            "route final {:?}, velocity {:?}, gaps {gaps:?}",
            drone.pose().unwrap(),
            drone.velocity().unwrap()
        );
        assert!(drone.pose().unwrap().position.distance(start) < 0.1);
        assert!(drone.velocity().unwrap().length() < 0.1);
    }

    #[test]
    fn leaf_release_threshold_changes_without_changing_spring_motion() {
        use molla_solvers::attachments::step_cpu;
        let a = Attachment {
            mass: 0.0003,
            stiffness: 0.6,
            damping: 0.015,
            strength: 0.01,
            drag: 0.0,
            radius: 0.002,
            petiole: None,
        };
        let anchor = Anchor {
            position: Vec3::Y + Vec3::X * 0.045,
            velocity: Vec3::ZERO,
            valid: true,
        };
        let mut baseline = Particle::attached(Vec3::Y);
        let mut stronger = baseline;
        let c = Config {
            dt: AIR_DT,
            gravity: Vec3::ZERO,
            ..Default::default()
        };
        step_cpu(a, &mut baseline, anchor, c).unwrap();
        step_cpu(
            a,
            &mut stronger,
            anchor,
            Config {
                strength_scale: LEAF_STRENGTH_SCALE,
                ..c
            },
        )
        .unwrap();
        assert!(LEAF_STRENGTH_SCALE > 1.0 && LEAF_STRENGTH_SCALE < 1.5);
        assert!(!baseline.attached && stronger.attached);
        assert!((baseline.peak_force - stronger.peak_force).abs() < 1e-12);
        let anchor = Anchor {
            position: Vec3::Y + Vec3::X * 0.005,
            ..anchor
        };
        baseline = Particle::attached(Vec3::Y);
        stronger = baseline;
        step_cpu(a, &mut baseline, anchor, c).unwrap();
        step_cpu(
            a,
            &mut stronger,
            anchor,
            Config {
                strength_scale: LEAF_STRENGTH_SCALE,
                ..c
            },
        )
        .unwrap();
        assert!(baseline.attached && stronger.attached);
        assert_eq!(baseline, stronger);
    }
}

