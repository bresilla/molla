use super::*;
use molla_math::Transform;

#[derive(Clone, PartialEq)]
pub(crate) struct SleepWheelGeometry {
    wheels: Vec<(BodyHandle, Transform, Transform)>,
    grounds: Vec<(ColliderHandle, Transform)>,
}

pub(super) struct GroundHit<'a> {
    pub collider: ColliderHandle,
    pub point: Vec3,
    pub normal: Vec3,
    pub friction: Real,
    pub surface: GroundSurface<'a>,
}

pub(super) struct GroundSurface<'a> {
    pub grid: Option<&'a FrictionGrid>,
    pub reference: Real,
    fallback: Real,
    pose: Transform,
}

impl GroundSurface<'_> {
    pub fn sample(&self, point: Vec3) -> Real {
        self.grid
            .and_then(|grid| grid.sample(self.pose.inverse_transform_point(point)))
            .unwrap_or(self.fallback)
    }
}

/// The ground under `hub`, by a ray straight down through the grounds whose
/// world `bounds` cover it.
pub(super) fn query_ground<'a>(
    queries: &mut CpuRaycaster,
    scene: &RigidScene,
    grounds: &'a BTreeMap<ColliderHandle, GroundMaterial>,
    bounds: &BTreeMap<ColliderHandle, molla_geometry::aabb::Aabb>,
    (top, bottom): (Real, Real),
    hub: Vec3,
    mut include: impl FnMut(ColliderHandle) -> bool,
) -> Result<Option<GroundHit<'a>>> {
    let _span = tracing::debug_span!("wheel_ground_query").entered();
    let under = |b: &molla_geometry::aabb::Aabb| {
        b.min.x <= hub.x && hub.x <= b.max.x && b.min.z <= hub.z && hub.z <= b.max.z
    };
    let Some(hit) = queries.cast_ray_among(
        scene,
        bounds
            .iter()
            .filter(|(_, b)| under(b))
            .map(|(&id, _)| id)
            .filter(|&id| include(id)),
        Vec3::new(hub.x, top + 1.0, hub.z),
        -Vec3::Y,
        top - bottom + 2.0,
    )?
    else {
        return Ok(None);
    };
    let normal = if hit.normal.y < 0.0 {
        -hit.normal
    } else {
        hit.normal
    };
    if normal.y < 1e-6 {
        return Ok(None);
    }
    let collider = scene
        .collider(hit.collider)
        .ok_or_else(|| invalid("missing terrain collider"))?;
    let material = &grounds[&hit.collider];
    let surface = GroundSurface {
        grid: material.grid.as_ref(),
        reference: material.max_friction.max(collider.friction),
        fallback: collider.friction,
        pose: scene.collider_world_pose(hit.collider)?,
    };
    let friction = surface.sample(hit.point);
    Ok(Some(GroundHit {
        collider: hit.collider,
        point: hit.point,
        normal,
        friction,
        surface,
    }))
}

impl WheelForces {
    pub(crate) fn sleep_geometry(
        &self,
        scene: &RigidScene,
        bodies: &[BodyHandle],
    ) -> Result<SleepWheelGeometry> {
        let mut wheels = Vec::new();
        for w in self
            .wheels
            .values()
            .filter(|w| bodies.contains(&w.body) && active(scene, w))
        {
            wheels.push((
                w.body,
                scene.body_pose(w.body)?,
                scene.body_pose(w.heading_body)?,
            ));
        }
        let mut grounds = Vec::new();
        if !wheels.is_empty() {
            for &ground in self
                .grounds
                .keys()
                .filter(|&&id| scene.collider(id).is_some())
            {
                grounds.push((ground, scene.collider_world_pose(ground)?));
            }
        }
        Ok(SleepWheelGeometry { wheels, grounds })
    }

    pub(crate) fn observe_sleeping(
        &mut self,
        scene: &RigidScene,
        bodies: &[BodyHandle],
        mut excluded: impl FnMut(Option<BodyHandle>, Option<BodyHandle>) -> bool,
    ) -> Result<Vec<WheelSample>> {
        let mut samples = Vec::new();
        let Some(range) = self.height_range else {
            return Ok(samples);
        };
        for w in self
            .wheels
            .values()
            .filter(|w| bodies.contains(&w.body) && active(scene, w))
        {
            let hub = scene.body_pose(w.body)?.transform_point(w.local_hub);
            let Some(hit) =
                query_ground(&mut self.queries, scene, &self.grounds, &self.ground_bounds, range, hub, |id| {
                    !excluded(Some(w.body), scene.collider(id).and_then(|c| c.parent))
                })?
            else {
                continue;
            };
            let contact = WheelContact {
                hub_pos: hub.to_array(),
                hub_vel: [0.0; 3],
                heading: (scene.body_pose(w.heading_body)?.rotation * w.local_heading.normalize())
                    .to_array(),
                spin: 0.0,
                radius: w.radius,
                normal_stiffness: w.normal_stiffness,
                normal_damping: w.normal_damping,
            };
            let (output, pressure) = if let Some(state) = self.pressure_states.get(&w.body) {
                let friction_at = |point| hit.surface.sample(point);
                let (output, pressure) = crate::pressure_tire::evaluate_with_surface(
                    state,
                    &contact,
                    hit.normal,
                    (hub - hit.point).dot(hit.normal),
                    hit.surface.reference,
                    None,
                    hit.surface
                        .grid
                        .map(|_| &friction_at as &dyn Fn(Vec3) -> Real),
                )?;
                (output, Some(pressure))
            } else {
                (
                    wheel_ground_wrench(
                        &contact,
                        &Surface {
                            point: hit.point,
                            normal: hit.normal,
                            friction: hit.friction,
                        },
                        w.tire.as_ref(),
                    ),
                    None,
                )
            };
            if output.force.iter().any(|v| !v.is_finite())
                || !output.fz.is_finite()
                || !output.slip_ratio.is_finite()
                || !output.slip_angle.is_finite()
                || !output.aligning_moment.is_finite()
            {
                return Err(Error::Solver("non-finite sleeping tyre observation".into()));
            }
            samples.push(WheelSample {
                wheel: w.body,
                ground: hit.collider,
                ground_point: hit.point,
                dt: 0.0,
                normal: hit.normal,
                friction: hit.friction,
                output,
                pressure,
            });
        }
        Ok(samples)
    }
}
