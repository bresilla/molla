// --- Camera scene (Stage 14 — multi-channel raytraced sensor) ------
// 720p source resolution. Pane thumbnails downsample at draw time, but
// the underlying camera renders at full HD-class resolution so the
// "shape index = instance segmentation" channel is readable when zoomed.
const CAMERA_IMG_W: u32 = 1280;
const CAMERA_IMG_H: u32 = 720;
const CAMERA_NUM_VIEWS: usize = 1;
const CAMERA_ORBIT_RADIUS: f32 = 6.0;
const CAMERA_ORBIT_HEIGHT: f32 = 2.5;
const CAMERA_TARGET_HEIGHT: f32 = 1.0;

// Stage-14 camera scene — a "bowl" of vivid-coloured primitives
// dropped onto a plane. Each shape gets its own
// `set_shape_visual_material` colour so the right-rail thumbnails
// (color / albedo / shape index) read at a glance.
fn build_camera_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    b.set_angular_damping(0.05);
    b.set_linear_damping(0.20);

    let mut visuals: Vec<BodyVisual> = Vec::new();

    // Static plane to settle on.
    let plane = b
        .add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
        .expect("plane");
    let _ = b.set_shape_visual_material(plane, MVec3::new(0.55, 0.55, 0.60), 0.0, 0.85);

    // ── Three spheres in a row, vivid hues. ──
    let sphere_specs = [
        (
            MVec3::new(-1.6, 1.5, 0.0),
            0.5,
            MVec3::new(0.95, 0.20, 0.20),
        ), // red
        (MVec3::new(0.0, 1.5, 0.0), 0.5, MVec3::new(0.20, 0.85, 0.30)), // green
        (MVec3::new(1.6, 1.5, 0.0), 0.5, MVec3::new(0.20, 0.40, 0.95)), // blue
    ];
    for (pos, r, color) in sphere_specs {
        let body = b.add_body(
            mp::BodyParams::new()
                .with_spatial_inertia(mp::SpatialInertia::sphere(1.0, r))
                .with_position(pos),
        );
        let shape = b
            .add_shape_sphere(body, r, mp::Transform::IDENTITY)
            .expect("sphere");
        let _ = b.set_shape_visual_material(shape, color, 0.0, 0.4);
        visuals.push(BodyVisual::Sphere(r as f32));
    }

    // ── A box behind, magenta. ──
    {
        let half = MVec3::new(0.55, 0.45, 0.55);
        let body = b.add_body(
            mp::BodyParams::new()
                .with_spatial_inertia(mp::SpatialInertia {
                    mass: 2.0,
                    com: MVec3::ZERO,
                    inertia: box_inertia(2.0, half.max_element()),
                })
                .with_position(MVec3::new(0.0, 1.0, -1.6)),
        );
        let shape = b
            .add_shape_box(body, half, mp::Transform::IDENTITY)
            .expect("box");
        let _ = b.set_shape_visual_material(shape, MVec3::new(0.85, 0.20, 0.85), 0.0, 0.5);
        visuals.push(BodyVisual::Box(Vec3::new(
            half.x as f32,
            half.y as f32,
            half.z as f32,
        )));
    }

    // ── A capsule on the left-front, yellow. ──
    {
        let half_h: f64 = 0.45;
        let radius: f64 = 0.25;
        let body = b.add_body(
            mp::BodyParams::new()
                .with_spatial_inertia(mp::SpatialInertia {
                    mass: 0.8,
                    com: MVec3::ZERO,
                    inertia: capsule_inertia(0.8, half_h, radius),
                })
                .with_position(MVec3::new(-2.4, 1.0, 1.6)),
        );
        let shape = b
            .add_shape_capsule(body, half_h, radius, mp::Transform::IDENTITY)
            .expect("capsule");
        let _ = b.set_shape_visual_material(shape, MVec3::new(0.95, 0.85, 0.20), 0.0, 0.45);
        visuals.push(BodyVisual::Capsule {
            half_h: half_h as f32,
            radius: radius as f32,
        });
    }

    // ── A cone on the right-front, cyan. ──
    {
        let half_h: f64 = 0.6;
        let radius: f64 = 0.45;
        let body = b.add_body(
            mp::BodyParams::new()
                .with_spatial_inertia(mp::SpatialInertia {
                    mass: 1.2,
                    com: MVec3::ZERO,
                    inertia: capsule_inertia(1.2, half_h, radius),
                })
                .with_position(MVec3::new(2.4, 1.2, 1.6)),
        );
        let shape = b
            .add_shape_cone(body, half_h, radius, mp::Transform::IDENTITY)
            .expect("cone");
        let _ = b.set_shape_visual_material(shape, MVec3::new(0.20, 0.85, 0.85), 0.0, 0.45);
        visuals.push(BodyVisual::Cone {
            half_h: half_h as f32,
            radius: radius as f32,
        });
    }

    (b.build().expect("camera scene model"), visuals)
}

// Per-frame state for the camera-sensor demo. The `TiledCamera` is
// built lazily on first entry to the scene; pixel bytes for every
// channel get written into Bevy `Image` assets each frame and the
// right-rail pane displays them via `bevy_egui`'s user-texture path.
#[derive(Resource)]
struct CameraExampleState {
    sensor: Option<TiledCamera>,
    /// Wall-clock seconds since the scene started; drives the orbit.
    elapsed: f32,
    /// FOV slider value (radians). Synchronised back into the sensor's
    /// config every frame. Ignored when `optics_enabled` is true and
    /// `focal_length_mm` / `sensor_width_mm` are both > 0.
    fov_y: f32,
    /// H10 — toggle the physical-optics path. When off, the legacy
    /// pinhole renderer runs (bit-identical to pre-H10 output).
    optics_enabled: bool,
    /// H10 — focal length (mm). Combined with `sensor_width_mm`
    /// drives the effective FOV.
    focal_length_mm: f32,
    /// H10 — physical sensor width (mm).
    sensor_width_mm: f32,
    /// H10 — aperture entrance diameter (m). `0` = pinhole.
    lens_diameter_m: f32,
    /// H10 — focal-plane distance (m).
    focal_distance_m: f32,
    /// H10 — aperture samples per pixel for the K-sample DOF render.
    /// `1` = single sample (no DOF averaging).
    aperture_samples: u32,
    /// H11 — exposure multiplier applied to the colour buffer in the
    /// camera post-process pass. `1.0` = identity.
    exposure: f32,
    /// H11 — Reinhard tone-mapping toggle (caps highlights).
    tone_map_enabled: bool,
    /// H11 — sRGB gamma compression toggle (linear → display).
    gamma_enabled: bool,
    /// Most recent camera world pose, for the readout.
    last_pose: Option<mp::Transform>,
    /// Bevy `Image` assets backing each channel thumbnail. Allocated
    /// once on first dispatch and rewritten in-place each frame; the
    /// matching `egui::TextureId` is cached after registering the
    /// handle with `EguiUserTextures`.
    image_color: Option<Handle<Image>>,
    image_depth: Option<Handle<Image>>,
    image_normal: Option<Handle<Image>>,
    image_albedo: Option<Handle<Image>>,
    image_index: Option<Handle<Image>>,
    tex_id_color: Option<egui::TextureId>,
    tex_id_depth: Option<egui::TextureId>,
    tex_id_normal: Option<egui::TextureId>,
    tex_id_albedo: Option<egui::TextureId>,
    tex_id_index: Option<egui::TextureId>,
    /// Min / max depth observed last frame — used to scale the depth
    /// thumbnail to a useful greyscale range.
    last_depth_range: Option<(f32, f32)>,
}

impl Default for CameraExampleState {
    fn default() -> Self {
        Self {
            sensor: None,
            elapsed: 0.0,
            fov_y: 50f32.to_radians(),
            optics_enabled: false,
            focal_length_mm: 50.0,
            sensor_width_mm: 36.0, // 35-mm "full-frame" reference
            lens_diameter_m: 0.0,
            focal_distance_m: 8.0,
            aperture_samples: 1,
            exposure: 1.0,
            tone_map_enabled: false,
            gamma_enabled: false,
            last_pose: None,
            image_color: None,
            image_depth: None,
            image_normal: None,
            image_albedo: None,
            image_index: None,
            tex_id_color: None,
            tex_id_depth: None,
            tex_id_normal: None,
            tex_id_albedo: None,
            tex_id_index: None,
            last_depth_range: None,
        }
    }
}

// ---- Bevy systems -------------------------------------------------

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    sim: Res<Sim>,
    visuals: Res<BodyVisuals>,
    mut activity: ResMut<ActivityView>,
) {
    // Sleeping material — uniform muted grey, lower roughness so it
    // doesn't blend completely into the dark background.
    let sleeping = materials.add(StandardMaterial {
        base_color: Color::srgb(0.32, 0.34, 0.38),
        metallic: 0.0,
        perceptual_roughness: 0.85,
        ..default()
    });
    activity.sleeping_mat = Some(sleeping);

    let sun_shadow = CascadeShadowConfigBuilder {
        num_cascades: 2,
        minimum_distance: 0.1,
        maximum_distance: 200.0,
        first_cascade_far_bound: 30.0,
        overlap_proportion: 0.2,
    }
    .build();
    // Three-point rig: key + fill + back. The key has shadows, the
    // fill softens the shadowed side without doubling shadow cost,
    // the back rim light separates objects from the dark background.
    commands.spawn((
        Name::new("Sun"),
        Transform::from_xyz(20.0, 60.0, 15.0).looking_at(Vec3::ZERO, Vec3::Y),
        DirectionalLight {
            illuminance: 14_000.0,
            shadows_enabled: true,
            ..default()
        },
        sun_shadow,
    ));
    commands.spawn((
        Name::new("FillLight"),
        Transform::from_xyz(-25.0, 35.0, -10.0).looking_at(Vec3::ZERO, Vec3::Y),
        DirectionalLight {
            illuminance: 6_500.0,
            shadows_enabled: false,
            color: Color::srgb(0.85, 0.90, 1.00),
            ..default()
        },
    ));
    commands.spawn((
        Name::new("RimLight"),
        Transform::from_xyz(0.0, 25.0, -45.0).looking_at(Vec3::ZERO, Vec3::Y),
        DirectionalLight {
            illuminance: 4_000.0,
            shadows_enabled: false,
            color: Color::srgb(1.00, 0.90, 0.78),
            ..default()
        },
    ));

    let chase = ChaseCamera {
        focus: Vec3::new(0.0, 4.0, 0.0),
        distance: 30.0,
        elevation: 20f32.to_radians(),
        min_elevation: -2f32.to_radians(),
        // Zoom range — `min_distance` (≈ 5 cm) lets the orbit camera
        // get right up against small primitives without the default
        // 3 m wall; `max_distance` opens room for the larger outdoor
        // scenes (aerial-lidar pass, solar sky).
        min_distance: 0.05,
        max_distance: 2000.0,
        ..ChaseCamera::default()
    };
    let mut tr = Transform::default();
    apply_rig(&chase, &mut tr);
    commands.spawn((
        Name::new("MainCamera"),
        Camera3d::default(),
        tr,
        AmbientLight {
            color: Color::srgb(0.85, 0.90, 1.0),
            brightness: 280.0,
            ..default()
        },
        chase,
        GizmoCamera,
    ));

    commands.spawn((
        Name::new("OriginAxes"),
        Transform::from_xyz(0.0, 0.001, 0.0),
        AxisGizmo {
            length: 1.5,
            ..AxisGizmo::default()
        },
    ));

    spawn_body_entities(&mut commands, &mut meshes, &mut materials, &sim, &visuals);

    info!(
        "molla viewer: scene = {}, {} bodies, {}× @ {:.0} Hz physics",
        sim.scene.label(),
        sim.model.body_count,
        SUBSTEPS_PER_FRAME,
        1.0 / FIXED_DT
    );
    info!("keys: 1 / 2 / 3 switch scene · Space pause · R reset · Esc quit");
}

// Build a per-visual mesh + material and spawn one entity per molla
// body. Called from `setup` and after every scene switch.
fn spawn_body_entities(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    sim: &Sim,
    visuals: &BodyVisuals,
) {
    let palette = [
        Color::srgb(0.88, 0.35, 0.28),
        Color::srgb(0.90, 0.60, 0.20),
        Color::srgb(0.85, 0.78, 0.25),
        Color::srgb(0.45, 0.78, 0.35),
        Color::srgb(0.25, 0.70, 0.85),
        Color::srgb(0.40, 0.50, 0.95),
        Color::srgb(0.65, 0.42, 0.92),
        Color::srgb(0.92, 0.50, 0.70),
    ];
    let mats: Vec<_> = palette
        .iter()
        .map(|c| {
            materials.add(StandardMaterial {
                base_color: *c,
                metallic: 0.05,
                perceptual_roughness: 0.45,
                ..default()
            })
        })
        .collect();

    let body_q = sim.state_a.body_q.host().expect("body_q").to_vec();
    for (i, pose) in body_q.iter().enumerate() {
        // Per-visual extra scale (Ellipsoid bakes its semi-axes into a
        // non-uniform transform scale; GaussianSplat collapses to a
        // billboard-radius scale for the stub renderer).
        let mut extra_scale = Vec3::ONE;
        let mesh = match visuals.0[i] {
            // ico(3) → 1280 tris/sphere (vs 320 at ico(2)). Smooth at
            // grab range, still cheap at 144 spheres.
            BodyVisual::Sphere(r) => meshes.add(Sphere::new(r).mesh().ico(4).expect("ico")),
            // Cuboid only has 12 triangles regardless of subdivision —
            // looks fine even in wireframe (12 cube edges visible).
            BodyVisual::Box(h) => meshes.add(Cuboid::from_size(h * 2.0).mesh()),
            BodyVisual::Capsule { half_h, radius } => meshes.add(
                Capsule3d::new(radius, half_h * 2.0)
                    .mesh()
                    .longitudes(32)
                    .latitudes(16),
            ),
            BodyVisual::Cone { half_h, radius } => meshes.add(
                Cone {
                    radius,
                    height: half_h * 2.0,
                }
                .mesh()
                .resolution(32)
                .build(),
            ),
            // Ellipsoid: render a unit sphere mesh with non-uniform
            // scale baked into the entity transform. Bevy's PBR
            // pipeline handles anisotropic scales — wireframe still
            // shows the deformed icosphere edges, so the stretched
            // shape is obvious.
            BodyVisual::Ellipsoid { half_axes } => {
                extra_scale = half_axes;
                meshes.add(Sphere::new(1.0).mesh().ico(4).expect("ico"))
            }
            // GaussianSplat stub: render as a small billboarded
            // sphere. Real rendering (oriented anisotropic Gaussian
            // disc) is deferred — this gives the viewer something
            // visible without dragging in a custom shader.
            BodyVisual::GaussianSplat { covariance_inv, .. } => {
                // Use the (capped) inverse-determinant as a rough
                // stddev proxy. Real splat rendering would want
                // the eigenvectors of `covariance_inv⁻¹`.
                let det = covariance_inv[0].x * covariance_inv[1].y * covariance_inv[2].z;
                let r = (1.0 / det.abs().max(1e-3)).cbrt().clamp(0.05, 0.5);
                meshes.add(Sphere::new(r).mesh().ico(2).expect("ico"))
            }
        };
        let awake = mats[i % mats.len()].clone();
        let mut t = Transform::from_xyz(
            pose.position.x as f32,
            pose.position.y as f32,
            pose.position.z as f32,
        );
        t.scale = extra_scale;
        commands.spawn((
            MollaBody(mp::BodyId::from_index(i)),
            Mesh3d(mesh),
            MeshMaterial3d(awake.clone()),
            AwakeMaterial(awake),
            NotShadowCaster,
            t,
        ));
    }

    // Cloth: build a single deforming triangle mesh from
    // `model.tri_indices`, attach it to one entity. The
    // `sync_cloth_mesh` system updates vertex positions each frame.
    if sim.model.particle_count > 0 && sim.model.tri_count > 0 {
        let mesh = build_cloth_mesh(sim);
        let mesh_handle = meshes.add(mesh);
        let mat = materials.add(StandardMaterial {
            base_color: Color::srgb(0.85, 0.55, 0.30),
            metallic: 0.05,
            perceptual_roughness: 0.65,
            cull_mode: None, // double-sided so back faces are visible
            ..default()
        });
        commands.spawn((
            ClothMesh,
            Mesh3d(mesh_handle),
            MeshMaterial3d(mat),
            NotShadowCaster,
            Transform::default(),
        ));
    }

    // Cloth-scene static sphere obstacle.
    if sim.scene == Scene::Cloth {
        let mesh = meshes.add(Sphere::new(CLOTH_SPHERE_RADIUS).mesh().ico(4).expect("ico"));
        let mat = materials.add(StandardMaterial {
            base_color: Color::srgb(0.55, 0.62, 0.75),
            metallic: 0.10,
            perceptual_roughness: 0.55,
            ..default()
        });
        commands.spawn((
            ClothObstacle,
            Mesh3d(mesh),
            MeshMaterial3d(mat),
            Transform::from_xyz(0.0, CLOTH_SPHERE_Y, 0.0),
        ));
    }

    // Stage-20 VBD cloth scene — same kinematic-sphere obstacle
    // pattern as the XPBD scene but with the larger sphere from the
    // VBD constants.
    if sim.scene == Scene::VbdCloth {
        let mesh = meshes.add(
            Sphere::new(VBD_CLOTH_SPHERE_RADIUS)
                .mesh()
                .ico(5)
                .expect("ico"),
        );
        let mat = materials.add(StandardMaterial {
            base_color: Color::srgb(0.45, 0.55, 0.70),
            metallic: 0.15,
            perceptual_roughness: 0.50,
            ..default()
        });
        commands.spawn((
            ClothObstacle,
            Mesh3d(mesh),
            MeshMaterial3d(mat),
            Transform::from_xyz(0.0, VBD_CLOTH_SPHERE_Y, 0.0),
        ));
    }

    // Stage-20.1 CABLE scene — render the static drape-cylinder
    // obstacle. The orbiting puller body is rendered through the
    // generic body-visual loop above; the cable polyline itself is
    // rendered via gizmos by `draw_cable_gizmos`.
    if sim.scene == Scene::Cable {
        let mesh = meshes.add(
            Cylinder::new(CABLE_CYLINDER_RADIUS, CABLE_CYLINDER_HALF_HEIGHT * 2.0)
                .mesh()
                .resolution(48),
        );
        let mat = materials.add(StandardMaterial {
            base_color: Color::srgb(0.55, 0.62, 0.75),
            metallic: 0.10,
            perceptual_roughness: 0.55,
            ..default()
        });
        commands.spawn((
            ClothObstacle,
            Mesh3d(mesh),
            MeshMaterial3d(mat),
            Transform::from_xyz(
                CABLE_CYLINDER_POS.x,
                CABLE_CYLINDER_POS.y,
                CABLE_CYLINDER_POS.z,
            ),
        ));
    }

    // Shapes-scene static visuals: heightfield (deforming triangle
    // strip from the model's heights) + torus mesh.
    if sim.scene == Scene::Shapes {
        if let Some(hf) = sim.model.heightfields.first() {
            let mesh = build_heightfield_mesh(hf);
            let mat = materials.add(StandardMaterial {
                base_color: Color::srgb(0.45, 0.55, 0.40),
                metallic: 0.05,
                perceptual_roughness: 0.85,
                cull_mode: None,
                ..default()
            });
            commands.spawn((
                ClothObstacle, // reuse the despawn tag
                Mesh3d(meshes.add(mesh)),
                MeshMaterial3d(mat),
                Transform::default(),
            ));
        }
        // Torus visual at the SDF position.
        let torus = Torus {
            major_radius: SHAPES_TORUS_MAJOR,
            minor_radius: SHAPES_TORUS_MINOR,
        };
        let torus_mat = materials.add(StandardMaterial {
            base_color: Color::srgb(0.85, 0.65, 0.30),
            metallic: 0.10,
            perceptual_roughness: 0.45,
            ..default()
        });
        commands.spawn((
            ClothObstacle,
            Mesh3d(meshes.add(torus.mesh().major_resolution(64).minor_resolution(24))),
            MeshMaterial3d(torus_mat),
            Transform::from_xyz(0.0, SHAPES_TORUS_Y, 0.0),
        ));
    }
}


fn update_camera_example(
    time: Res<Time>,
    sim: ResMut<Sim>,
    mut state: ResMut<CameraExampleState>,
    mut images: ResMut<Assets<Image>>,
    mut user_textures: ResMut<EguiUserTextures>,
) {
    if sim.scene != Scene::Camera {
        return;
    }
    state.elapsed += time.delta_secs();

    if state.sensor.is_none() {
        let cfg = TiledCameraConfig {
            width: CAMERA_IMG_W,
            height: CAMERA_IMG_H,
            fov_y_radians: 50f32.to_radians(),
            channels: CameraChannels::ALL,
            ambient: MVec3::new(0.18, 0.18, 0.22),
            light_dir: MVec3::new(-0.4, -0.85, -0.35).normalize(),
            light_color: MVec3::new(1.0, 0.96, 0.88),
            max_distance: 100.0,
            near: 0.05,
            optics: Default::default(),
        };
        match TiledCamera::new(&sim.engine, cfg, CAMERA_NUM_VIEWS) {
            Ok(s) => {
                state.fov_y = cfg.fov_y_radians;
                state.sensor = Some(s);
            }
            Err(e) => {
                warn!("TiledCamera init failed: {e}");
                return;
            }
        }
    }

    // Lazily allocate one Bevy `Image` asset per channel + register it
    // with `EguiUserTextures` so the pane can address it via TextureId.
    if state.image_color.is_none() {
        let (h_color, id_color) = alloc_thumb_image(&mut images, &mut user_textures);
        let (h_depth, id_depth) = alloc_thumb_image(&mut images, &mut user_textures);
        let (h_normal, id_normal) = alloc_thumb_image(&mut images, &mut user_textures);
        let (h_albedo, id_albedo) = alloc_thumb_image(&mut images, &mut user_textures);
        let (h_index, id_index) = alloc_thumb_image(&mut images, &mut user_textures);
        state.image_color = Some(h_color);
        state.image_depth = Some(h_depth);
        state.image_normal = Some(h_normal);
        state.image_albedo = Some(h_albedo);
        state.image_index = Some(h_index);
        state.tex_id_color = Some(id_color);
        state.tex_id_depth = Some(id_depth);
        state.tex_id_normal = Some(id_normal);
        state.tex_id_albedo = Some(id_albedo);
        state.tex_id_index = Some(id_index);
    }

    // Push the latest FOV / optics sliders into the sensor config
    // (width/height can't change post-construction; everything else
    // is per-dispatch). Snapshot the state values before borrowing
    // `state.sensor` mutably.
    {
        let fov_y = state.fov_y;
        let optics_enabled = state.optics_enabled;
        let focal_length_mm = state.focal_length_mm;
        let sensor_width_mm = state.sensor_width_mm;
        let lens_diameter_m = state.lens_diameter_m;
        let focal_distance_m = state.focal_distance_m;
        let aperture_samples = state.aperture_samples;
        let sensor = state.sensor.as_mut().unwrap();
        let mut cfg = *sensor.config();
        cfg.fov_y_radians = fov_y;
        if optics_enabled {
            cfg.optics.focal_length_mm = focal_length_mm;
            cfg.optics.sensor_width_mm = sensor_width_mm;
            cfg.optics.lens_diameter_m = lens_diameter_m;
            cfg.optics.focal_distance_m = focal_distance_m;
            cfg.optics.aperture_samples = aperture_samples;
        } else {
            cfg.optics = molla::sensors::PhysicalOptics::default();
        }
        let _ = sensor.set_config(cfg);
    }

    // Orbit the camera around the scene at fixed radius / height.
    let theta = state.elapsed * 0.5; // rad/s — slow circle.
    let eye = MVec3::new(
        (CAMERA_ORBIT_RADIUS as f64) * theta.cos() as f64,
        CAMERA_ORBIT_HEIGHT as f64,
        (CAMERA_ORBIT_RADIUS as f64) * theta.sin() as f64,
    );
    let target = MVec3::new(0.0, CAMERA_TARGET_HEIGHT as f64, 0.0);
    {
        let sensor = state.sensor.as_mut().unwrap();
        if let Err(e) = sensor.set_camera_look_at(0, eye, target, MVec3::Y) {
            warn!("set_camera_look_at: {e}");
            return;
        }
    }

    // Run both kernels and download every channel.
    let Sim {
        ref engine,
        ref model,
        ref state_a,
        ..
    } = *sim;
    let (
        last_pose,
        snapshot_color,
        snapshot_depth,
        snapshot_normal,
        snapshot_albedo,
        snapshot_index,
    ) = {
        let sensor = state.sensor.as_mut().unwrap();
        if let Err(e) = sensor.dispatch(engine, model, state_a) {
            warn!("TiledCamera::dispatch: {e}");
            return;
        }
        let pose = sensor.poses_host().ok().and_then(|p| p.first().copied());
        let c = sensor.download_color(engine).map(|s| s.to_vec());
        let d = sensor.download_depth(engine).map(|s| s.to_vec());
        let n = sensor.download_normal(engine).map(|s| s.to_vec());
        let a = sensor.download_albedo(engine).map(|s| s.to_vec());
        let i = sensor.download_shape_index(engine).map(|s| s.to_vec());
        (pose, c, d, n, a, i)
    };
    state.last_pose = last_pose;

    if let (Ok(c), Some(handle)) = (snapshot_color.as_ref(), state.image_color.as_ref()) {
        // H11 — photographic post-process. Each pass mutates the
        // `u32`-packed buffer in place; identity when the toggles
        // are off and `exposure == 1.0`.
        let mut color_proc = c.clone();
        if (state.exposure - 1.0).abs() > 1.0e-3 {
            molla::sensors::camera_post::apply_exposure(&mut color_proc, state.exposure);
        }
        if state.tone_map_enabled {
            molla::sensors::camera_post::apply_reinhard_tone_map(&mut color_proc);
        }
        if state.gamma_enabled {
            molla::sensors::camera_post::apply_gamma_srgb(&mut color_proc);
        }
        if let Some(img) = images.get_mut(handle) {
            write_rgba_u32_into_image(img, &color_proc);
        }
    }
    if let Ok(d) = snapshot_depth.as_ref() {
        let mut lo = f32::INFINITY;
        let mut hi = 0.0_f32;
        for &v in d {
            let f = v as f32;
            if f > 0.0 {
                if f < lo {
                    lo = f;
                }
                if f > hi {
                    hi = f;
                }
            }
        }
        if !lo.is_finite() || hi <= lo {
            lo = 0.0;
            hi = 1.0;
        }
        state.last_depth_range = Some((lo, hi));
        if let Some(handle) = state.image_depth.as_ref()
            && let Some(img) = images.get_mut(handle) {
                write_depth_into_image(img, d, lo, hi);
            }
    }
    if let (Ok(n), Some(handle)) = (snapshot_normal.as_ref(), state.image_normal.as_ref())
        && let Some(img) = images.get_mut(handle) {
            write_normal_into_image(img, n);
        }
    if let (Ok(a), Some(handle)) = (snapshot_albedo.as_ref(), state.image_albedo.as_ref())
        && let Some(img) = images.get_mut(handle) {
            write_rgba_u32_into_image(img, a);
        }
    if let (Ok(i), Some(handle)) = (snapshot_index.as_ref(), state.image_index.as_ref())
        && let Some(img) = images.get_mut(handle) {
            write_shape_index_into_image(img, i);
        }
}

// Allocate a `CAMERA_IMG_W × CAMERA_IMG_H` RGBA8 Bevy image and
// register it with bevy_egui's user-texture map. The image data
// starts black; subsequent frames mutate the bytes in place.
fn alloc_thumb_image(
    images: &mut Assets<Image>,
    user_textures: &mut EguiUserTextures,
) -> (Handle<Image>, egui::TextureId) {
    use bevy::asset::RenderAssetUsages;
    use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
    let size = Extent3d {
        width: CAMERA_IMG_W,
        height: CAMERA_IMG_H,
        depth_or_array_layers: 1,
    };
    let img = Image::new_fill(
        size,
        TextureDimension::D2,
        &[0, 0, 0, 255],
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::RENDER_WORLD | RenderAssetUsages::MAIN_WORLD,
    );
    let handle = images.add(img);
    let id = user_textures.add_image(EguiTextureHandle::Strong(handle.clone()));
    (handle, id)
}

fn write_rgba_u32_into_image(img: &mut Image, src: &[u32]) {
    let Some(data) = img.data.as_mut() else {
        return;
    };
    if data.len() != src.len() * 4 {
        return;
    }
    for (i, &p) in src.iter().enumerate() {
        let r = (p & 0xff) as u8;
        let g = ((p >> 8) & 0xff) as u8;
        let b = ((p >> 16) & 0xff) as u8;
        let a = ((p >> 24) & 0xff) as u8;
        let off = i * 4;
        data[off] = r;
        data[off + 1] = g;
        data[off + 2] = b;
        data[off + 3] = a;
    }
}

fn write_depth_into_image(img: &mut Image, src: &[molla::math::Real], lo: f32, hi: f32) {
    let Some(data) = img.data.as_mut() else {
        return;
    };
    if data.len() != src.len() * 4 {
        return;
    }
    let span = (hi - lo).max(1e-6);
    for (i, &v) in src.iter().enumerate() {
        let off = i * 4;
        let f = v as f32;
        let g = if f <= 0.0 {
            0
        } else {
            let t = ((f - lo) / span).clamp(0.0, 1.0);
            ((1.0 - t) * 255.0) as u8
        };
        data[off] = g;
        data[off + 1] = g;
        data[off + 2] = g;
        data[off + 3] = 255;
    }
}

fn write_normal_into_image(img: &mut Image, src: &[MVec3]) {
    let Some(data) = img.data.as_mut() else {
        return;
    };
    if data.len() != src.len() * 4 {
        return;
    }
    for (i, v) in src.iter().enumerate() {
        let off = i * 4;
        let len2 = v.length_squared();
        if len2 < 1e-9 {
            data[off] = 0;
            data[off + 1] = 0;
            data[off + 2] = 0;
        } else {
            data[off] = ((v.x as f32 * 0.5 + 0.5) * 255.0).clamp(0.0, 255.0) as u8;
            data[off + 1] = ((v.y as f32 * 0.5 + 0.5) * 255.0).clamp(0.0, 255.0) as u8;
            data[off + 2] = ((v.z as f32 * 0.5 + 0.5) * 255.0).clamp(0.0, 255.0) as u8;
        }
        data[off + 3] = 255;
    }
}

fn write_shape_index_into_image(img: &mut Image, src: &[u32]) {
    let Some(data) = img.data.as_mut() else {
        return;
    };
    if data.len() != src.len() * 4 {
        return;
    }
    const PALETTE: &[(u8, u8, u8)] = &[
        (231, 76, 60),
        (52, 152, 219),
        (46, 204, 113),
        (241, 196, 15),
        (155, 89, 182),
        (26, 188, 156),
        (230, 126, 34),
        (192, 57, 43),
        (41, 128, 185),
        (39, 174, 96),
        (211, 84, 0),
        (142, 68, 173),
    ];
    for (i, &idx) in src.iter().enumerate() {
        let off = i * 4;
        let (r, g, b) = if idx == 0xffff_ffff {
            (0, 0, 0)
        } else {
            PALETTE[(idx as usize) % PALETTE.len()]
        };
        data[off] = r;
        data[off + 1] = g;
        data[off + 2] = b;
        data[off + 3] = 255;
    }
}

// Draw a yellow arrow at every body's contact centroid pointing
// along the net push direction (`normal_sum`). Length is scaled by
// `max_depth` so deeper penetrations get longer arrows. No-op when

fn camera_pane(
    ui: &mut egui::Ui,
    state: &mut CameraExampleState,
    anchor: PaneAnchor,
    accent: egui::Color32,
) {
    let pane_id = egui::Id::new(PANE_SCENE);
    // Header readouts + FOV slider. These don't fold — they're always
    // visible at the top of the pane body, above the channel containers.
    ui.add_space(2.0);
    readout(ui, "Sensor", "GPU TiledCamera");
    readout(
        ui,
        "Resolution",
        &format!("{}×{}", CAMERA_IMG_W, CAMERA_IMG_H),
    );
    if let Some(p) = state.last_pose {
        readout(
            ui,
            "Eye",
            &format!(
                "({:>5.2},{:>5.2},{:>5.2})",
                p.position.x, p.position.y, p.position.z
            ),
        );
    }
    ui.separator();
    let mut fov_deg = state.fov_y.to_degrees();
    if ui
        .add(egui::Slider::new(&mut fov_deg, 20.0..=110.0).text("FOV° (vert)"))
        .changed()
    {
        state.fov_y = fov_deg.to_radians();
    }
    ui.separator();

    // ── H10: Physical optics knobs ────────────────────────────────
    ui.checkbox(&mut state.optics_enabled, "Physical optics (H10)");
    if state.optics_enabled {
        ui.add(
            egui::Slider::new(&mut state.focal_length_mm, 8.0..=200.0).text("Focal length (mm)"),
        );
        ui.add(egui::Slider::new(&mut state.sensor_width_mm, 8.0..=60.0).text("Sensor width (mm)"));
        ui.add(egui::Slider::new(&mut state.lens_diameter_m, 0.0..=0.40).text("Lens diameter (m)"));
        ui.add(
            egui::Slider::new(&mut state.focal_distance_m, 0.5..=50.0).text("Focal distance (m)"),
        );
        ui.add(egui::Slider::new(&mut state.aperture_samples, 1..=16).text("Aperture samples (K)"));
        let hfov = if state.focal_length_mm > 0.0 && state.sensor_width_mm > 0.0 {
            2.0 * (state.sensor_width_mm / (2.0 * state.focal_length_mm)).atan()
        } else {
            0.0
        };
        ui.small(format!("→ effective HFOV ≈ {:.1}°", hfov.to_degrees()));
    }
    ui.separator();

    // ── H11: Photographic post-process knobs ──────────────────────
    ui.label("Post-process (H11)");
    ui.add(egui::Slider::new(&mut state.exposure, 0.1..=4.0).text("Exposure"));
    ui.checkbox(&mut state.tone_map_enabled, "Reinhard tone-map");
    ui.checkbox(&mut state.gamma_enabled, "sRGB gamma");
    ui.separator();

    if state.tex_id_color.is_none() {
        ui.add_space(6.0);
        ui.small("waiting for first GPU dispatch…");
        return;
    }

    let img_w = CAMERA_IMG_W as f32;
    let img_h = CAMERA_IMG_H as f32;
    let aspect = img_h / img_w;
    let mk_image_pod = move |slot: &str, tex: egui::TextureId| -> Pod {
        Pod::new(pane_id.with(("cam_pod", slot))).with_custom(move |ui| {
            // Scale the image to whatever width the container body
            // gives us; height tracks the camera's aspect ratio so
            // dragging the pane wider/taller grows the thumbnail.
            let w = ui.available_width().max(64.0);
            let h = w * aspect;
            ui.image((tex, egui::vec2(w, h)));
        })
    };

    if let Some(tex) = state.tex_id_color {
        let cid = pane_id.with("cam_container_color");
        Normal::new("Color", anchor, accent, cid).show(ui, vec![mk_image_pod("color", tex)]);
    }
    if let Some(tex) = state.tex_id_depth {
        let cid = pane_id.with("cam_container_depth");
        let (lo, hi) = state.last_depth_range.unwrap_or((0.0, 1.0));
        Normal::new(
            format!("Depth [{lo:.2}..{hi:.2}m]").as_str(),
            anchor,
            accent,
            cid,
        )
        .show(ui, vec![mk_image_pod("depth", tex)]);
    }
    if let Some(tex) = state.tex_id_normal {
        let cid = pane_id.with("cam_container_normal");
        Normal::new("Normal", anchor, accent, cid).show(ui, vec![mk_image_pod("normal", tex)]);
    }
    if let Some(tex) = state.tex_id_albedo {
        let cid = pane_id.with("cam_container_albedo");
        Normal::new("Albedo", anchor, accent, cid).show(ui, vec![mk_image_pod("albedo", tex)]);
    }
    if let Some(tex) = state.tex_id_index {
        let cid = pane_id.with("cam_container_index");
        Normal::new("Shape index", anchor, accent, cid).show(ui, vec![mk_image_pod("index", tex)]);
    }
}

// ── Stage-24d MPM (sand) viewer wiring ─────────────────────────────

// Trivial Model for the MPM scene — just a static "ground" body
// used by the standard renderer to keep the camera oriented.  The

