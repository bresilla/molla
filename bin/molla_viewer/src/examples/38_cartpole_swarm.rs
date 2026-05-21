// CartPoleSwarm scene — N independent cart-poles in a row, each in
// its own articulation. Demonstrates batched PD control. Port of
// `xtra/newton/newton/examples/selection/example_selection_cartpole.py`.
const CARTPOLE_SWARM_N: usize = 16;
const CARTPOLE_SWARM_SPACING: f64 = 1.2;
fn build_cartpole_swarm_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    b.set_angular_damping(0.02);

    let mut visuals = Vec::with_capacity(CARTPOLE_SWARM_N * 2);
    let cart_inertia = box_inertia_xyz(CART_MASS, CART_HALF_X, CART_HALF_Y, CART_HALF_Z);
    let pole_inertia = capsule_inertia(POLE_MASS, POLE_HALF_LEN, POLE_RADIUS);

    for i in 0..CARTPOLE_SWARM_N {
        let z = (i as f64 - (CARTPOLE_SWARM_N as f64 - 1.0) * 0.5) * CARTPOLE_SWARM_SPACING;
        b.begin_articulation();
        // Cart prismatic.
        let cart = b.add_body(
            mp::BodyParams::new()
                .with_spatial_inertia(mp::SpatialInertia {
                    mass: CART_MASS,
                    com: MVec3::ZERO,
                    inertia: cart_inertia,
                })
                .with_position(MVec3::new(0.0, CART_RAIL_Y, z)),
        );
        b.add_shape_box(
            cart,
            MVec3::new(CART_HALF_X, CART_HALF_Y, CART_HALF_Z),
            mp::Transform::IDENTITY,
        )
        .expect("cart box");
        visuals.push(BodyVisual::Box(Vec3::new(
            CART_HALF_X as f32,
            CART_HALF_Y as f32,
            CART_HALF_Z as f32,
        )));
        let x_p_cart = mp::Transform::from_translation(MVec3::new(0.0, CART_RAIL_Y, z));
        b.add_joint_prismatic(
            mp::BodyId::NONE,
            cart,
            x_p_cart,
            mp::Transform::IDENTITY,
            MVec3::X,
        )
        .expect("cart prismatic");
        // Pole revolute.
        let pole = b.add_body(
            mp::BodyParams::new()
                .with_spatial_inertia(mp::SpatialInertia {
                    mass: POLE_MASS,
                    com: MVec3::ZERO,
                    inertia: pole_inertia,
                })
                .with_position(MVec3::new(0.0, CART_RAIL_Y - POLE_HALF_LEN, z)),
        );
        b.add_shape_capsule(pole, POLE_HALF_LEN, POLE_RADIUS, mp::Transform::IDENTITY)
            .expect("pole capsule");
        visuals.push(BodyVisual::Capsule {
            half_h: POLE_HALF_LEN as f32,
            radius: POLE_RADIUS as f32,
        });
        let x_p_pole = mp::Transform::from_translation(MVec3::new(0.0, CART_HALF_Y, 0.0));
        let x_c_pole = mp::Transform::from_translation(MVec3::new(0.0, POLE_HALF_LEN, 0.0));
        b.add_joint_revolute(cart, pole, x_p_pole, x_c_pole, MVec3::Z)
            .expect("pole revolute");
    }

    (b.build().expect("cartpole swarm model"), visuals)
}

fn cartpole_swarm_pane(ui: &mut egui::Ui, sim: &Sim) {
    let pairs = sim.model.body_count / 2;
    readout(ui, "Cart-poles", &format!("{}", pairs));
    readout(ui, "Bodies", &format!("{}", sim.model.body_count));
    ui.separator();
    ui.small("16 independent cart-poles in a row.");
    ui.small("Each balanced by its own PD controller.");
}
