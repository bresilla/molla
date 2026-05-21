fn build_nut_bolt_hydro_model() -> (mp::Model, Vec<BodyVisual>) {
    build_nut_bolt_sdf_model()
}

// CartPole scene — identical topology to `NeuralPolicy` (cart on
// prismatic + pole on revolute) but driven by a hand-written PD on
// the pole angle. Port of `example_robot_cartpole.py` /

