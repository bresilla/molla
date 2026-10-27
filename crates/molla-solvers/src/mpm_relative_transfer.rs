use super::*;

#[path = "mpm_transfer_timing.rs"]
mod timing;

fn source(mut source: String) -> String {
    for (old, new, count) in [
        (
            "let vel = p_vel[pid].xyz;",
            "let vel = p_vel[pid].xyz - transfer_reference();",
            2,
        ),
        (
            "let mom_new = stamp_mass * v;",
            "let mom_new = stamp_mass * (v - transfer_reference());",
            1,
        ),
        (
            "var impulse_to_sand = m_old * v - mom_old;",
            "var impulse_to_sand = m_old * (v - transfer_reference()) - mom_old;",
            1,
        ),
    ] {
        assert_eq!(source.matches(old).count(), count);
        source = source.replace(old, new);
    }
    let start = source.find("fn grid_update(").unwrap();
    let a = start + source[start..].find("    let m_bits =").unwrap();
    let b = start
        + source[start..]
            .find("    // Skip integration on rigid-imprinted cells.")
            .unwrap();
    source.replace_range(
        a..b,
        r#"    let m = bitcast<f32>(atomicLoad(&g_mass[cidx]));
    let reference = transfer_reference();
    let mb = cidx * 3u;
    let relative = vec3<f32>(
        bitcast<f32>(atomicLoad(&g_momentum[mb])),
        bitcast<f32>(atomicLoad(&g_momentum[mb + 1u])),
        bitcast<f32>(atomicLoad(&g_momentum[mb + 2u])),
    );
    let absolute = relative + m * reference;
    atomicStore(&g_momentum[mb], bitcast<GridAtomicScalar>(absolute.x));
    atomicStore(&g_momentum[mb + 1u], bitcast<GridAtomicScalar>(absolute.y));
    atomicStore(&g_momentum[mb + 2u], bitcast<GridAtomicScalar>(absolute.z));
    if (m <= 1.0e-9 || !(m == m)) {
        g_velocity[cidx] = vec4<f32>(0.0);
        return;
    }
    var vel = relative / m + reference;

"#,
    );
    source.push_str(
        r#"
fn transfer_reference()->vec3<f32> {
    if u.particle_count == 0u { return vec3<f32>(0.0); }
    let velocity = p_vel[0].xyz;
    return select(vec3<f32>(0.0), velocity, all(abs(velocity) <= vec3<f32>(3.402823e38)));
}
"#,
    );
    source
}

impl MpmSimulation {
    pub(super) fn install_relative_momentum_transfer(&mut self) {
        self.install_relative_probe(false, false);
    }

    pub(super) fn install_bounded_relative_transfer(&mut self) {
        self.bounded_reference = Some(super::bounded_reference::Reference::new(self));
        self.install_relative_probe(true, false);
    }

    pub(super) fn install_bounded_centered_transfer(&mut self) {
        self.bounded_reference = Some(super::bounded_reference::Reference::new(self));
        self.install_relative_probe(true, true);
    }

    fn install_relative_probe(&mut self, bounded: bool, centered: bool) {
        assert!(!self.use_deterministic_p2g && !self.use_fused_gather);
        let source = source(if self.native_float_atomics {
            native_atomics::shader_source()
        } else {
            include_str!("../shaders/mpm.wgsl").to_owned()
        });
        let source = if bounded {
            super::bounded_reference::source(source)
        } else {
            source
        };
        let source = if centered {
            super::local_transfer::centered_source(source)
        } else {
            source
        };
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("test relative-momentum APIC"),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
        let layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[
                    Some(&self.layout_uniforms),
                    Some(&self.layout_particles),
                    Some(&self.layout_grid),
                    Some(&self.layout_rigid),
                ],
                immediate_size: 0,
            });
        let pipeline = |entry: &str, active: bool, deferred: bool| {
            self.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry),
                    layout: Some(&layout),
                    module: &shader,
                    entry_point: Some(entry),
                    compilation_options: wgpu::PipelineCompilationOptions {
                        constants: &[
                            ("ACTIVE_GRID", f64::from(active)),
                            ("SPARSE_GRID", f64::from(self.sparse_grid)),
                            ("DEFER_P2G_LOOKUP", f64::from(deferred)),
                        ],
                        ..Default::default()
                    },
                    cache: None,
                })
        };
        self.pipe_p2g = pipeline("p2g", false, false);
        self.pipe_p2g_two_pass = pipeline("p2g_two_pass", false, false);
        if centered {
            self.pipe_g2p = pipeline("g2p", false, false);
        }
        self.pipe_p2g_deferred = self
            .sparse_grid
            .then(|| pipeline("p2g_two_pass", false, true));
        self.pipe_rigid = pipeline("rigid_imprint", false, false);
        self.pipe_grid = pipeline("grid_update", false, false);
        self.active_pipelines[1] = pipeline("rigid_imprint", true, false);
        self.active_pipelines[2] = pipeline("grid_update", true, false);
        self.invalidate_prepared_steps();
    }
}

#[test]
fn relative_transfer_source_contract() {
    for input in [
        include_str!("../shaders/mpm.wgsl").to_owned(),
        native_atomics::shader_source(),
    ] {
        let candidate = source(input.clone());
        for marker in ["var<storage", "var<workgroup>"] {
            assert_eq!(
                input.matches(marker).count(),
                candidate.matches(marker).count()
            );
        }
        for name in [
            "reset_grid",
            "g2p",
            "plasticity",
            "mpm_implicit_apply",
            "mpm_rigid_force_integrate",
        ] {
            let body = |s: &str| {
                let a = s.find(&format!("fn {name}(")).unwrap();
                let b = a + s[a..].find("\n}\n").unwrap();
                s[a..b].to_owned()
            };
            assert_eq!(body(&input), body(&candidate));
        }
        assert!(candidate.contains("var vel = relative / m + reference;"));
        assert!(candidate.contains("let absolute = relative + m * reference;"));
    }
}

#[test]
#[ignore = "requires a GPU adapter; checks first-particle reference dynamic range"]
fn relative_transfer_reference_range() {
    reference_range(MpmSimulation::install_relative_momentum_transfer);
}

#[test]
fn bounded_centered_source_contract() {
    let candidate = super::local_transfer::centered_source(super::bounded_reference::source(
        source(include_str!("../shaders/mpm.wgsl").to_owned()),
    ));
    let module = naga::front::wgsl::parse_str(&candidate).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
    assert_eq!(candidate.matches("scale * affine_velocity.").count(), 9);
    assert_eq!(
        candidate
            .matches("let vel = p_vel[pid].xyz - transfer_reference();")
            .count(),
        2
    );
    assert!(candidate.contains("v_new = v_new + w * gv;"));
    assert!(candidate.contains("let flags = u.reference_flags.xyz;"));
}

#[test]
#[ignore = "requires a GPU adapter"]
fn bounded_centered_velocity_range() {
    reference_range(MpmSimulation::install_bounded_centered_transfer);
}

#[test]
#[ignore = "requires a GPU adapter"]
fn bounded_reference_velocity_range() {
    reference_range(MpmSimulation::install_bounded_relative_transfer);
}

fn reference_range(install: fn(&mut MpmSimulation)) {
    use super::replay_tests::read_words;
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut failures = 0;
    for reference in [0_f32, 1024., 1e8] {
        for candidate in [false, true] {
            let mut cfg = MpmConfig::sand();
            cfg.grid_size = 32;
            cfg.grid_origin = [-1.; 3];
            cfg.cell_size = 0.125;
            cfg.max_particles = 2;
            cfg.particle_mass = 1.;
            cfg.gravity_y = 0.;
            cfg.youngs = 0.;
            let slow = [0.0001_f32, -0.0001, 0.01];
            let mut sim = MpmSimulation::new(&engine, cfg).unwrap();
            if candidate {
                install(&mut sim);
            }
            sim.set_particles_with_velocity(
                queue,
                &[[-0.5; 3], [1.; 3]],
                &[[reference, -reference, reference], slow],
            )
            .unwrap();
            queue.write_buffer(
                &sim.uniform_buf,
                0,
                bytemuck::bytes_of(&sim.step_uniforms(0.)),
            );
            queue.write_buffer(
                &sim.material_table,
                0,
                bytemuck::cast_slice(&sim.step_material_table()),
            );
            let mut encoder = device.create_command_encoder(&Default::default());
            sim.encode_bounded_reference(&mut encoder);
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                sim.bind_step(&mut pass);
                for (pipeline, groups) in [
                    (&sim.pipe_reset, 32_u32.pow(3).div_ceil(WORKGROUP_SIZE)),
                    (sim.p2g_pipeline(), 1),
                    (&sim.pipe_grid, 32_u32.pow(3).div_ceil(WORKGROUP_SIZE)),
                ] {
                    pass.set_pipeline(pipeline);
                    pass.dispatch_workgroups(groups, 1, 1);
                }
            }
            queue.submit([encoder.finish()]);
            let index = 16 + 32 * (16 + 32 * 16);
            let velocity = read_words(device, queue, &sim.g_velocity);
            let momentum = read_words(device, queue, &sim.g_momentum);
            let actual: [f32; 3] =
                std::array::from_fn(|axis| f32::from_bits(velocity[index * 4 + axis]));
            let absolute: [f32; 3] =
                std::array::from_fn(|axis| f32::from_bits(momentum[index * 3 + axis]));
            assert!(actual.into_iter().chain(absolute).all(f32::is_finite));
            let error = actual
                .into_iter()
                .zip(slow)
                .map(|(a, b)| (a - b).abs())
                .fold(0_f32, f32::max);
            let momentum_error = absolute
                .into_iter()
                .zip(slow)
                .map(|(a, b)| (a - 0.421875 * b).abs())
                .fold(0_f32, f32::max);
            let passed = error <= 2e-5 && momentum_error <= 2e-5;
            println!(
                "relative-range reference={reference} candidate={candidate} velocity={actual:?} absolute_momentum={absolute:?} velocity_error={error} momentum_error={momentum_error} passed={passed}"
            );
            if candidate {
                failures += usize::from(!passed);
            } else {
                assert!(passed, "original analytic uniform single-particle transfer");
            }
        }
    }
    assert_eq!(
        failures, 0,
        "first-particle reference loses small velocities at large dynamic range"
    );
}
