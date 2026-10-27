use super::*;

#[path = "mpm_centered_affine_tests.rs"]
mod centered_tests;

fn world_source(input: String) -> String {
    let local = "let xn_xp = (vec3<f32>(f32(i), f32(j), f32(k)) - fx) * dx;";
    assert_eq!(input.matches(local).count(), 3);
    input.replace(local, "let node_pos = u.grid_origin + vec3<f32>(f32(ix), f32(iy), f32(iz)) * dx;\nlet xn_xp = node_pos - pos;")
}

impl MpmSimulation {
    pub(crate) fn inject_late_particle_fault(&mut self) -> wgpu::ComputePipeline {
        let original = self.pipe_g2p.clone();
        let epoch = self.step_epoch.clone();
        self.install_transfer_probe(|source| {
            let marker = "fn g2p(@builtin(global_invocation_id) gid: vec3<u32>) {";
            assert_eq!(source.matches(marker).count(), 1);
            source.replace(marker, &format!("{marker}\n    if gid.x == 0u {{ p_F[0][0][1] = bitcast<f32>(u.grid_size | 0x7f800000u); }}"))
        }, "test late nonfinite deformation after contact solve");
        self.step_epoch = epoch;
        original
    }

    pub(crate) fn restore_test_gather(&mut self, pipeline: wgpu::ComputePipeline) {
        self.pipe_g2p = pipeline;
    }

    pub(super) fn install_world_transfer_coordinates(&mut self) {
        self.install_transfer_probe(world_source, "test legacy world APIC moment arms");
    }

    pub(super) fn install_centered_affine_transfer(&mut self) {
        self.install_transfer_probe(centered_source, "test centered APIC velocity moment");
    }

    fn install_transfer_probe(&mut self, transform: fn(String) -> String, label: &str) {
        assert!(!self.use_deterministic_p2g && !self.use_fused_gather);
        let input = if self.native_float_atomics {
            native_atomics::shader_source()
        } else {
            include_str!("../shaders/mpm.wgsl").to_owned()
        };
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(transform(input).into()),
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
        let pipeline = |entry: &str, deferred: bool| {
            self.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry),
                    layout: Some(&layout),
                    module: &shader,
                    entry_point: Some(entry),
                    compilation_options: wgpu::PipelineCompilationOptions {
                        constants: &[
                            ("SPARSE_GRID", f64::from(self.sparse_grid)),
                            ("DEFER_P2G_LOOKUP", f64::from(deferred)),
                        ],
                        ..Default::default()
                    },
                    cache: None,
                })
        };
        self.pipe_p2g = pipeline("p2g", false);
        self.pipe_p2g_two_pass = pipeline("p2g_two_pass", false);
        self.pipe_g2p = pipeline("g2p", false);
        self.pipe_p2g_deferred = self.sparse_grid.then(|| pipeline("p2g_two_pass", true));
        self.invalidate_prepared_steps();
    }
}

pub(super) fn centered_source(mut input: String) -> String {
    let start = input.find("fn g2p(").unwrap();
    let end = start + input[start..].find("\n}\n").unwrap() + 2;
    let mut body = input[start..end].to_owned();
    let marker = "    var v_new = vec3<f32>(0.0, 0.0, 0.0);";
    assert_eq!(body.matches(marker).count(), 1);
    body = body.replace(marker, &format!("    let complete_stencil = in_bounds(base.x, base.y, base.z) && in_bounds(base.x + 2, base.y + 2, base.z + 2);\n    var affine_reference = vec3<f32>(0.0);\n{marker}"));
    let marker = "                let gv = g_velocity[cidx].xyz;";
    assert_eq!(body.matches(marker).count(), 1);
    body = body.replace(marker, &format!("{marker}\n                if complete_stencil && i == 0u && j == 0u && k == 0u {{ affine_reference = gv; }}\n                let affine_velocity = gv - affine_reference;"));
    assert_eq!(body.matches("scale * gv.").count(), 9);
    body = body.replace("scale * gv.", "scale * affine_velocity.");
    input.replace_range(start..end, &body);
    input
}

#[test]
fn local_transfer_centered_source_contract() {
    for source in [
        include_str!("../shaders/mpm.wgsl").to_owned(),
        native_atomics::shader_source(),
    ] {
        let candidate = centered_source(source.clone());
        let start = source.find("fn g2p(").unwrap();
        let end = source[start..].find("\n}\n").unwrap() + start + 2;
        let candidate_end = candidate[start..].find("\n}\n").unwrap() + start + 2;
        assert_eq!(&source[..start], &candidate[..start]);
        assert_eq!(&source[end..], &candidate[candidate_end..]);
        assert_eq!(candidate.matches("scale * affine_velocity.").count(), 9);
        assert!(candidate.contains("v_new = v_new + w * gv;"));
    }
}

#[test]
#[ignore = "requires a GPU adapter"]
fn local_transfer_centered_affine_fields() {
    affine_fields(MpmSimulation::install_centered_affine_transfer);
}

#[test]
#[ignore = "requires a GPU adapter"]
fn bounded_centered_affine_fields() {
    affine_fields(MpmSimulation::install_bounded_centered_transfer);
}

fn affine_fields(install: fn(&mut MpmSimulation)) {
    use super::replay_tests::read_words;
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut worst_control = 0_f32;
    for origin in [0., -1., 1024., -1024.] {
        for spacing in [0.0625, 0.001] {
            for field in 0..3 {
                if field == 1 && spacing != 0.0625 {
                    continue;
                }
                let mut cfg = MpmConfig::sand();
                cfg.grid_size = 32;
                cfg.grid_origin = [origin; 3];
                cfg.cell_size = spacing;
                cfg.max_particles = 1;
                cfg.youngs = 0.;
                let coordinates = if field == 2 {
                    [0.25, 14.75, 12.5]
                } else {
                    [13.25, 14.75, 12.5]
                };
                let position = coordinates.map(|x| origin + spacing * x);
                let mut baseline = None;
                for centered in [false, true] {
                    let mut sim = MpmSimulation::new(&engine, cfg).unwrap();
                    if centered {
                        install(&mut sim);
                    }
                    sim.set_particles(queue, &[position]).unwrap();
                    let grid: Vec<[f32; 4]> = (0..cfg.grid_size.pow(3))
                        .map(|i| {
                            let p = [(i % 32) as f32, ((i / 32) % 32) as f32, (i / 1024) as f32]
                                .map(|x| x * spacing);
                            if field == 1 {
                                [
                                    4. + p[0] + 0.5 * p[1],
                                    -0.125 + p[1] - 0.25 * p[2],
                                    0.5 + p[2] + 0.25 * p[0],
                                    1.,
                                ]
                            } else {
                                [4., -0.125, 0.5, 1.]
                            }
                        })
                        .collect();
                    queue.write_buffer(&sim.g_velocity, 0, bytemuck::cast_slice(&grid));
                    queue.write_buffer(
                        &sim.uniform_buf,
                        0,
                        bytemuck::bytes_of(&sim.step_uniforms(0.)),
                    );
                    let mut encoder = device.create_command_encoder(&Default::default());
                    {
                        let mut pass = encoder.begin_compute_pass(&Default::default());
                        sim.bind_step(&mut pass);
                        pass.set_pipeline(&sim.pipe_g2p);
                        pass.dispatch_workgroups(1, 1, 1);
                    }
                    queue.submit([encoder.finish()]);
                    let affine = read_words(device, queue, &sim.p_c);
                    let velocity = read_words(device, queue, &sim.p_vel);
                    if !centered {
                        if field == 0 {
                            worst_control = worst_control.max(
                                affine
                                    .iter()
                                    .map(|x| f32::from_bits(*x).abs())
                                    .fold(0., f32::max),
                            );
                        }
                        baseline = Some((affine, velocity));
                        continue;
                    }
                    let (control, control_velocity) = baseline.as_ref().unwrap();
                    assert_eq!(
                        &velocity, control_velocity,
                        "linear gather must remain exact"
                    );
                    if field == 2 {
                        assert_eq!(&affine, control, "clipped stencil retains original moment");
                    } else {
                        let expected = if field == 0 {
                            [0.; 12]
                        } else {
                            [1., 0., 0.25, 0., 0.5, 1., 0., 0., 0., -0.25, 1., 0.]
                        };
                        for (actual, expected) in
                            affine.iter().map(|x| f32::from_bits(*x)).zip(expected)
                        {
                            assert!(actual.is_finite());
                            assert!(
                                (actual - expected).abs() <= 2e-5,
                                "origin={origin} spacing={spacing} field={field}: {actual} != {expected}"
                            );
                            if field == 0 {
                                assert_eq!(actual, 0.);
                            }
                        }
                    }
                    println!(
                        "centered-affine origin={origin} spacing={spacing} field={field} verified"
                    );
                }
            }
        }
    }
    assert!(
        worst_control > 2e-5,
        "constant-field defect witness must reproduce"
    );
    println!("centered-affine constant maximum: control={worst_control} candidate=0");
}

#[test]
fn local_transfer_source_contract() {
    for input in [
        include_str!("../shaders/mpm.wgsl").to_owned(),
        native_atomics::shader_source(),
    ] {
        let candidate = world_source(input.clone());
        assert_eq!(input.matches("let xn_xp = (vec3<f32>").count(), 3);
        assert_eq!(candidate.matches("let xn_xp = node_pos - pos;").count(), 3);
        assert_eq!(
            candidate.matches("var<storage").count(),
            input.matches("var<storage").count()
        );
        assert_eq!(
            candidate.matches("var<workgroup").count(),
            input.matches("var<workgroup").count()
        );
        for entry in ["reset_grid", "grid_update", "plasticity"] {
            let extract = |s: &str| {
                let a = s.find(&format!("fn {entry}(")).unwrap();
                let b = a + s[a..].find("\n}").unwrap();
                s[a..b].to_owned()
            };
            assert_eq!(extract(&candidate), extract(&input));
        }
    }
}

#[test]
#[ignore = "requires a GPU adapter"]
fn local_transfer_uniform_grid_affine_moment() {
    use super::replay_tests::read_words;
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut worst_control = 0_f32;
    let mut worst_candidate = 0_f32;
    for origin in [0., -1., 1024., -1024.] {
        for spacing in [0.0625, 0.001] {
            let mut cfg = MpmConfig::sand();
            cfg.grid_size = 32;
            cfg.grid_origin = [origin; 3];
            cfg.cell_size = spacing;
            cfg.max_particles = 1;
            cfg.youngs = 0.;
            let position = [13.25, 14.75, 12.5].map(|x| origin + spacing * x);
            for candidate in [false, true] {
                let mut sim = MpmSimulation::new(&engine, cfg).unwrap();
                if !candidate {
                    sim.install_world_transfer_coordinates();
                }
                sim.set_particles(queue, &[position]).unwrap();
                let grid = vec![[4_f32, -0.125, 0.5, 1.]; cfg.grid_size.pow(3) as usize];
                queue.write_buffer(&sim.g_velocity, 0, bytemuck::cast_slice(&grid));
                queue.write_buffer(
                    &sim.uniform_buf,
                    0,
                    bytemuck::bytes_of(&sim.step_uniforms(0.)),
                );
                let mut encoder = device.create_command_encoder(&Default::default());
                {
                    let mut pass = encoder.begin_compute_pass(&Default::default());
                    sim.bind_step(&mut pass);
                    pass.set_pipeline(&sim.pipe_g2p);
                    pass.dispatch_workgroups(1, 1, 1);
                }
                queue.submit([encoder.finish()]);
                let affine = read_words(device, queue, &sim.p_c);
                let maximum = affine
                    .into_iter()
                    .map(f32::from_bits)
                    .map(|v| {
                        assert!(v.is_finite());
                        v.abs()
                    })
                    .fold(0_f32, f32::max);
                println!(
                    "local-transfer uniform origin={origin} spacing={spacing} candidate={candidate} affine_error={maximum}"
                );
                if candidate {
                    worst_candidate = worst_candidate.max(maximum);
                } else {
                    worst_control = worst_control.max(maximum);
                }
                let velocity = sim.read_velocities_with(device, queue).unwrap()[0];
                for (actual, expected) in velocity.into_iter().zip([4., -0.125, 0.5]) {
                    assert!((actual - expected).abs() <= 2e-5);
                }
            }
        }
    }
    println!(
        "local-transfer maximum affine error control={worst_control} candidate={worst_candidate}"
    );
    assert!(
        worst_control > 1.,
        "translated witness must reproduce cancellation"
    );
    assert!(worst_candidate < worst_control * 0.001);
}
