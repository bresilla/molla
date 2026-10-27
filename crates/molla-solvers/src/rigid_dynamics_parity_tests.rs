use super::*;
use crate::kernels::{FeatherstoneExecution, FeatherstoneRounding};
use std::io::Write;

pub(super) struct Probe {
    solver: FeatherstoneSolver,
    output: State,
    staging: wgpu::Buffer,
    checked: usize,
    drift_checked: usize,
    compare_drift: bool,
    drift_pending: bool,
    delta_copy: Option<StorageCopy>,
    trace: std::fs::File,
    directory: std::path::PathBuf,
    reference_execution: FeatherstoneExecution,
}

fn buffers(
    state: &State,
    solver: &FeatherstoneSolver,
) -> Result<Vec<(&'static str, wgpu::Buffer)>> {
    Ok(vec![
        ("body_q", state.body_q.device_buffer_current()?.clone()),
        ("body_qd", state.body_qd.device_buffer_current()?.clone()),
        ("body_f", state.body_f.device_buffer_current()?.clone()),
        ("joint_q", state.joint_q.device_buffer_current()?.clone()),
        ("joint_qd", state.joint_qd.device_buffer_current()?.clone()),
        (
            "coordinate_error",
            solver.gpu_coordinate_compensation_buffer(state)?.clone(),
        ),
        (
            "position_error",
            solver.gpu_position_compensation_buffer(state)?.clone(),
        ),
        (
            "velocity_delta",
            solver.gpu_velocity_delta_buffer().unwrap().clone(),
        ),
        ("status", solver.gpu_status_buffer().unwrap().clone()),
    ])
}

impl Probe {
    fn new(
        engine: &Engine,
        scene: &RigidScene,
        states: &[State; 2],
        directory: std::path::PathBuf,
        reference_execution: FeatherstoneExecution,
        candidate_execution: FeatherstoneExecution,
        candidate_rounding: FeatherstoneRounding,
        compare_drift: bool,
    ) -> Self {
        let mut solver = FeatherstoneSolver::new_gpu_with_options(
            engine,
            scene.model(),
            candidate_rounding,
            candidate_execution,
        )
        .unwrap();
        let mut output = scene.model().state().unwrap();
        output
            .upload_to_device(engine.device().unwrap(), engine.queue().unwrap())
            .unwrap();
        solver
            .prepare_gpu_coordinate_compensation(&[&states[0], &states[1], &output])
            .unwrap();
        let bytes: u64 = buffers(&output, &solver)
            .unwrap()
            .iter()
            .map(|(_, b)| b.size())
            .sum();
        let staging = engine
            .device()
            .unwrap()
            .create_buffer(&wgpu::BufferDescriptor {
                label: Some("identical-input dynamics comparison"),
                size: 2 * bytes,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
        let trace = std::fs::File::create_new(directory.join("steps.jsonl")).unwrap();
        Self {
            solver,
            output,
            staging,
            checked: 0,
            drift_checked: 0,
            compare_drift,
            drift_pending: false,
            delta_copy: None,
            trace,
            directory,
            reference_execution,
        }
    }

    pub(super) fn check(
        &mut self,
        engine: &Engine,
        reference: &FeatherstoneSolver,
        scene: &RigidScene,
        input: &State,
        expected: &State,
        dt: Real,
    ) -> Result<()> {
        assert_eq!(reference.substeps, 1);
        assert_eq!(reference.gpu_execution(), Some(self.reference_execution));
        let device = engine.device().unwrap();
        let queue = engine.queue().unwrap();
        let mut encoder = device.create_command_encoder(&Default::default());
        for (source, target) in [
            (
                reference.gpu_coordinate_compensation_buffer(input)?,
                self.solver.gpu_coordinate_compensation_buffer(input)?,
            ),
            (
                reference.gpu_position_compensation_buffer(input)?,
                self.solver.gpu_position_compensation_buffer(input)?,
            ),
        ] {
            encoder.copy_buffer_to_buffer(source, 0, target, 0, source.size());
        }
        queue.submit([encoder.finish()]);
        self.solver.joint_damping = reference.joint_damping;
        self.solver.rigorous_d_solve = reference.rigorous_d_solve;
        self.solver.rigorous_distance = reference.rigorous_distance;
        self.solver.stable_motor_feedback = reference.stable_motor_feedback;
        self.solver.set_gpu_active_articulations(&[true])?;
        self.solver.set_external_scalar_bounds(false)?;
        self.solver.step(
            scene.model(),
            input,
            scene.control(),
            &Contacts::with_capacity(0),
            dt,
            &mut self.output,
        )?;
        self.compare(engine, reference, expected, dt, false)
    }

    pub(super) fn before_drift(
        &mut self,
        engine: &Engine,
        reference: &FeatherstoneSolver,
        scene: &RigidScene,
        input: &State,
        projected: &State,
        dt: Real,
    ) -> Result<()> {
        if !self.compare_drift {
            return Ok(());
        }
        assert!(!self.drift_pending);
        let mut encoder = engine
            .device()
            .unwrap()
            .create_command_encoder(&Default::default());
        for (source, target) in [
            (
                reference.gpu_coordinate_compensation_buffer(input)?,
                self.solver.gpu_coordinate_compensation_buffer(input)?,
            ),
            (
                reference.gpu_position_compensation_buffer(input)?,
                self.solver.gpu_position_compensation_buffer(input)?,
            ),
            (
                projected.joint_qd.device_buffer_current()?,
                self.output.joint_qd.device_buffer_current()?,
            ),
            (
                reference.gpu_status_buffer().unwrap(),
                self.solver.gpu_status_buffer().unwrap(),
            ),
        ] {
            assert_eq!(source.size(), target.size());
            assert!(source.usage().contains(wgpu::BufferUsages::COPY_SRC));
            assert!(target.usage().contains(wgpu::BufferUsages::COPY_DST));
            encoder.copy_buffer_to_buffer(source, 0, target, 0, source.size());
        }
        let source = reference.gpu_velocity_delta_buffer().unwrap();
        let target = self.solver.gpu_velocity_delta_buffer().unwrap();
        let copy = self
            .delta_copy
            .get_or_insert_with(|| StorageCopy::new(engine, source, target));
        assert_eq!(&copy.source, source);
        assert_eq!(&copy.target, target);
        copy.encode(&mut encoder);
        engine.queue().unwrap().submit([encoder.finish()]);
        self.solver
            .drift_gpu(scene.model(), input, scene.control(), dt, &mut self.output)?;
        self.drift_pending = true;
        Ok(())
    }

    pub(super) fn after_drift(
        &mut self,
        engine: &Engine,
        reference: &FeatherstoneSolver,
        expected: &State,
        dt: Real,
    ) -> Result<()> {
        if !self.compare_drift {
            return Ok(());
        }
        assert!(self.drift_pending);
        self.drift_pending = false;
        self.compare(engine, reference, expected, dt, true)
    }

    fn compare(
        &mut self,
        engine: &Engine,
        reference: &FeatherstoneSolver,
        expected: &State,
        dt: Real,
        drift: bool,
    ) -> Result<()> {
        let device = engine.device().unwrap();
        let queue = engine.queue().unwrap();
        let left = buffers(expected, reference)?;
        let right = buffers(&self.output, &self.solver)?;
        let mut encoder = device.create_command_encoder(&Default::default());
        let mut offset = 0;
        for ((name, a), (other, b)) in left.iter().zip(&right) {
            assert_eq!(name, other);
            assert_eq!(a.size(), b.size());
            for source in [a, b] {
                encoder.copy_buffer_to_buffer(source, 0, &self.staging, offset, source.size());
                offset += source.size();
            }
        }
        assert_eq!(offset, self.staging.size());
        queue.submit([encoder.finish()]);
        let slice = self.staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(std::time::Duration::from_secs(10)),
            })
            .unwrap();
        rx.recv_timeout(std::time::Duration::from_secs(10))
            .unwrap()
            .unwrap();
        let bytes = slice.get_mapped_range();
        let words: &[u32] = bytemuck::cast_slice(&bytes);
        let mut offset = 0;
        let mut mismatches = Vec::new();
        let mut compared = 0;
        for (name, source) in &left {
            let count = source.size() as usize / 4;
            for index in 0..count {
                if name == &"position_error" && index % 4 == 3 {
                    continue;
                }
                let a = words[offset + index];
                let b = words[offset + count + index];
                if a != b || (name == &"status" && a != 0) {
                    mismatches.push(
                        serde_json::json!({"field":name,"word":index,"reference":a,"candidate":b}),
                    );
                }
                compared += 1;
            }
            offset += 2 * count;
        }
        let stage = if drift { "corrected-drift" } else { "dynamics" };
        let index = if drift {
            self.drift_checked
        } else {
            self.checked
        };
        let record = serde_json::json!({"stage":stage,"prediction":index,"dt":dt,"compared_words":compared,"mismatches":mismatches});
        writeln!(self.trace, "{record}").unwrap();
        self.trace.flush().unwrap();
        if !mismatches.is_empty() {
            std::fs::write(self.directory.join("first-mismatch.bin"), &*bytes).unwrap();
        }
        drop(bytes);
        self.staging.unmap();
        if drift {
            self.drift_checked += 1;
        } else {
            self.checked += 1;
        }
        if !mismatches.is_empty() {
            return Err(Error::Solver(format!(
                "identical-input {stage} mismatch: {record}"
            )));
        }
        Ok(())
    }
}

struct StorageCopy {
    source: wgpu::Buffer,
    target: wgpu::Buffer,
    pipeline: wgpu::ComputePipeline,
    bindings: wgpu::BindGroup,
}

impl StorageCopy {
    fn new(engine: &Engine, source: &wgpu::Buffer, target: &wgpu::Buffer) -> Self {
        assert_eq!(source.size(), target.size());
        assert_ne!(source, target);
        assert_eq!(source.size() % 4, 0);
        assert!(source.usage().contains(wgpu::BufferUsages::STORAGE));
        assert!(target.usage().contains(wgpu::BufferUsages::STORAGE));
        let device = engine.device().unwrap();
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("test storage-only word copy"),
            source: wgpu::ShaderSource::Wgsl(
                "@group(0) @binding(0) var<storage,read> input:array<u32>;\n\
                 @group(0) @binding(1) var<storage,read_write> output:array<u32>;\n\
                 @compute @workgroup_size(64) fn copy(@builtin(global_invocation_id) id:vec3<u32>) {\n\
                 if id.x<arrayLength(&input) { output[id.x]=input[id.x]; } }".into(),
            ),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &shader,
            entry_point: Some("copy"),
            compilation_options: Default::default(),
            cache: None,
        });
        let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: source.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: target.as_entire_binding(),
                },
            ],
        });
        Self {
            source: source.clone(),
            target: target.clone(),
            pipeline,
            bindings,
        }
    }

    fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bindings, &[]);
        pass.dispatch_workgroups((self.source.size() / 4).div_ceil(64) as u32, 1, 1);
    }
}

#[test]
#[ignore = "requires GPU"]
fn dynamics_probe_copies_storage_only_delta_bits() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let values: Vec<u32> = (0u32..65)
        .map(|i| 0xff800001u32.wrapping_add(i.wrapping_mul(0x04001231)))
        .collect();
    let buffer = |usage| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: values.len() as u64 * 4,
            usage,
            mapped_at_creation: false,
        })
    };
    let source = buffer(wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST);
    let target = buffer(wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
    let staging = buffer(wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST);
    queue.write_buffer(&source, 0, bytemuck::cast_slice(&values));
    let copy = StorageCopy::new(&engine, &source, &target);
    let mut encoder = device.create_command_encoder(&Default::default());
    copy.encode(&mut encoder);
    encoder.copy_buffer_to_buffer(&target, 0, &staging, 0, target.size());
    queue.submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        tx.send(r).unwrap();
    });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(5)),
        })
        .unwrap();
    rx.recv_timeout(std::time::Duration::from_secs(5))
        .unwrap()
        .unwrap();
    let bytes = staging.slice(..).get_mapped_range();
    assert_eq!(bytemuck::cast_slice::<u8, u32>(&bytes), values);
    drop(bytes);
    staging.unmap();
}

#[test]
#[ignore = "requires real GPU native atomics and actual 50k tractor with both cold machine profiles"]
fn live_tractor_dynamics_matches_identical_inputs_through_frame_twenty() {
    live_comparison(
        FeatherstoneExecution::Generic,
        FeatherstoneExecution::ExperimentalWorkgroupScratch,
        20,
    );
}

#[test]
#[ignore = "requires native GPU and both scratch profiles on an actual 50k tractor"]
fn live_constraint_scratch_matches_partial_through_frame_thirty_three() {
    live_comparison(
        FeatherstoneExecution::ExperimentalWorkgroupScratch,
        FeatherstoneExecution::ExperimentalConstraintScratch,
        33,
    );
}

fn live_comparison(
    reference: FeatherstoneExecution,
    candidate: FeatherstoneExecution,
    frames: usize,
) {
    let directory = std::path::PathBuf::from(
        std::env::var_os("SOIL_DYNAMICS_PARITY_DIR").expect("fresh output directory required"),
    );
    std::fs::create_dir(&directory).unwrap();
    std::fs::write(directory.join("manifest.json"), serde_json::to_vec_pretty(&serde_json::json!({
        "schema":1,"reference":reference.label(),"candidate":candidate.label(),"frames":frames,
        "particles":50000,"substeps_per_frame":34,"frame_seconds":1.0/60.0,
        "scope":"identical live machine prediction inputs; not trajectory or performance acceptance"
    })).unwrap()).unwrap();
    let engine =
        Arc::new(Engine::new_gpu_with_features(wgpu::Features::SHADER_FLOAT32_ATOMIC).unwrap());
    assert!(!engine.adapter_info().unwrap().is_software);
    eprintln!("building authoritative {} 50k tractor", reference.label());
    let mut tractor = crate::soil_tractor::SoilTractor::new_native_explicit_with_machine_options(
        engine.clone(),
        crate::soil_tractor::SoilTractorConfig::default(),
        FeatherstoneRounding::Storage,
        reference,
    )
    .unwrap();
    tractor.enable_mass_coupling(4096).unwrap();
    eprintln!("building isolated {} prediction probe", candidate.label());
    let world = tractor.world_mut_for_test();
    let gpu = world.gpu.as_mut().unwrap();
    gpu.dynamics_probe = Some(Probe::new(
        &engine,
        &world.scene,
        &gpu.compiled.as_ref().unwrap().states,
        directory,
        reference,
        candidate,
        FeatherstoneRounding::Storage,
        false,
    ));
    for frame in 0..frames {
        tractor.step(1.0 / 60.0, [300.0; 4]).unwrap();
        let world = tractor.world_mut_for_test();
        let checked = world
            .gpu
            .as_ref()
            .unwrap()
            .dynamics_probe
            .as_ref()
            .unwrap()
            .checked;
        assert_eq!(checked, (frame + 1) * 34);
        println!(
            "committed frame {}: {checked} identical-input machine predictions match",
            frame + 1
        );
    }
    assert_eq!(
        tractor.world().mpm_soil_stats().unwrap().committed_frames,
        frames as u64
    );
}

#[test]
#[ignore = "requires native GPU and two cold profiles for full 50k unilateral diagnostic"]
fn live_shared_rounding_matches_dynamics_and_corrected_drift() {
    unilateral_live_comparison(
        FeatherstoneExecution::Generic,
        FeatherstoneRounding::ExperimentalShared,
        33,
    );
}

#[test]
#[ignore = "requires native GPU and both cold execution profiles on the actual 50k tractor"]
fn single_lane_matches_identical_unilateral_dynamics_and_drift() {
    unilateral_live_comparison(
        FeatherstoneExecution::ExperimentalSingleLane,
        FeatherstoneRounding::Storage,
        33,
    );
}

fn unilateral_live_comparison(
    candidate_execution: FeatherstoneExecution,
    candidate_rounding: FeatherstoneRounding,
    frames: usize,
) {
    let directory = std::path::PathBuf::from(
        std::env::var_os("SOIL_DYNAMICS_PARITY_DIR").expect("fresh output directory required"),
    );
    std::fs::create_dir(&directory).unwrap();
    std::fs::write(directory.join("manifest.json"), serde_json::to_vec_pretty(&serde_json::json!({
        "schema":1,"reference_rounding":FeatherstoneRounding::Storage.label(),
        "candidate_rounding":candidate_rounding.label(),
        "reference_execution":FeatherstoneExecution::Generic.label(),
        "candidate_execution":candidate_execution.label(),"frames":frames,"particles":50000,
        "substeps_per_frame":34,"frame_seconds":1.0/60.0,"coupling":"unilateral",
        "scope":"identical resident inputs for dynamics and corrected drift; candidate discarded; no trajectory or timing acceptance"
    })).unwrap()).unwrap();
    let engine =
        Arc::new(Engine::new_gpu_with_features(wgpu::Features::SHADER_FLOAT32_ATOMIC).unwrap());
    assert!(!engine.adapter_info().unwrap().is_software);
    eprintln!("building authoritative storage-rounding 50k tractor");
    let mut tractor = crate::soil_tractor::SoilTractor::new_native_explicit(
        engine.clone(),
        crate::soil_tractor::SoilTractorConfig::default(),
    )
    .unwrap();
    tractor
        .enable_unilateral_coupling(crate::mpm::MpmUnilateralConfig {
            node_capacity: 4096,
            search_distance: 0.05,
            max_iterations: 64,
        })
        .unwrap();
    let world = tractor.world_mut_for_test();
    let gpu = world.gpu.as_mut().unwrap();
    eprintln!(
        "building isolated {} / {} dynamics/drift probe",
        candidate_execution.label(),
        candidate_rounding.label()
    );
    gpu.dynamics_probe = Some(Probe::new(
        &engine,
        &world.scene,
        &gpu.compiled.as_ref().unwrap().states,
        directory.clone(),
        FeatherstoneExecution::Generic,
        candidate_execution,
        candidate_rounding,
        true,
    ));
    for frame in 0..frames {
        tractor.step(1.0 / 60.0, [300.0; 4]).unwrap();
        let probe = tractor
            .world_mut_for_test()
            .gpu
            .as_ref()
            .unwrap()
            .dynamics_probe
            .as_ref()
            .unwrap();
        assert_eq!(probe.checked, (frame + 1) * 34);
        assert_eq!(probe.drift_checked, (frame + 1) * 34);
        assert!(!probe.drift_pending);
        eprintln!(
            "committed frame {}: dynamics={} drift={} identical-input checks",
            frame + 1,
            probe.checked,
            probe.drift_checked
        );
    }
    assert_eq!(
        tractor.world().mpm_soil_stats().unwrap().committed_frames,
        frames as u64
    );
    std::fs::write(
        directory.join("complete.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "committed_frames":frames,"dynamics_checked":frames*34,"corrected_drift_checked":frames*34,
            "mismatches":0,"production_accepted":false,"timing_accepted":false
        }))
        .unwrap(),
    )
    .unwrap();
}
