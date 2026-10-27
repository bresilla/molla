use super::*;

pub(super) fn unrestricted_model(model: &Model) -> Result<bool> {
    let kinds = model.joint_type.host()?;
    let enabled = model.joint_enabled.host()?;
    let lower = model.joint_limit_lower.host()?;
    let upper = model.joint_limit_upper.host()?;
    let caps = model.joint_velocity_limit.host()?;
    Ok(kinds.len() == model.joint_count
        && enabled.len() == model.joint_count
        && kinds.iter().all(|&kind| {
            kind <= molla_sim::JointType::D6 as u32 && kind != molla_sim::JointType::Distance as u32
        })
        && enabled.iter().all(|&flags| flags & !1 == 0)
        && lower.len() == model.joint_dof_total
        && lower.len() == upper.len()
        && lower.len() == caps.len()
        && lower.iter().all(|&v| v == f64::NEG_INFINITY)
        && upper.iter().all(|&v| v == f64::INFINITY)
        && caps.iter().all(|&v| v == f64::INFINITY))
}

pub(super) fn unrestricted_source(source: &str) -> String {
    let begin = source
        .find("    var has_stops = false;")
        .expect("stop scan begin");
    let end = source[begin..]
        .find("    var has_compliance = false;")
        .unwrap()
        + begin;
    let mut result = source.to_owned();
    result.replace_range(
        begin..end,
        "    var has_stops = false;\n    var mixed_stops = false;\n",
    );
    let begin = result
        .find("fn prepare_drive_locks(")
        .expect("drive lock function");
    let body = result[begin..].find('{').unwrap() + begin;
    let mut depth = 1;
    let end = result[body + 1..]
        .char_indices()
        .find_map(|(i, c)| {
            if c == '{' {
                depth += 1;
            }
            if c == '}' {
                depth -= 1;
            }
            (depth == 0).then_some(body + 1 + i)
        })
        .expect("drive lock function end");
    result.replace_range(body + 1..end, "\n    articulation_drive_locks.count=0u;\n    articulation_drive_locks.rank=0u;\n    root_drive_locks.enabled=false;\n    return true;\n");
    result
}

pub(super) fn split_acceleration_source(source: &str, acceleration: bool) -> String {
    let mut result = unrestricted_source(source);
    let integration = result.find("    // ── Integrate q, qd").unwrap();
    if acceleration {
        let body = result.find("fn featherstone_step(").unwrap();
        let body = result[body..].find('{').unwrap() + body;
        let mut depth = 1;
        let end = result[body + 1..]
            .char_indices()
            .find_map(|(i, c)| {
                if c == '{' {
                    depth += 1;
                }
                if c == '}' {
                    depth -= 1;
                }
                (depth == 0).then_some(body + 1 + i)
            })
            .unwrap();
        result.replace_range(integration..end, "    for (var k=0u; k<n; k++) {\n        let j=j_begin+k;\n        let dof=joint_dof_offset[j];\n        for (var d=0u; d<joint_dof_offset[j+1u]-dof; d++) { velocity_delta[dof+d]=q_dd_arr[k][d]; }\n    }\n");
        result = result.replace("uniforms.drift_only", "0u");
    } else {
        let dynamics = result
            .find("    if uniforms.drift_only == 0u || has_stops")
            .unwrap();
        result.replace_range(dynamics..integration, "    for (var k=0u; k<n; k++) {\n        let j=j_begin+k;\n        let dof=joint_dof_offset[j];\n        n_dof_arr[k]=joint_dof_offset[j+1u]-dof;\n        if uniforms.drift_only==0u {\n            for (var d=0u; d<n_dof_arr[k]; d++) { q_dd_arr[k][d]=velocity_delta[dof+d]; }\n        }\n    }\n\n");
    }
    result
}

pub(super) fn dispatch_source(source: &str, paired: bool) -> String {
    if !paired {
        return source.to_owned();
    }
    let wide = "@compute @workgroup_size(64)";
    let narrow = "@compute @workgroup_size(1)";
    assert_eq!(
        source.matches(wide).count() + source.matches(narrow).count(),
        1
    );
    source.replace(wide, narrow)
}

pub(super) fn external_prediction_source(source: &str) -> String {
    let mut source = split_acceleration_source(source, false);
    for (condition, value) in [
        ("uniforms.drift_only == 0u", "false"),
        ("uniforms.drift_only==0u", "false"),
        ("uniforms.drift_only != 0u", "true"),
        ("uniforms.drift_only!=0u", "true"),
    ] {
        source = source.replace(condition, value);
    }
    source
}

pub(super) fn shader_source(source: &str) -> String {
    let mut result = source.to_owned();
    for (from, to) in [
        (
            "return bitcast<f32>(atomicLoad(&position_error[4u * child + 3u]));",
            "return bitcast<f32>(atomicAdd(&position_error[4u * child + 3u],0u));",
        ),
        (
            "return bitcast<f32>(atomicLoad(&coordinate_error[coord]));",
            "return bitcast<f32>(atomicAdd(&coordinate_error[coord],0u));",
        ),
        (
            "let error = position_round(child, (a - va) + (b - vb));",
            "let error = position_round(child, position_round(child,a-va) + position_round(child,b-vb));",
        ),
        (
            "return vec2<f32>(sum, coordinate_round(coord, (a - virtual_a) + (b - virtual_b)));",
            "return vec2<f32>(sum, coordinate_round(coord,coordinate_round(coord,a-virtual_a)+coordinate_round(coord,b-virtual_b)));",
        ),
    ] {
        assert_eq!(
            result.matches(from).count(),
            1,
            "paired rate rounding anchor"
        );
        result = result.replace(from, to);
    }
    result
}

pub(super) struct RateState {
    pub mirror: wgpu::Buffer,
    pub pairs: wgpu::Buffer,
    pub body_mirror: wgpu::Buffer,
    pub body_low: wgpu::Buffer,
}

fn split(values: &[f64]) -> Result<Vec<[f32; 2]>> {
    values
        .iter()
        .map(|&value| {
            let high = value as f32;
            let low = (value - high as f64) as f32;
            if !high.is_finite() || !low.is_finite() {
                return Err(Error::Solver("nonfinite paired generalized rate".into()));
            }
            Ok([high, low])
        })
        .collect()
}

fn body_seed(state: &State) -> Result<Vec<[f32; 8]>> {
    state
        .body_qd
        .host()?
        .iter()
        .map(|v| {
            let values: Vec<_> = v
                .linear
                .to_array()
                .into_iter()
                .chain([0.0])
                .chain(v.angular.to_array())
                .chain([0.0])
                .collect();
            let pairs = split(&values)?;
            Ok(std::array::from_fn(|i| pairs[i][1]))
        })
        .collect()
}

impl FeatherstoneKernel {
    /// Whether this program uses retained high/low rates.
    pub fn uses_paired_rates(&self) -> bool {
        self.paired_rate_program
    }

    /// Prepare experimental interleaved high/low generalized rates and step deltas.
    /// Coordinate compensation must already cover every state. ABA force evaluation
    /// and external response consumers still use scalar views.
    /// All host seeds are validated before allocation or publication.
    pub fn prepare_paired_rates(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        states: &[&State],
        max_owned_bytes: u64,
    ) -> Result<()> {
        if !states.is_empty() && !self.paired_rate_program {
            return Err(Error::Solver(
                "paired rates require a paired-rate program".into(),
            ));
        }
        let count = self.dof_total as usize;
        let bytes = self.dof_total as u64 * 8;
        let body_bytes = self.body_count as u64 * 32;
        let required = (bytes + body_bytes)
            .checked_mul(states.len() as u64)
            .and_then(|size| size.checked_add(bytes))
            .ok_or_else(|| Error::Solver("paired rate budget overflow".into()))?;
        if !states.is_empty()
            && (required > max_owned_bytes
                || bytes.max(body_bytes) > device.limits().max_storage_buffer_binding_size as u64
                || bytes.max(body_bytes) > device.limits().max_buffer_size)
        {
            return Err(Error::Solver(
                "paired rates exceed retained buffer budget".into(),
            ));
        }
        let mut seeds = Vec::with_capacity(states.len());
        for state in states {
            self.compensated_state(state)?;
            if state.joint_qd.len() != count || state.body_qd.len() != self.body_count {
                return Err(Error::Solver(
                    "invalid paired generalized rate count".into(),
                ));
            }
            let mirror = state.joint_qd.device_buffer_current()?;
            let body_mirror = state.body_qd.device_buffer_current()?;
            if seeds
                .iter()
                .any(|(old, _, body, _)| old == mirror || body == body_mirror)
            {
                return Err(Error::Solver(
                    "duplicate paired generalized rate state".into(),
                ));
            }
            seeds.push((
                mirror.clone(),
                split(state.joint_qd.host()?)?,
                body_mirror.clone(),
                body_seed(state)?,
            ));
        }
        let allocate = |label, size| {
            molla_core::gpu_create_buffer!(
                device,
                &wgpu::BufferDescriptor {
                    label: Some(label),
                    size,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }
            )
        };
        let prepared = seeds
            .into_iter()
            .map(|(mirror, seed, body_mirror, low)| {
                let pairs = allocate("molla retained paired generalized rates", bytes);
                let body_low = allocate("molla retained body velocity residues", body_bytes);
                molla_core::gpu_write_buffer!(queue, &pairs, 0, bytemuck::cast_slice(&seed));
                molla_core::gpu_write_buffer!(queue, &body_low, 0, bytemuck::cast_slice(&low));
                RateState {
                    mirror,
                    pairs,
                    body_mirror,
                    body_low,
                }
            })
            .collect();
        self.paired_delta =
            (!states.is_empty()).then(|| allocate("molla paired generalized rate deltas", bytes));
        self.rate_states = prepared;
        self.bindings = std::array::from_fn(|_| None);
        Ok(())
    }

    /// Reseed after a host rate edit; checkpoint/restore this buffer with the state.
    pub fn seed_paired_rates(&self, queue: &wgpu::Queue, state: &State) -> Result<()> {
        let pairs = self.paired_rates_buffer(state)?;
        let seed = split(state.joint_qd.host()?)?;
        let body = body_seed(state)?;
        let body_buffer = self.body_velocity_compensation_buffer(state)?;
        if seed.len() != self.dof_total as usize || body.len() != self.body_count {
            return Err(Error::Solver(
                "invalid paired generalized rate seed size".into(),
            ));
        }
        molla_core::gpu_write_buffer!(queue, pairs, 0, bytemuck::cast_slice(&seed));
        molla_core::gpu_write_buffer!(queue, body_buffer, 0, bytemuck::cast_slice(&body));
        Ok(())
    }

    /// Normalized vec2<f32> rates. Scalar joint_qd is only their high-part view.
    pub fn paired_rates_buffer(&self, state: &State) -> Result<&wgpu::Buffer> {
        let mirror = state.joint_qd.device_buffer_current()?;
        let body_mirror = state.body_qd.device_buffer_current()?;
        self.rate_states
            .iter()
            .find(|s| &s.mirror == mirror && &s.body_mirror == body_mirror)
            .map(|s| &s.pairs)
            .ok_or_else(|| Error::Solver("unprepared paired generalized rate state".into()))
    }

    /// Per-body spatial velocity residues, two padded vec4<f32>; restore with body_qd.
    pub fn body_velocity_compensation_buffer(&self, state: &State) -> Result<&wgpu::Buffer> {
        let mirror = state.joint_qd.device_buffer_current()?;
        let body = state.body_qd.device_buffer_current()?;
        self.rate_states
            .iter()
            .find(|s| &s.mirror == mirror && &s.body_mirror == body)
            .map(|s| &s.body_low)
            .ok_or_else(|| Error::Solver("unprepared paired body velocity state".into()))
    }

    /// Vec2<f32> predictor changes; corrected drift leaves these unchanged.
    pub fn paired_velocity_delta_buffer(&self) -> Option<&wgpu::Buffer> {
        self.paired_delta.as_ref()
    }

    pub fn paired_rate_owned_bytes(&self) -> u64 {
        self.rate_states
            .iter()
            .map(|s| s.pairs.size() + s.body_low.size())
            .sum::<u64>()
            + self.paired_delta.as_ref().map_or(0, wgpu::Buffer::size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn body_seed_preserves_layout_and_rejects_late_nan() {
        let mut builder = molla_sim::ModelBuilder::new();
        builder.add_body(molla_sim::BodyParams::new());
        builder.add_body(molla_sim::BodyParams::new());
        let model = builder.build().unwrap();
        let mut state = State::for_model(&model).unwrap();
        state.body_qd.host_mut().unwrap()[0].linear.x = 1.0 + 2f64.powi(-30);
        state.body_qd.host_mut().unwrap()[1].angular.z = 2.0 + 2f64.powi(-29);
        let low = body_seed(&state).unwrap();
        assert_eq!(low[0], [2f32.powi(-30), 0., 0., 0., 0., 0., 0., 0.]);
        assert_eq!(low[1], [0., 0., 0., 0., 0., 0., 2f32.powi(-29), 0.]);
        state.body_qd.host_mut().unwrap()[1].angular.z = f64::NAN;
        assert!(body_seed(&state).is_err());
    }

    #[test]
    fn split_retains_sub_ulp_and_rejects_nonfinite() {
        let value = 1.0 + 2f64.powi(-30);
        let parts = split(&[value]).unwrap()[0];
        assert_eq!(parts[0], 1.0);
        assert_eq!(parts[0] as f64 + parts[1] as f64, value);
        for value in [f64::NAN, f64::INFINITY, -f64::INFINITY, f64::MAX] {
            assert!(split(&[1.0, value]).is_err());
        }
    }
}
