#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RateEncoding {
    Scalar,
    Paired,
}

impl RateEncoding {
    pub(crate) fn bytes(self, dofs: u64) -> u64 {
        dofs * if self == Self::Paired { 8 } else { 4 }
    }
}

fn replace(source: &mut String, from: &str, to: &str) {
    assert_eq!(
        source.matches(from).count(),
        1,
        "paired consumer anchor: {from}"
    );
    *source = source.replace(from, to);
}

pub(crate) fn candidate_source(mut source: String, encoding: RateEncoding) -> String {
    if encoding == RateEncoding::Scalar {
        return source;
    }
    replace(&mut source, "rates:array<f32>", "rates:array<vec2<f32>>");
    replace(
        &mut source,
        "fn certify(@builtin(global_invocation_id) id:vec3<u32>) {",
        "fn certify(@builtin(global_invocation_id) id:vec3<u32>) { rate_lane=id.x%256u;",
    );
    replace(
        &mut source,
        "!finite(rates[i])",
        "!all(abs(rates[i])<=vec2<f32>(3.402823e38))",
    );
    replace(
        &mut source,
        "var machine=0.0;",
        "var machine_pair=vec2<f32>(0.0);",
    );
    replace(
        &mut source,
        "var row=0.0;\n            for (var axis=0u;axis<3u;axis++) { row+=normal[axis]*jacobians[(6u*b+axis)*params.width+i]+moment[axis]*jacobians[(6u*b+axis+3u)*params.width+i]; }\n            machine+=row*rates[i];",
        "var linear=vec3<f32>(0.0); var angular=vec3<f32>(0.0);\n            for (var axis=0u;axis<3u;axis++) { linear[axis]=jacobians[(6u*b+axis)*params.width+i]; angular[axis]=jacobians[(6u*b+axis+3u)*params.width+i]; }\n            machine_pair=normal_rate_add(machine_pair,normal,linear+cross(angular,p-centers[b].xyz),rates[i]);",
    );
    replace(
        &mut source,
        "let slack=(soil-machine)+gap;",
        "let machine=machine_pair.x+machine_pair.y; let residual=normal_rate_residual(velocity,normal,gap,machine_pair); let slack=residual.x;",
    );
    replace(
        &mut source,
        "if slack < -1e-6*max(1.0,max(abs(soil),max(abs(machine),gap)))",
        "if slack < -1e-6*residual.y",
    );
    format!(
        "{}\n{}\n{source}",
        arithmetic(256),
        include_str!("../shaders/soil_normal_rate_contract.wgsl")
    )
}

pub(crate) fn motion_source(encoding: RateEncoding) -> String {
    let mut source = include_str!("../shaders/soil_contact_motion.wgsl").to_owned();
    if encoding == RateEncoding::Scalar {
        return source;
    }
    replace(&mut source, "rates:array<f32>", "rates:array<vec2<f32>>");
    replace(
        &mut source,
        "for (var axis=0u;axis<3u;axis++) {",
        "for (var axis=0u;axis<3u;axis++) { var lp=vec2<f32>(0.0); var ap=vec2<f32>(0.0);",
    );
    replace(
        &mut source,
        "linear[axis]+=jacobians[(6u*b+axis)*params.width+i]*rates[i];\n                angular[axis]+=jacobians[(6u*b+axis+3u)*params.width+i]*rates[i];\n            }",
        "lp=rate_add(lp,rate_mul(jacobians[(6u*b+axis)*params.width+i],rates[i]));\n                ap=rate_add(ap,rate_mul(jacobians[(6u*b+axis+3u)*params.width+i],rates[i]));\n            }\n            linear[axis]=lp.x+lp.y; angular[axis]=ap.x+ap.y;",
    );
    format!("{}\n{source}", arithmetic(1))
}

pub(crate) fn arithmetic(lanes: u32) -> String {
    include_str!("../shaders/soil_rate_consumer_pairs.wgsl")
        .replace("__LANES__", &lanes.to_string())
}

pub(crate) fn sorted_arithmetic(lanes: u32) -> String {
    let mut source = arithmetic(lanes);
    replace(
        &mut source,
        "let s=rate_round(a+b); let vb=rate_round(s-a); let va=rate_round(s-vb);\n    return vec2<f32>(s,rate_round(rate_round(a-va)+rate_round(b-vb)));",
        "let swap=abs(b)>abs(a);let large=select(a,b,swap);let small=select(b,a,swap);\n    let s=rate_round(large+small);let z=rate_round(s-large);\n    return vec2<f32>(s,rate_round(small-z));",
    );
    source
}

#[cfg(test)]
pub(crate) fn native_arithmetic(lanes: u32) -> String {
    include_str!("../shaders/soil_rate_native_pairs.wgsl").replace("__LANES__", &lanes.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn paired_consumer_shader_contracts() {
        let candidates = include_str!("../shaders/soil_candidates.wgsl").replace(
            "// ANALYTIC_GEOMETRY",
            include_str!("../shaders/soil_contact_geometry_common.wgsl"),
        );
        for encoding in [RateEncoding::Scalar, RateEncoding::Paired] {
            for source in [
                candidate_source(candidates.clone(), encoding),
                motion_source(encoding),
            ] {
                let module = wgpu::naga::front::wgsl::parse_str(&source)
                    .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&source)));
                wgpu::naga::valid::Validator::new(
                    wgpu::naga::valid::ValidationFlags::all(),
                    wgpu::naga::valid::Capabilities::all(),
                )
                .validate(&module)
                .unwrap();
            }
        }
    }
}
