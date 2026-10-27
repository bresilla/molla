use molla_core::{Error, Result};

pub(crate) const MARKER: &str = "// NEWTON_EXECUTION_LANES:";

fn invalid() -> Error {
    Error::Solver("Newton width probe: unsupported width or source contract".into())
}

pub(crate) fn specialize(mut source: String) -> Result<String> {
    if !source.contains(MARKER) {
        return Ok(source);
    }
    if source.matches(MARKER).count() != 1 {
        return Err(invalid());
    }
    let lanes = source
        .split_once(MARKER)
        .unwrap()
        .1
        .lines()
        .next()
        .ok_or_else(invalid)?
        .parse::<u32>()
        .map_err(|_| invalid())?;
    if !matches!(lanes, 128 | 256) {
        return Err(invalid());
    }
    for name in ["slot", "c", "index", "i", "j"] {
        let token = format!("{name}+=64u");
        for (at, _) in source.match_indices(&token) {
            let prefix = &source[..at];
            let start = prefix.rfind("for").ok_or_else(invalid)?;
            if !prefix[start..].contains(&format!("var {name}=lane;")) {
                return Err(Error::Solver(format!(
                    "Newton width probe: non-lane stride {}",
                    &prefix[start..]
                )));
            }
        }
        source = source.replace(&token, &format!("{name}+={lanes}u"));
    }
    if source.contains("+=64u") {
        return Err(Error::Solver(format!(
            "Newton width probe: remaining stride {:?}",
            source
                .lines()
                .filter(|l| l.contains("+=64u") || l.contains("rate_lane=c%64u"))
                .collect::<Vec<_>>()
        )));
    }
    let reduction = "i<64u;i++) {";
    for (at, _) in source.match_indices(reduction) {
        if !source[at..].lines().next().unwrap().contains("parts[i]") {
            return Err(Error::Solver(format!(
                "Newton width probe: unknown reduction {}",
                source[at..].lines().next().unwrap()
            )));
        }
    }
    source = source.replace(reduction, &format!("i<{lanes}u;i++) {{"));
    for (old, new) in [
        (
            "@workgroup_size(64)\nfn newton",
            format!("@workgroup_size({lanes})\nfn newton"),
        ),
        (
            "let chunks=64u/params.width;",
            format!("let chunks={lanes}u/params.width;"),
        ),
        (
            "var<workgroup> parts:array<vec2<f32>,64>;",
            format!("var<workgroup> parts:array<vec2<f32>,{lanes}>;"),
        ),
        (
            "var<workgroup> publication_energy_parts:array<vec2<f32>,64>;",
            format!("var<workgroup> publication_energy_parts:array<vec2<f32>,{lanes}>;"),
        ),
        (
            "var<workgroup> publication_scales:array<f32,64>;",
            format!("var<workgroup> publication_scales:array<f32,{lanes}>;"),
        ),
        (
            "var<workgroup> rate_scratch:array<atomic<u32>,64>;",
            format!("var<workgroup> rate_scratch:array<atomic<u32>,{lanes}>;"),
        ),
        (
            "for (var stride=32u;stride>0u;stride/=2u)",
            format!("for (var stride={}u;stride>0u;stride/=2u)", lanes / 2),
        ),
    ] {
        if source.matches(old).count() != 1 {
            return Err(Error::Solver(format!(
                "Newton width probe: expected one {old}"
            )));
        }
        source = source.replacen(old, &new, 1);
    }
    Ok(source)
}

#[test]
fn newton_width_source_preserves_mathematical_budgets() {
    use crate::soil_newton_gpu::{
        ExperimentalSoilNewtonGpu as Solver, NewtonVariant, shader_source_for,
    };
    let source = Solver::stalled_release_source(shader_source_for(NewtonVariant::Fischer));
    if let Some(root) = std::env::var_os("MOLLA_NEWTON_WIDTH_SOURCE_DIR") {
        std::fs::write(
            std::path::PathBuf::from(root).join("width-64.wgsl"),
            &source,
        )
        .unwrap();
    }
    assert_eq!(specialize(source.clone()).unwrap(), source);
    for lanes in [128, 256] {
        let marked = format!("{MARKER}{lanes}\n{source}");
        let widened = specialize(marked.clone()).unwrap();
        let module = wgpu::naga::front::wgsl::parse_str(&widened).unwrap();
        assert_eq!(module.entry_points.len(), 2);
        assert_eq!(
            module
                .entry_points
                .iter()
                .find(|e| e.name == "newton")
                .unwrap()
                .workgroup_size,
            [lanes, 1, 1]
        );
        assert_eq!(
            module
                .entry_points
                .iter()
                .find(|e| e.name == "prepare")
                .unwrap()
                .workgroup_size,
            [64, 1, 1]
        );
        for invariant in [
            "h:array<f32,4160>",
            "rotations:array<vec2<f32>,64>",
            "coefficients:array<f32,64>",
            "h[i*64u+j]",
            "backtrack<16u",
            "attempt==8u",
            "outer.releases)>=8u",
        ] {
            assert!(widened.contains(invariant), "{invariant}");
        }
        let words = crate::soil_newton_protected::compile(&marked, "newton").unwrap();
        if let Some(root) = std::env::var_os("MOLLA_NEWTON_WIDTH_SOURCE_DIR") {
            let root = std::path::PathBuf::from(root);
            std::fs::write(root.join(format!("width-{lanes}.wgsl")), &widened).unwrap();
            std::fs::write(
                root.join(format!("width-{lanes}.spv")),
                bytemuck::cast_slice(&words),
            )
            .unwrap();
        }
        assert!(
            specialize(marked.replace("parts:array<vec2<f32>,64>", "parts:array<vec2<f32>,32>"))
                .is_err()
        );
    }
    assert!(specialize(format!("{MARKER}32\n{source}")).is_err());
    assert!(specialize(MARKER.to_owned()).is_err());
    assert!(specialize(format!("{MARKER}128\n{MARKER}256\n{source}")).is_err());
}
