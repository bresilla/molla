use super::{instructions, invalid, string};
use molla_core::Result;
use std::collections::BTreeMap;

pub(crate) const MARKER: &str = "// NEWTON_NOINLINE_PHASES\n";
pub(crate) const DIVISION_MARKER: &str = "// NEWTON_NOINLINE_DIVISION\n";
const FUNCTIONS: [&str; 10] = [
    "nonlinear_contact",
    "build_inverse",
    "apply_operator",
    "apply_transpose",
    "apply_direction",
    "publication_certify",
    "release_local",
    "conditioned_prepare",
    "conditioned_factor",
    "krylov_cycle",
];

pub(crate) fn transform(words: &[u32]) -> Result<Vec<u32>> {
    transform_selected(words, &FUNCTIONS)
}

pub(crate) fn transform_division(words: &[u32]) -> Result<Vec<u32>> {
    transform_selected(words, &["fischer_div"])
}

fn transform_selected(words: &[u32], functions: &[&str]) -> Result<Vec<u32>> {
    let code = instructions(words)?;
    let mut names = BTreeMap::new();
    for (_, op) in &code {
        if op[0] & 0xffff == 5 {
            let name = string(&op[2..])?;
            if functions.contains(&name.as_str()) && names.insert(op[1], name).is_some() {
                return Err(invalid("duplicate noinline function name"));
            }
        }
    }
    if names.len() != functions.len()
        || functions
            .iter()
            .any(|name| names.values().filter(|v| v.as_str() == *name).count() != 1)
    {
        return Err(invalid("missing noinline function identity"));
    }
    let mut result = words.to_vec();
    let mut changed = 0;
    for (at, op) in &code {
        if op[0] & 0xffff == 54 && names.contains_key(&op[2]) {
            if op.len() != 5 || op[3] != 0 {
                return Err(invalid("unexpected function control"));
            }
            result[at + 3] = 2;
            changed += 1;
        }
    }
    if changed != functions.len() {
        return Err(invalid("missing noinline function body"));
    }
    Ok(result)
}

#[test]
fn newton_division_noinline_compiler_contract() {
    use crate::soil_newton_gpu::{
        ExperimentalSoilNewtonGpu as Solver, NewtonVariant, shader_source_for,
    };
    let source = crate::soil_newton_parallel_release::schur_inverse::source(
        crate::soil_newton_parallel_release::core_source(
            crate::soil_newton_parallel_release::source(Solver::cached_contact_source(
                Solver::cached_delta_source(Solver::stalled_release_source(shader_source_for(
                    NewtonVariant::Fischer,
                ))),
            )),
        ),
    );
    let source = format!(
        "{}{}256\n{source}",
        super::VOLATILE_MARKER,
        crate::soil_newton_width::MARKER
    );
    let control = super::compile(&source, "newton").unwrap();
    let candidate = super::compile(&format!("{DIVISION_MARKER}{source}"), "newton").unwrap();
    assert_eq!(candidate, transform_division(&control).unwrap());
    assert_eq!(control.len(), candidate.len());
    assert_eq!(
        control
            .iter()
            .zip(&candidate)
            .filter(|(a, b)| a != b)
            .count(),
        1
    );
    assert!(transform_division(&candidate).is_err());
    assert!(transform_division(&[]).is_err());
    assert!(super::compile(&format!("{MARKER}{DIVISION_MARKER}{source}"), "newton").is_err());
    let mut renamed = control.clone();
    let name = instructions(&control)
        .unwrap()
        .iter()
        .find(|(_, op)| op[0] & 0xffff == 5 && string(&op[2..]).unwrap() == "fischer_div")
        .unwrap()
        .0;
    renamed[name + 2] ^= 1;
    assert!(transform_division(&renamed).is_err());
    if let Some(root) = std::env::var_os("MOLLA_NEWTON_DIV_NOINLINE_ARTIFACTS") {
        let root = std::path::PathBuf::from(root);
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("control.wgsl"), &source).unwrap();
        for (name, words) in [("control", &control), ("candidate", &candidate)] {
            std::fs::write(
                root.join(format!("{name}.spv")),
                words
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        }
    }
}

#[test]
fn newton_noinline_compiler_contract() {
    use crate::soil_newton_gpu::{
        ExperimentalSoilNewtonGpu as Solver, NewtonVariant, shader_source_for,
    };
    let source = crate::soil_newton_parallel_release::core_source(
        crate::soil_newton_parallel_release::source(Solver::cached_contact_source(
            Solver::cached_delta_source(Solver::stalled_release_source(shader_source_for(
                NewtonVariant::Fischer,
            ))),
        )),
    );
    let source = format!(
        "{}{}256\n{source}",
        super::VOLATILE_MARKER,
        crate::soil_newton_width::MARKER
    );
    let control = super::compile(&source, "newton").unwrap();
    let candidate = super::compile(&format!("{MARKER}{source}"), "newton").unwrap();
    assert_eq!(candidate, transform(&control).unwrap());
    assert_eq!(control.len(), candidate.len());
    assert_eq!(
        control
            .iter()
            .zip(&candidate)
            .filter(|(a, b)| a != b)
            .count(),
        FUNCTIONS.len()
    );
    assert!(transform(&candidate).is_err());
    assert!(transform(&[]).is_err());
    let mut renamed = control.clone();
    let name = instructions(&control)
        .unwrap()
        .iter()
        .find(|(_, op)| op[0] & 0xffff == 5 && string(&op[2..]).unwrap() == FUNCTIONS[0])
        .unwrap()
        .0;
    renamed[name + 2] ^= 1;
    assert!(transform(&renamed).is_err());
    if let Some(root) = std::env::var_os("MOLLA_NEWTON_NOINLINE_ARTIFACTS") {
        let root = std::path::PathBuf::from(root);
        std::fs::create_dir_all(&root).unwrap();
        for (name, words) in [("control", &control), ("candidate", &candidate)] {
            std::fs::write(
                root.join(format!("{name}.spv")),
                words
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        }
    }
}
