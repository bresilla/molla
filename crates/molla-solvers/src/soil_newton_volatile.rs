use super::*;

pub(crate) fn transform(words: &[u32]) -> Result<Vec<u32>> {
    let code = instructions(words)?;
    let mut names = BTreeMap::new();
    let mut unsigned = BTreeSet::new();
    let mut integers = BTreeSet::new();
    let mut pointers = BTreeSet::new();
    let mut constants = BTreeMap::new();
    let mut chains = BTreeMap::new();
    let mut workgroup = BTreeSet::new();
    for (_, op) in &code {
        match op[0] & 0xffff {
            5 => {
                names.insert(op[1], string(&op[2..])?);
            }
            21 if op.len() == 4 && op[2] == 32 && op[3] <= 1 => {
                integers.insert(op[1]);
                if op[3] == 0 {
                    unsigned.insert(op[1]);
                }
            }
            32 if op[2] == 4 && unsigned.contains(&op[3]) => {
                pointers.insert(op[1]);
            }
            43 if op.len() == 4 && integers.contains(&op[1]) => {
                constants.insert(op[2], (op[1], op[3]));
            }
            59 if (op.len() == 4 || op.len() == 5) && op[3] == 4 => {
                workgroup.insert(op[2]);
            }
            65 if op.len() >= 5 && pointers.contains(&op[1]) => {
                chains.insert(op[2], op[3]);
            }
            _ => {}
        }
    }
    let named = |name: &str| -> Result<u32> {
        let ids: Vec<_> = names
            .iter()
            .filter(|(_, v)| v.as_str() == name)
            .map(|(&id, _)| id)
            .collect();
        if ids.len() != 1 {
            return Err(invalid(format!("volatile helper identity {name}")));
        }
        Ok(ids[0])
    };
    let helper = named("retained_round")?;
    let scratch = named("rate_scratch")?;
    if !workgroup.contains(&scratch) {
        return Err(invalid("volatile scratch must be workgroup storage"));
    }
    let valid_pointer = |id: u32| chains.get(&id) == Some(&scratch);
    let constant = |id: u32, value: u32| constants.get(&id).is_some_and(|&(_, v)| v == value);
    let mut result = words[..5].to_vec();
    let mut function = 0;
    let mut loads = 0;
    let mut stores = 0;
    for (_, op) in &code {
        match op[0] & 0xffff {
            54 => function = op[2],
            56 => function = 0,
            _ => {}
        }
        if function == helper {
            match op[0] & 0xffff {
                228 => {
                    if op.len() != 5
                        || !valid_pointer(op[1])
                        || !constant(op[2], 2)
                        || !constant(op[3], 0)
                    {
                        return Err(invalid("volatile store contract"));
                    }
                    result.extend_from_slice(&[(4 << 16) | 62, op[1], op[4], 1]);
                    stores += 1;
                    continue;
                }
                234 => {
                    if op.len() != 7
                        || !unsigned.contains(&op[1])
                        || !valid_pointer(op[3])
                        || !constant(op[4], 2)
                        || !constant(op[5], 0)
                        || constants.get(&op[6]) != Some(&(op[1], 0))
                    {
                        return Err(invalid("volatile load contract"));
                    }
                    result.extend_from_slice(&[(5 << 16) | 61, op[1], op[2], op[3], 1]);
                    loads += 1;
                    continue;
                }
                227..=242 => return Err(invalid("unexpected retained-round atomic")),
                _ => {}
            }
        }
        result.extend_from_slice(op);
    }
    if (loads, stores) != (1, 1) {
        return Err(invalid("volatile helper must contain one load/store pair"));
    }
    Ok(result)
}

#[test]
fn newton_volatile_compiler_contract() {
    use crate::soil_newton_gpu::{
        ExperimentalSoilNewtonGpu as Solver, NewtonVariant, shader_source_for,
    };
    let source = Solver::cached_contact_source(Solver::cached_delta_source(
        Solver::stalled_release_source(shader_source_for(NewtonVariant::Fischer)),
    ));
    let source = crate::soil_newton_parallel_release::core_source(
        crate::soil_newton_parallel_release::source(source),
    );
    let source = format!("{}256\n{source}", crate::soil_newton_width::MARKER);
    let control = compile(&source, "newton").unwrap();
    let candidate = compile(&format!("{VOLATILE_MARKER}{source}"), "newton").unwrap();
    assert_eq!(candidate, transform(&control).unwrap());
    if let Some(root) = std::env::var_os("MOLLA_NEWTON_VOLATILE_ARTIFACTS") {
        let root = std::path::PathBuf::from(root);
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("source.wgsl"), &source).unwrap();
        for (name, words) in [("atomic", &control), ("volatile", &candidate)] {
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
    assert!(transform(&candidate).is_err());
    assert!(
        compile(
            &format!("{VOLATILE_MARKER}{VOLATILE_MARKER}{source}"),
            "newton"
        )
        .is_err()
    );
    let mut helper = 0;
    let mut function = 0;
    let mut accesses = Vec::new();
    for (_, op) in instructions(&candidate).unwrap() {
        match op[0] & 0xffff {
            5 if string(&op[2..]).unwrap() == "retained_round" => helper = op[1],
            54 => function = op[2],
            56 => function = 0,
            61 if function == helper && op.len() == 5 && op[4] == 1 => {
                accesses.push(61);
            }
            62 if function == helper && op.len() == 4 && op[3] == 1 => {
                accesses.push(op[0] & 0xffff);
            }
            227..=242 => assert_ne!(function, helper),
            _ => {}
        }
    }
    assert_eq!(accesses, [62, 61]);
    let atomic = instructions(&control)
        .unwrap()
        .into_iter()
        .find(|(_, op)| op[0] & 0xffff == 234)
        .unwrap()
        .0;
    let mut bad = control.clone();
    bad[atomic + 6] = u32::MAX;
    assert!(transform(&bad).is_err());
}
