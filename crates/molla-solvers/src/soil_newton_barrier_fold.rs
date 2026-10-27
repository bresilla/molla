use super::*;

pub(crate) const MARKER: &str = "// NEWTON_BARRIER_FOLD_TEST\n";

pub(crate) fn transform(words: &[u32]) -> Result<(Vec<u32>, usize)> {
    let code = instructions(words)?;
    let mut pointer_types = BTreeMap::new();
    let mut pointers = BTreeMap::new();
    for (_, op) in &code {
        match op[0] & 0xffff {
            32 if op.len() == 4 => {
                pointer_types.insert(op[1], op[2]);
            }
            59 | 65 | 66 | 67 | 70 | 83 if op.len() >= 3 => {
                if let Some(&class) = pointer_types.get(&op[1]) {
                    pointers.insert(op[2], class);
                }
            }
            _ => {}
        }
    }
    let local = |id: u32| {
        pointers
            .get(&id)
            .is_some_and(|class| matches!(class, 6 | 7))
    };
    let immutable = |id: u32| {
        pointers
            .get(&id)
            .is_some_and(|class| matches!(class, 0 | 1 | 2 | 9))
    };
    let mut result = words[..5].to_vec();
    let mut previous: Option<&[u32]> = None;
    let mut removed = 0;
    for (_, op) in code {
        let opcode = op[0] & 0xffff;
        if opcode == 224 && op.len() == 4 {
            if previous == Some(op) {
                removed += 1;
                continue;
            }
            previous = Some(op);
        } else {
            let harmless = match opcode {
                0 | 8 | 317 | 65 | 66 | 67 | 70 | 79..=84 | 109..=200 => true,
                61 if op.len() == 4 => local(op[3]) || immutable(op[3]),
                62 if op.len() == 3 => local(op[1]),
                _ => false,
            };
            if !harmless {
                previous = None;
            }
        }
        result.extend_from_slice(op);
    }
    Ok((result, removed))
}

fn words(ops: &[&[u32]]) -> Vec<u32> {
    let mut words = vec![0x07230203, 0x00010300, 0, 100, 0];
    for op in ops {
        words.push(((op.len() as u32) << 16) | op[0]);
        words.extend_from_slice(&op[1..]);
    }
    words
}

#[test]
fn newton_barrier_fold_memory_and_control_hazards() {
    let barrier = [224, 20, 20, 21];
    let prefix: &[&[u32]] = &[
        &[32, 1, 7, 30],
        &[32, 2, 4, 30],
        &[32, 3, 12, 30],
        &[32, 4, 2, 30],
        &[59, 1, 10, 7],
        &[59, 2, 11, 4],
        &[59, 3, 12, 12],
        &[59, 4, 13, 2],
    ];
    for (middle, expected) in [
        (vec![vec![0]], 1),
        (vec![vec![61, 30, 31, 10], vec![62, 10, 31]], 1),
        (vec![vec![61, 30, 31, 13], vec![129, 30, 32, 31, 31]], 1),
        (vec![vec![61, 30, 31, 11]], 0),
        (vec![vec![61, 30, 31, 12]], 0),
        (vec![vec![62, 11, 31]], 0),
        (vec![vec![62, 12, 31]], 0),
        (vec![vec![61, 30, 31, 10, 1]], 0),
        (vec![vec![57, 30, 31, 40]], 0),
        (vec![vec![248, 40]], 0),
        (vec![vec![249, 40]], 0),
        (vec![vec![225, 20, 21]], 0),
        (vec![vec![234, 30, 31, 11, 20, 21, 30]], 0),
        (vec![vec![65535]], 0),
    ] {
        let mut ops = prefix.to_vec();
        ops.push(&barrier);
        ops.extend(middle.iter().map(Vec::as_slice));
        ops.push(&barrier);
        let original = words(&ops);
        let (folded, removed) = transform(&original).unwrap();
        assert_eq!(removed, expected, "{middle:?}");
        assert_eq!(original.len() - folded.len(), expected * 4);
        assert_eq!(transform(&folded).unwrap(), (folded, 0));
    }
    assert_eq!(
        transform(&words(&[&barrier, &[224, 20, 20, 22]]))
            .unwrap()
            .1,
        0
    );
    assert!(transform(&[0; 5]).is_err());
}

#[test]
#[cfg(feature = "soa-scratch-newton")]
fn newton_barrier_fold_production_binary_contract() {
    use crate::soil_newton_gpu::{NewtonVariant, shader_source_for, soa_scratch};
    let source = soa_scratch::program(shader_source_for(NewtonVariant::Fischer), true);
    let original = compile(&source, "newton").unwrap();
    let (expected, removed) = transform(&original).unwrap();
    let candidate = compile(&format!("{source}\n{MARKER}"), "newton").unwrap();
    assert_eq!(candidate, expected);
    assert!(removed > 0, "no eligible production barriers");
    let strip = |binary: &[u32]| {
        instructions(binary)
            .unwrap()
            .into_iter()
            .filter(|(_, op)| op[0] & 0xffff != 224)
            .flat_map(|(_, op)| op.iter().copied())
            .collect::<Vec<_>>()
    };
    assert_eq!(strip(&original), strip(&candidate));
    let total = instructions(&original)
        .unwrap()
        .iter()
        .filter(|(_, op)| op[0] & 0xffff == 224)
        .count();
    if let Some(root) = std::env::var_os("MOLLA_BARRIER_FOLD_BINARY_OUTPUT") {
        let root = std::path::PathBuf::from(root);
        std::fs::create_dir(&root).unwrap();
        for (name, words) in [("original", &original), ("candidate", &candidate)] {
            let bytes: Vec<_> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
            std::fs::write(root.join(format!("{name}.spv")), bytes).unwrap();
        }
        let names: BTreeMap<_, _> = instructions(&original)
            .unwrap()
            .iter()
            .filter(|(_, op)| op[0] & 0xffff == 5)
            .map(|(_, op)| (op[1], string(&op[2..]).unwrap()))
            .collect();
        let counts = |binary: &[u32]| {
            let mut function = 0;
            let mut counts = BTreeMap::<u32, usize>::new();
            for (_, op) in instructions(binary).unwrap() {
                if op[0] & 0xffff == 54 {
                    function = op[2];
                }
                if op[0] & 0xffff == 224 {
                    *counts.entry(function).or_default() += 1;
                }
            }
            counts
        };
        let after = counts(&candidate);
        let sites: Vec<_> = counts(&original)
            .iter()
            .filter_map(|(id, before)| {
                let after = after.get(id).copied().unwrap_or(0);
                (*before != after).then(
                    || serde_json::json!({"function":names.get(id),"before":before,"after":after}),
                )
            })
            .collect();
        std::fs::write(root.join("audit.json"), serde_json::to_vec_pretty(&serde_json::json!({
            "removed":removed,"original_barriers":total,"all_nonbarrier_words_exact":true,"sites":sites
        })).unwrap()).unwrap();
    }
    println!(
        "barrier-fold removed={removed} original_barriers={total} all_nonbarrier_words_exact=true"
    );
}
