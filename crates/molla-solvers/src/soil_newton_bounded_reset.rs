use super::*;

const FULL_H: &str = "for (var i=lane;i<4160u;i+=64u) {h[i]=0.0;}";
const FULL_G: &str = "for (var i=lane;i<65u;i+=64u) {g[i]=0.0;}";
const BOUNDED_H: &str = "for (var i=lane;i<(cycle_limit+1u)*64u;i+=64u) {h[i]=0.0;}";
const BOUNDED_G: &str = "for (var i=lane;i<cycle_limit+1u;i+=64u) {g[i]=0.0;}";

fn users<'a>(source: &'a str, array: &str) -> Vec<&'a str> {
    source
        .split("\nfn ")
        .skip(1)
        .filter_map(|function| {
            let name = function.split('(').next().unwrap();
            let body = function.split("\n}").next().unwrap();
            body.match_indices(&format!("{array}["))
                .any(|(at, _)| {
                    at == 0
                        || !body.as_bytes()[at - 1].is_ascii_alphanumeric()
                            && body.as_bytes()[at - 1] != b'_'
                })
                .then_some(name)
        })
        .collect()
}

pub(crate) fn source(mut source: String) -> String {
    assert!(source.starts_with(crate::soil_newton_protected::PREFIX));
    assert_eq!(
        users(&source, "h"),
        [
            "recover_direction",
            "krylov_cycle",
            "publication_certify",
            "certify_machine"
        ]
    );
    assert_eq!(users(&source, "g"), ["recover_direction", "krylov_cycle"]);
    for region in [
        include_str!("../shaders/soil_newton_publication_products.wgsl"),
        include_str!("../shaders/soil_newton_publication_reactions.wgsl"),
    ] {
        assert_eq!(source.matches(region).count(), 1);
    }
    let start = source
        .find("fn krylov_cycle(id:u32,cycle_limit:u32)")
        .unwrap();
    let end = start + source[start..].find("\nfn least_squares(").unwrap();
    let mut cycle = source[start..end].to_owned();
    replace(&mut cycle, FULL_H, BOUNDED_H);
    replace(&mut cycle, FULL_G, BOUNDED_G);
    source.replace_range(start..end, &cycle);
    source
}

#[test]
fn newton_bounded_reset_source_contract() {
    let control = soa_scratch::program(shader_source_for(NewtonVariant::Fischer), true);
    let candidate = source(control.clone());
    let mut restored = candidate.clone();
    replace(&mut restored, BOUNDED_H, FULL_H);
    replace(&mut restored, BOUNDED_G, FULL_G);
    assert_eq!(restored, control);
    for marker in [
        "var<storage",
        "var<workgroup>",
        "@binding(",
        "rate_add(",
        "rate_mul(",
        "workgroupBarrier();",
        "storageBarrier();",
    ] {
        assert_eq!(
            candidate.matches(marker).count(),
            control.matches(marker).count()
        );
    }
    let binary = crate::soil_newton_protected::compile(&candidate, "newton").unwrap();
    let specialized = crate::soil_newton_width::specialize(candidate.clone()).unwrap();
    assert!(specialized.contains(&BOUNDED_H.replace("i+=64u", "i+=256u")));
    assert!(specialized.contains(&BOUNDED_G.replace("i+=64u", "i+=256u")));
    if let Some(root) = std::env::var_os("MOLLA_NEWTON_RESET_ARTIFACTS") {
        let root = std::path::PathBuf::from(root);
        std::fs::create_dir_all(&root).unwrap();
        for (name, source) in [
            ("control.wgsl", control),
            ("candidate.wgsl", candidate),
            ("specialized.wgsl", specialized),
        ] {
            std::fs::write(root.join(name), source).unwrap();
        }
        std::fs::write(
            root.join("candidate.spv"),
            binary
                .iter()
                .flat_map(|word| word.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    }
}

#[test]
fn newton_bounded_reset_covers_krylov_reads() {
    for limit in 1usize..=64 {
        let mut old = vec![f32::NAN; 4160];
        let mut bounded = old.clone();
        old.fill(0.);
        bounded[..(limit + 1) * 64].fill(0.);
        let mut rhs = [false; 65];
        rhs[..=limit].fill(true);
        let mut rotations = [false; 64];
        for j in 0..limit {
            for i in 0..=j {
                let at = i * 64 + j;
                assert_eq!(bounded[at].to_bits(), old[at].to_bits());
                old[at] += (i + j + 1) as f32;
                bounded[at] += (i + j + 1) as f32;
            }
            old[(j + 1) * 64 + j] = 1.;
            bounded[(j + 1) * 64 + j] = 1.;
            for i in 0..j {
                assert!(rotations[i]);
                for at in [i * 64 + j, (i + 1) * 64 + j] {
                    assert_eq!(bounded[at].to_bits(), old[at].to_bits());
                }
            }
            rotations[j] = true;
            assert!(rhs[j]);
            rhs[j + 1] = true;
            let mut coefficients = [false; 64];
            for i in (0..=j).rev() {
                assert!(rhs[i]);
                for k in i + 1..=j {
                    assert!(coefficients[k]);
                }
                coefficients[i] = true;
            }
            for i in 0..=j {
                for k in i..=j {
                    assert_eq!(bounded[i * 64 + k].to_bits(), old[i * 64 + k].to_bits());
                }
            }
        }
        assert!((limit + 1) * 64 <= 4160);
        assert!(limit + 1 <= 65);
    }
    assert_eq!((8 + 1) * 64 + 8 + 1, 585);
    assert_eq!(4160 + 65, 4225);
}

#[test]
fn newton_bounded_reset_publication_overwrites_scratch() {
    for width in 1usize..=32 {
        for count in 0usize..=16 {
            for axes in [1usize, 3] {
                let mut written = [false; 4160];
                for lane in 0..256 {
                    for item in (lane..count * width * axes).step_by(256) {
                        for part in 0..2 {
                            let at = 2 * item + part;
                            assert!(!written[at]);
                            written[at] = true;
                        }
                    }
                }
                for lane in 0..width {
                    for local in 0..count {
                        for axis in 0..axes {
                            let item = (local * width + lane) * axes + axis;
                            assert!(written[2 * item] && written[2 * item + 1]);
                        }
                    }
                }
            }
        }
    }
}
