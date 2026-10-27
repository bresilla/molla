use super::*;

fn source(compensated: bool) -> String {
    shared_chunks::source(compensated).unwrap()
}

pub(super) fn gather(sim: &MpmSimulation, candidate: bool) -> Gather {
    if !candidate {
        return if sim.native_float_atomics {
            Gather::new_masked(sim).unwrap()
        } else {
            Gather::new_compensated(sim).unwrap()
        };
    }
    Gather::new_shared(sim, !sim.native_float_atomics).unwrap()
}

#[test]
fn gather_chunk_source_contract() {
    for (native, compensated) in [(false, false), (false, true), (true, false)] {
        let base = if native {
            native_atomics::shader_source()
        } else {
            include_str!("../shaders/mpm.wgsl").to_owned()
        };
        let chunk = source(compensated);
        for marker in [
            "let bit = firstTrailingBit(mask);",
            "mask &= mask - 1u;",
            "mass_sum += w * mass;",
            "momentum_sum += mom;",
        ] {
            assert_eq!(chunk.matches(marker).count(), 1);
        }
        let text = format!(
            "{base}\n{}\n{chunk}",
            include_str!("../shaders/mpm_deterministic_p2g.wgsl")
        );
        let module = naga::front::wgsl::parse_str(&text).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
        let record = module
            .types
            .iter()
            .find(|(_, ty)| ty.name.as_deref() == Some("GatherChunk"))
            .unwrap()
            .1;
        let naga::TypeInner::Struct { span, .. } = &record.inner else {
            panic!("chunk type");
        };
        assert_eq!(*span, 208);
    }
}

#[test]
#[ignore = "requires GPU; shared particle chunk exact grid and evolution"]
fn gather_chunk_exact_grid_and_evolution() {
    pruning_tests::exact_grid_shared_chunk();
}

#[test]
#[ignore = "requires GPU and saved50k native fixture; exact shared chunk paired cost"]
fn gather_chunk_saved_50k_pairs() {
    pruning_tests::saved_50k_shared_chunk();
}
