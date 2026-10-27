//! Exact-input memoization of deterministic transfer constitutive records.

use super::*;

pub(super) const KEY_BYTES: u64 = 96;

pub(super) fn source(input: String) -> Result<String> {
    let entry = "@compute @workgroup_size(64)\nfn gather_prepare(@builtin(global_invocation_id) gid: vec3<u32>)";
    if input.matches(entry).count() != 1 {
        return Err(Error::Solver(
            "Gather stress-cache source contract changed".into(),
        ));
    }
    let mut source = input.replace(entry, "fn gather_prepare_uncached(gid: vec3<u32>)");
    source.push_str(include_str!("../shaders/mpm_gather_stress_cache.wgsl"));
    Ok(source)
}

#[cfg(test)]
#[path = "mpm_gather_stress_cache_tests.rs"]
mod tests;

#[cfg(all(test, feature = "resource-counters"))]
#[path = "mpm_gather_stress_cache_profile_tests.rs"]
pub(super) mod profile_tests;
