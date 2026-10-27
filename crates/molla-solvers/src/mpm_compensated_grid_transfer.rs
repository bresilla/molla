use super::*;

pub(super) fn source() -> Result<String> {
    let mut source = include_str!("../shaders/mpm.wgsl").to_owned();
    source.push_str(include_str!(
        "../shaders/mpm_compensated_grid_transfer.wgsl"
    ));
    for (name, array, component) in [
        ("mass", "mass", 0),
        ("mx", "mx", 1),
        ("my", "my", 2),
        ("mz", "mz", 3),
        ("kdiag", "kdiag", 4),
    ] {
        replace_function(
            &mut source,
            &format!("wg_atomic_add_{name}"),
            &format!(
                "fn wg_atomic_add_{name}(slot:u32,val:f32) {{\n var old_bits=atomicLoad(&wg_{array}[slot]);\n loop {{\n let old=bitcast<f32>(old_bits);let next=transfer_round(old+val);\n let result=atomicCompareExchangeWeak(&wg_{array}[slot],old_bits,bitcast<u32>(next));\n if result.exchanged {{transfer_local_tail(5u*slot+{component}u,transfer_error(old,val,next));break;}}\n old_bits=result.old_value;\n }}\n}}"
            ),
        )?;
    }
    for (name, array, address) in [
        ("mass", "mass", "5u*idx"),
        ("mom", "momentum", "5u*(idx/3u)+1u+idx%3u"),
        ("kdiag", "kdiag", "5u*idx+4u"),
    ] {
        replace_function(
            &mut source,
            &format!("atomic_add_{name}"),
            &format!(
                "fn atomic_add_{name}(idx:u32,val:f32) {{\n var old_bits=atomicLoad(&g_{array}[idx]);\n loop {{\n let old=bitcast<f32>(old_bits);let next=transfer_round(old+val);\n let result=atomicCompareExchangeWeak(&g_{array}[idx],old_bits,bitcast<u32>(next));\n if result.exchanged {{transfer_global_tail({address},transfer_error(old,val,next));break;}}\n old_bits=result.old_value;\n }}\n}}"
            ),
        )?;
    }
    replace(
        &mut source,
        "fn p2g(@builtin(global_invocation_id) gid: vec3<u32>) {",
        "fn p2g(@builtin(global_invocation_id) gid: vec3<u32>) {\n    transfer_lane=gid.x%64u;",
    )?;
    replace(
        &mut source,
        "    atomicStore(&wg_kdiag[lid.x], 0u);",
        "    atomicStore(&wg_kdiag[lid.x], 0u);\n    transfer_lane=lid.x;\n    for(var field=0u;field<5u;field++) {atomicStore(&transfer_local_residual[5u*lid.x+field],0u);}",
    )?;
    replace(
        &mut source,
        "    atomic_add_kdiag(cidx, bitcast<f32>(atomicLoad(&wg_kdiag[slot])));",
        "    atomic_add_kdiag(cidx, bitcast<f32>(atomicLoad(&wg_kdiag[slot])));\n    for(var field=0u;field<5u;field++) {transfer_global_tail(5u*cidx+field,bitcast<f32>(atomicLoad(&transfer_local_residual[5u*slot+field])));}",
    )?;
    Ok(source)
}

#[cfg(test)]
#[path = "mpm_compensated_grid_transfer_tests.rs"]
mod tests;
