use super::*;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MpmSoilReactionInfo {
    pub compensated: bool,
    /// Logical reduction storage, separate from the coupling and motion-guard budgets.
    pub retained_bytes: u64,
}

fn admit(
    enabled: bool,
    native: bool,
    projected: bool,
    retained: u64,
    max_bytes: u64,
) -> Result<()> {
    if retained > max_bytes {
        return Err(invalid(
            "reaction reduction retained storage exceeds budget",
        ));
    }
    if enabled && (native || projected) {
        return Err(invalid(
            "compensated reactions require portable direct coupling",
        ));
    }
    Ok(())
}

impl RigidWorld {
    /// Attached reaction-reduction selection and retained bytes, without readback.
    pub fn mpm_soil_reaction_info(&self) -> Option<MpmSoilReactionInfo> {
        let soil = &self.gpu.as_ref()?.soil.as_ref()?.soil;
        Some(MpmSoilReactionInfo {
            compensated: soil.compensated_rigid_wrench_enabled(),
            retained_bytes: soil.compensated_rigid_wrench_storage_bytes(),
        })
    }

    /// Selects compensated reactions for portable direct coupling; this is cold preparation.
    /// Returns retained bytes: 32 * max(max_rigid, 1), separately budgeted from coupling.
    /// Disabling retains storage. Native atomics and projected coupling reject enabling.
    pub fn set_mpm_soil_compensated_reactions(
        &mut self,
        enabled: bool,
        max_bytes: u64,
    ) -> Result<u64> {
        let gpu = self
            .gpu
            .as_mut()
            .ok_or_else(|| invalid("GPU world required"))?;
        gpu.engine.check_device()?;
        let soil = gpu
            .soil
            .as_mut()
            .ok_or_else(|| invalid("no soil attached"))?;
        soil.validate(&self.scene)?;
        admit(
            enabled,
            soil.soil.uses_native_float_atomics(),
            soil.projection_config.is_some(),
            soil.soil.compensated_rigid_wrench_storage_bytes(),
            max_bytes,
        )?;
        if enabled {
            soil.soil.prepare_compensated_rigid_wrench(max_bytes)?;
        }
        if soil.soil.compensated_rigid_wrench_enabled() != enabled {
            soil.soil.set_compensated_rigid_wrench(enabled)?;
        }
        Ok(soil.soil.compensated_rigid_wrench_storage_bytes())
    }
}

#[test]
fn coupled_reaction_selection_admission() {
    for native in [false, true] {
        for projected in [false, true] {
            for retained in [0, 32, 64] {
                for budget in [0, 31, 32, 63, 64, u64::MAX] {
                    for enabled in [false, true] {
                        assert_eq!(
                            admit(enabled, native, projected, retained, budget).is_ok(),
                            retained <= budget && (!enabled || (!native && !projected))
                        );
                    }
                }
            }
        }
    }
}
