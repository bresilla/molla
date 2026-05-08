//! Writes simulated state back onto the USD stage as time samples.
//!
//! For each imported body, write `xformOp:translate` and
//! `xformOp:orient` time samples at the requested time. Reuses the
//! prim's existing `xformOpOrder` if authored; otherwise authors a
//! standard `["xformOp:translate", "xformOp:orient"]` order.
//!
//! Joint coords are not yet sampled — that's a follow-up once the
//! consumer has a use case for it.

use anyhow::Result;
use molla::sim::State;
use openusd::Stage;

use crate::ImportInfo;

/// Sample every imported body's world pose at `time` (seconds) and
/// store as time-sampled `xformOp:translate` + `xformOp:orient` on
/// the prim.
///
/// The current implementation is a stub — full Sdf-level field
/// writing requires the bevy_openusd helpers (`set_pose`,
/// `set_xform_op_order`) which aren't in molla's path-deps yet.
/// Marker / tests track the intent; landing this is part of the
/// USD round-trip task (#41) once the helper layer is wired up.
pub fn write_body_poses(
    _stage: &mut Stage,
    info: &ImportInfo,
    state: &State,
    _time: f64,
) -> Result<()> {
    let body_q = state
        .body_q
        .host()
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    if body_q.len() != info.body_paths.len() {
        return Err(anyhow::anyhow!(
            "write_back: body_q length {} doesn't match info.body_paths length {}",
            body_q.len(),
            info.body_paths.len()
        ));
    }
    log::debug!(
        "usd_molla::write_back: would write {} body poses (stub)",
        body_q.len()
    );
    Ok(())
}
