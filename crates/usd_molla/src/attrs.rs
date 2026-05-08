//! Small `openusd::sdf::Value` → native-Rust attribute helpers.
//!
//! Mirrors the helpers `urdf2usd::to_urdf::read` keeps locally.
//! Lives inside `usd_molla` so the bridge crate doesn't have to
//! depend on `urdf2usd`.

use anyhow::Result;
use openusd::sdf::{FieldKey, Path, Value};
use openusd::Stage;

pub fn attr_value(stage: &Stage, prim: &Path, attr: &str) -> Result<Option<Value>> {
    let attr_path = prim
        .append_property(attr)
        .map_err(|e| anyhow::anyhow!("property path: {e}"))?;
    stage
        .field(attr_path, FieldKey::Default)
        .map_err(|e| anyhow::anyhow!("USD field read: {e}"))
}

pub fn attr_f64(stage: &Stage, prim: &Path, attr: &str) -> Option<f64> {
    attr_value(stage, prim, attr).ok().flatten().and_then(|v| match v {
        Value::Double(d) => Some(d),
        Value::Float(f) => Some(f as f64),
        _ => None,
    })
}

pub fn attr_vec3_array(
    stage: &Stage,
    prim: &Path,
    attr: &str,
) -> Option<Vec<[f64; 3]>> {
    attr_value(stage, prim, attr).ok().flatten().and_then(|v| match v {
        Value::Vec3fVec(arr) => Some(
            arr.into_iter()
                .map(|p| [p[0] as f64, p[1] as f64, p[2] as f64])
                .collect(),
        ),
        Value::Vec3dVec(arr) => Some(arr),
        _ => None,
    })
}

pub fn attr_int_array(stage: &Stage, prim: &Path, attr: &str) -> Option<Vec<i32>> {
    attr_value(stage, prim, attr).ok().flatten().and_then(|v| match v {
        Value::IntVec(arr) => Some(arr),
        _ => None,
    })
}

/// Read the prim's `typeName` field, returning the type token (e.g.
/// `"Sphere"`, `"Cube"`, `"PhysicsRevoluteJoint"`).
pub fn type_name(stage: &Stage, prim: &Path) -> Option<String> {
    stage
        .field(prim.clone(), FieldKey::TypeName)
        .ok()
        .flatten()
        .and_then(|v: Value| match v {
            Value::Token(t) => Some(t),
            Value::String(s) => Some(s),
            _ => None,
        })
}
