use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Certificate {
    Unspecified,
    GapIndependentVelocity,
}

impl Certificate {
    #[cfg(test)]
    pub(crate) const CURRENT: Self = Self::GapIndependentVelocity;

    pub(crate) fn from_packet(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 64
            || bytes[..4] != 1u32.to_le_bytes()
            || bytes[44..64].iter().any(|&b| b != 0)
        {
            return Err(invalid());
        }
        match u32::from_le_bytes(bytes[40..44].try_into().unwrap()) {
            0 => Ok(Self::Unspecified),
            1 => Ok(Self::GapIndependentVelocity),
            _ => Err(Error::Solver("unknown Newton certificate version".into())),
        }
    }

    pub(crate) fn tag(self) -> u32 {
        match self {
            Self::Unspecified => 0,
            Self::GapIndependentVelocity => 1,
        }
    }

    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Unspecified => "unspecified",
            Self::GapIndependentVelocity => "gap-independent-velocity-v1",
        }
    }
}

#[cfg(test)]
pub(crate) fn validate_replay(
    bytes: &[u8],
    meta: &serde_json::Value,
    exact: bool,
) -> Result<Certificate> {
    let certificate = Certificate::from_packet(bytes)?;
    let word = |i: usize| u32::from_le_bytes(bytes[4 * i..4 * i + 4].try_into().unwrap()) as usize;
    let n = word(1);
    let c = word(2);
    if !(1..=32).contains(&n)
        || c > 16384
        || !(1..=64).contains(&word(5))
        || !(1..=64).contains(&word(6))
        || bytes.len() != 128 + 4 * n * n + 8 * n + (32 * n + 128) * c.max(1)
    {
        return Err(Error::Solver("invalid Newton replay packet shape".into()));
    }
    let declared = meta.get("certificate_version").is_some() || meta.get("certificate").is_some();
    if (declared || certificate != Certificate::Unspecified)
        && (meta["certificate_version"].as_u64() != Some(u64::from(certificate.tag()))
            || meta["certificate"].as_str() != Some(certificate.name()))
    {
        return Err(Error::Solver(
            "Newton certificate metadata does not match packet".into(),
        ));
    }
    if exact && certificate != Certificate::CURRENT {
        return Err(Error::Solver("exact Newton report replay requires capture-time gap-independent-velocity-v1 certificate provenance; legacy inputs remain usable without exact-report comparison".into()));
    }
    Ok(certificate)
}

#[cfg(test)]
pub(crate) fn exact_requested() -> bool {
    [
        "MOLLA_NEWTON_SUCCESS_EXACT",
        "MOLLA_NEWTON_SEED_COST",
        "MOLLA_NEWTON_FIRST_RELEASE_COMPARE",
    ]
    .iter()
    .any(|key| std::env::var_os(key).is_some())
}

#[cfg(test)]
pub(crate) fn preflight(root: &std::path::Path, exact: bool) -> Result<Certificate> {
    use std::io::Read;
    let read = |name: &str, limit: u64| -> Result<Vec<u8>> {
        let file =
            std::fs::File::open(root.join(name)).map_err(|e| Error::Solver(e.to_string()))?;
        let mut bytes = Vec::new();
        file.take(limit + 1)
            .read_to_end(&mut bytes)
            .map_err(|e| Error::Solver(e.to_string()))?;
        if bytes.len() as u64 > limit {
            return Err(Error::Solver("Newton replay input exceeds bound".into()));
        }
        Ok(bytes)
    };
    let bytes = read(
        "newton.bin",
        128 + 4 * 32 * 32 + 8 * 32 + (32 * 32 + 128) * 16384,
    )?;
    let meta: serde_json::Value = serde_json::from_slice(&read("capture.json", 65536)?)
        .map_err(|e| Error::Solver(e.to_string()))?;
    if meta["blake3"].as_str() != Some(blake3::hash(&bytes).to_hex().as_str()) {
        return Err(Error::Solver("Newton replay packet digest mismatch".into()));
    }
    validate_replay(&bytes, &meta, exact)
}

#[cfg(test)]
#[path = "soil_newton_certificate_tests.rs"]
mod tests;
