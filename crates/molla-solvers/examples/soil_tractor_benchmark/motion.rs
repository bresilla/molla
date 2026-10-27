use super::*;
use molla_solvers::rigid_world::MpmSoilMotionInfo;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Bound {
    pub metres: f64,
    pub max_bytes: u64,
}

fn required_bytes() -> u64 {
    16 + 128
        * u64::from(
            SoilTractorConfig::default()
                .particle_count()
                .unwrap()
                .max(1),
        )
}

pub(super) fn parse_fusion(value: &str) -> Result<bool, String> {
    match value {
        "0" => Ok(false),
        "1" => Ok(true),
        _ => Err("--soil-motion-fusion must be 0 or 1".into()),
    }
}

pub(super) fn verified_fusion(requested: bool, live: Option<bool>) -> Result<Value, String> {
    if live != Some(requested) {
        return Err("live motion fusion does not match selection".into());
    }
    Ok(json!(requested))
}

#[test]
fn motion_fusion_selection_requires_guard_and_exact_live_mode() {
    let options = |extra: &[&str]| {
        parse_options(
            ["--output", "unused"]
                .into_iter()
                .chain(extra.iter().copied())
                .map(str::to_owned),
        )
    };
    assert!(!options(&[]).unwrap().motion_fusion);
    assert!(
        !options(&["--soil-motion-fusion", "0"])
            .unwrap()
            .motion_fusion
    );
    assert!(options(&["--soil-motion-fusion", "1"]).is_err());
    assert!(options(&["--soil-motion-fusion", "1", "--soil-motion-bound", "off"]).is_err());
    assert!(
        options(&[
            "--soil-motion-fusion",
            "1",
            "--soil-motion-bound",
            "1:6400016"
        ])
        .unwrap()
        .motion_fusion
    );
    assert!(options(&["--soil-motion-fusion", "0", "--soil-motion-fusion", "0"]).is_err());
    for bad in ["", "true", "on", "2", "01", " 1", "1 "] {
        assert!(parse_fusion(bad).is_err());
    }
    for selected in [false, true] {
        assert_eq!(
            verified_fusion(selected, Some(selected)).unwrap(),
            json!(selected)
        );
        assert!(verified_fusion(selected, Some(!selected)).is_err());
        assert!(verified_fusion(selected, None).is_err());
    }
}

pub(super) fn parse(value: &str) -> Result<Option<Bound>, String> {
    if value == "off" {
        return Ok(None);
    }
    let (metres, bytes) = value
        .split_once(':')
        .ok_or("--soil-motion-bound must be off or METRES:BYTES")?;
    if metres.trim() != metres
        || bytes.is_empty()
        || bytes.starts_with('0')
        || !bytes.bytes().all(|b| b.is_ascii_digit())
    {
        return Err("invalid motion-bound value or canonical byte budget".into());
    }
    let metres = metres.parse::<f64>().map_err(|e| e.to_string())?;
    let max_bytes = bytes.parse::<u64>().map_err(|e| e.to_string())?;
    if !metres.is_finite() || metres < 0.0 || max_bytes < required_bytes() {
        return Err(
            "motion bound must be finite/nonnegative with sufficient proposal storage".into(),
        );
    }
    Ok(Some(Bound { metres, max_bytes }))
}

pub(super) fn requested(bound: Option<Bound>) -> Value {
    match bound {
        None => json!({"enabled":false}),
        Some(b) => {
            let axes = [b.metres; 3];
            json!({"enabled":true,"metres_per_axis_per_substep":axes,"max_bytes":b.max_bytes})
        }
    }
}

pub(super) fn verified_live(
    bound: Option<Bound>,
    live: MpmSoilMotionInfo,
) -> Result<Value, String> {
    let expected = bound.map(|bound| {
        let mut rounded = bound.metres.min(f64::from(f32::MAX)) as f32;
        if f64::from(rounded) > bound.metres {
            rounded = rounded.next_down();
        }
        if rounded < f32::MIN_POSITIVE {
            rounded = 0.0;
        }
        [rounded; 3]
    });
    if live.bounds != expected || live.retained_bytes != bound.map_or(0, |_| required_bytes()) {
        return Err("live motion guard does not match requested bounds/storage".into());
    }
    Ok(
        json!({"bounds_metres_per_axis_per_substep":live.bounds,"retained_bytes":live.retained_bytes}),
    )
}

#[test]
fn motion_guard_options_and_live_manifest_are_checked() {
    let options = |extra: &[&str]| {
        parse_options(
            ["--output", "unused"]
                .into_iter()
                .chain(extra.iter().copied())
                .map(str::to_owned),
        )
    };
    assert_eq!(options(&[]).unwrap().motion_bound, None);
    assert!(options(&["--soil-motion-bound"]).is_err());
    assert!(options(&["--soil-motion-bound", "off", "--soil-motion-bound", "off"]).is_err());
    assert_eq!(
        options(&["--soil-motion-bound", "1:6400016"])
            .unwrap()
            .motion_bound
            .unwrap()
            .metres,
        1.0
    );
    assert_eq!(parse("off").unwrap(), None);
    for value in [
        "",
        "on",
        "1",
        "1:",
        "NaN:6400016",
        "inf:6400016",
        "-1:6400016",
        "1:6400015",
        "1:06400016",
        "1:+6400016",
        "1:6400016 ",
        " 1:6400016",
        "1:18446744073709551616",
    ] {
        assert!(parse(value).is_err(), "{value}");
    }
    for metres in [0.0, 0.1, 1.0, f64::MAX, 1e-300] {
        let bound = parse(&format!("{metres}:6400016")).unwrap();
        assert_eq!(requested(bound)["enabled"], true);
        let mut rounded = metres.min(f64::from(f32::MAX)) as f32;
        if f64::from(rounded) > metres {
            rounded = rounded.next_down();
        }
        if rounded < f32::MIN_POSITIVE {
            rounded = 0.0;
        }
        assert!(
            verified_live(
                bound,
                MpmSoilMotionInfo {
                    bounds: Some([rounded; 3]),
                    retained_bytes: required_bytes()
                }
            )
            .is_ok()
        );
        assert!(verified_live(bound, MpmSoilMotionInfo::default()).is_err());
    }
    let on = parse("1:6400016").unwrap();
    for bad in [
        MpmSoilMotionInfo {
            bounds: Some([0.0; 3]),
            retained_bytes: required_bytes(),
        },
        MpmSoilMotionInfo {
            bounds: Some([1.0; 3]),
            retained_bytes: required_bytes() - 1,
        },
        MpmSoilMotionInfo {
            bounds: Some([f32::NAN; 3]),
            retained_bytes: required_bytes(),
        },
    ] {
        assert!(verified_live(on, bad).is_err());
    }
    assert!(verified_live(None, MpmSoilMotionInfo::default()).is_ok());
    assert!(
        verified_live(
            None,
            MpmSoilMotionInfo {
                bounds: None,
                retained_bytes: required_bytes()
            }
        )
        .is_err()
    );
}
