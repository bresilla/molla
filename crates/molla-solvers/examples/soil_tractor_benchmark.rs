use molla_compute::Engine;
use molla_core::gpu_metrics::{DeviceScope, DispatchScope, TransferScope};
use molla_solvers::mpm::{MpmGridStorage, MpmUnilateralConfig};
use molla_solvers::soil_contact_projection_gpu::SoilBoundaryScan;
use molla_solvers::soil_tractor::{SoilTractor, SoilTractorConfig, SoilTractorTransfer};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

#[path = "soil_tractor_benchmark/history.rs"]
mod history;
#[path = "soil_tractor_benchmark/profiling.rs"]
mod profiling;
#[path = "soil_tractor_benchmark/visual.rs"]
mod visual;

#[path = "soil_tractor_benchmark/motion.rs"]
mod motion;

const DT: f64 = 1.0 / 60.0;
const DRIVE_TORQUE_NM: [f64; 4] = [300.0; 4];

fn axle_angles(
    scene: &molla_sim::runtime::RigidScene,
    axles: [molla_sim::runtime::JointHandle; 4],
) -> Result<[f64; 4], String> {
    let offsets = scene
        .model()
        .joint_coord_offset
        .host()
        .map_err(|e| e.to_string())?;
    let kinds = scene.model().joint_type.host().map_err(|e| e.to_string())?;
    let coordinates = scene.state().joint_q.host().map_err(|e| e.to_string())?;
    let mut angles = [0.0; 4];
    for (handle, angle) in axles.into_iter().zip(&mut angles) {
        let joint = scene.joint_index(handle).ok_or("missing axle")?.index();
        if kinds[joint] != molla_sim::JointType::Revolute as u32 {
            return Err("axle work requires unwrapped revolute coordinates".into());
        }
        *angle = coordinates[offsets[joint] as usize];
        if !angle.is_finite() {
            return Err("nonfinite axle coordinate".into());
        }
    }
    Ok(angles)
}

fn commanded_motor_work(initial: [f64; 4], current: [f64; 4], drive: [f64; 4]) -> f64 {
    initial
        .into_iter()
        .zip(current)
        .zip(drive)
        .map(|((before, after), torque)| -torque * (after - before))
        .sum()
}

fn write_json(path: &Path, value: &Value) -> Result<(), String> {
    let bytes = serde_json::to_vec_pretty(value).map_err(|e| e.to_string())?;
    fs::write(path, bytes).map_err(|e| e.to_string())
}

fn percentile(values: &[f64], fraction: f64) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[((fraction * sorted.len() as f64).ceil() as usize).saturating_sub(1)]
}

struct Options {
    output: PathBuf,
    warmup: usize,
    frames: usize,
    profile_mpm: bool,
    profile_preparation: bool,
    frame_loads: bool,
    contact_motion: bool,
    mass_coupling_nodes: u32,
    unilateral: Option<MpmUnilateralConfig>,
    friction_mu: Option<f32>,
    friction_walls: bool,
    rounding: molla_kernels::FeatherstoneRounding,
    execution: molla_kernels::FeatherstoneExecution,
    paired_rates: bool,
    native_prediction: bool,
    newton: bool,
    boundary_scan: SoilBoundaryScan,
    capture_every: usize,
    grid_storage: MpmGridStorage,
    transfer: SoilTractorTransfer,
    final_soil_state: bool,
    motion_bound: Option<motion::Bound>,
    motion_fusion: bool,
}

fn parse_transfer(value: &str) -> Result<SoilTractorTransfer, String> {
    if value == "atomic" {
        return Ok(SoilTractorTransfer::Atomic);
    }
    let (bytes, mode) = if let Some(bytes) = value.strip_prefix("shared-chunk:") {
        (bytes, "shared-chunk")
    } else if let Some(bytes) = value.strip_prefix("grid-reuse:") {
        (bytes, "grid-reuse")
    } else {
        (
            value
                .strip_prefix("fixed-order:")
                .ok_or("--soil-transfer must be atomic, fixed-order:BYTES, shared-chunk:BYTES or grid-reuse:BYTES")?,
            "fixed-order",
        )
    };
    let max_gather_bytes = bytes
        .parse::<u64>()
        .map_err(|_| "invalid gather scratch budget")?;
    if max_gather_bytes.to_string() != bytes {
        return Err("gather scratch budget must be canonical unsigned decimal".into());
    }
    let transfer = match mode {
        "shared-chunk" => SoilTractorTransfer::SharedChunk { max_gather_bytes },
        "grid-reuse" => SoilTractorTransfer::GridReuse { max_gather_bytes },
        _ => SoilTractorTransfer::FixedOrder { max_gather_bytes },
    };
    let cfg = SoilTractorConfig::default();
    transfer
        .validate_with_grid(
            cfg.particle_count().map_err(|e| e.to_string())?,
            cfg.grid_size,
        )
        .map_err(|e| e.to_string())?;
    Ok(transfer)
}

fn transfer_json(transfer: SoilTractorTransfer) -> Value {
    match transfer {
        SoilTractorTransfer::Atomic => json!({"mode":"atomic"}),
        SoilTractorTransfer::FixedOrder { max_gather_bytes } => {
            json!({"mode":"fixed-order","max_gather_bytes":max_gather_bytes})
        }
        SoilTractorTransfer::SharedChunk { max_gather_bytes } => {
            json!({"mode":"shared-chunk","max_gather_bytes":max_gather_bytes})
        }
        SoilTractorTransfer::GridReuse { max_gather_bytes } => {
            json!({"mode":"grid-reuse","max_gather_bytes":max_gather_bytes})
        }
    }
}

fn selected_soil_backend_label(
    storage: MpmGridStorage,
    transfer: SoilTractorTransfer,
) -> &'static str {
    if transfer == SoilTractorTransfer::Atomic {
        return soil_backend_label(storage);
    }
    if matches!(transfer, SoilTractorTransfer::GridReuse { .. }) {
        return match storage {
            MpmGridStorage::Dense => "native-grid-reuse-dense-explicit-p2g",
            MpmGridStorage::ActiveTiles => "native-grid-reuse-active-explicit-p2g",
            MpmGridStorage::Compact { .. } => "native-grid-reuse-compact-explicit-p2g",
        };
    }
    if matches!(transfer, SoilTractorTransfer::SharedChunk { .. }) {
        return match storage {
            MpmGridStorage::Dense => "native-shared-chunk-dense-explicit-p2g",
            MpmGridStorage::ActiveTiles => "native-shared-chunk-active-explicit-p2g",
            MpmGridStorage::Compact { .. } => "native-shared-chunk-compact-explicit-p2g",
        };
    }
    match storage {
        MpmGridStorage::Dense => "native-fixed-order-dense-explicit-p2g",
        MpmGridStorage::ActiveTiles => "native-fixed-order-active-explicit-p2g",
        MpmGridStorage::Compact { .. } => "native-fixed-order-compact-explicit-p2g",
    }
}

fn parse_grid_storage(value: &str) -> Result<MpmGridStorage, String> {
    match value {
        "dense" => Ok(MpmGridStorage::Dense),
        "active" => Ok(MpmGridStorage::ActiveTiles),
        _ => {
            let capacity = value
                .strip_prefix("compact:")
                .ok_or("--soil-grid must be dense, active or compact:N")?;
            let tile_capacity: u32 = capacity
                .parse()
                .map_err(|_| "invalid compact tile capacity")?;
            if tile_capacity.to_string() != capacity {
                return Err("compact tile capacity must be a canonical positive integer".into());
            }
            let storage = MpmGridStorage::Compact { tile_capacity };
            SoilTractorConfig::default()
                .validate_grid_storage(storage)
                .map_err(|e| e.to_string())?;
            Ok(storage)
        }
    }
}

fn grid_storage_json(storage: MpmGridStorage) -> Value {
    match storage {
        MpmGridStorage::Dense => json!({"layout":"dense", "tile_capacity":null}),
        MpmGridStorage::ActiveTiles => json!({"layout":"active", "tile_capacity":null}),
        MpmGridStorage::Compact { tile_capacity } => {
            json!({"layout":"compact", "tile_capacity":tile_capacity})
        }
    }
}

fn soil_backend_label(storage: MpmGridStorage) -> &'static str {
    match storage {
        MpmGridStorage::Dense => "native-float-atomic-dense-explicit-global-p2g",
        MpmGridStorage::ActiveTiles => "native-float-atomic-active-explicit-global-p2g",
        MpmGridStorage::Compact { .. } => "native-float-atomic-compact-explicit-global-p2g",
    }
}

fn options() -> Result<Options, String> {
    parse_options(std::env::args().skip(1))
}

#[test]
fn transfer_selection_is_explicit_bounded_and_manifested() {
    let parse = |extra: &[&str]| {
        parse_options(
            ["--output", "unused"]
                .into_iter()
                .chain(extra.iter().copied())
                .map(str::to_owned),
        )
    };
    assert_eq!(parse(&[]).unwrap().transfer, SoilTractorTransfer::Atomic);
    assert_eq!(
        parse_transfer("atomic").unwrap(),
        SoilTractorTransfer::Atomic
    );
    for value in [
        "",
        "fixed-order",
        "fixed-order:",
        "fixed-order:0",
        "fixed-order:3199999",
        "fixed-order:03200000",
        "fixed-order:+3200000",
        "fixed-order:3200000 ",
        "fixed-order:3.2e6",
        "fixed-order:18446744073709551616",
        "shared-chunk",
        "shared-chunk:",
        "shared-chunk:0",
        "shared-chunk:3199999",
        "shared-chunk:03200000",
        "shared-chunk:+3200000",
        "shared-chunk:3200000 ",
        "shared-chunk:3.2e6",
        "shared-chunk:18446744073709551616",
        "grid-reuse",
        "grid-reuse:",
        "grid-reuse:0",
        "grid-reuse:14075711",
        "grid-reuse:014075712",
        "grid-reuse:+14075712",
        "grid-reuse:14075712 ",
        "grid-reuse:1.4e7",
        "grid-reuse:18446744073709551616",
    ] {
        assert!(parse_transfer(value).is_err(), "{value}");
    }
    assert!(parse(&["--soil-transfer", "fixed-order:3200000"]).is_err());
    assert!(parse(&["--soil-transfer", "shared-chunk:3200000"]).is_err());
    assert!(parse(&["--soil-transfer", "grid-reuse:14075712"]).is_err());
    assert!(parse(&["--soil-transfer"]).is_err());
    assert!(parse(&["--soil-transfer", "atomic", "--soil-transfer", "atomic"]).is_err());
    let base = [
        "--paired-machine-rates",
        "1",
        "--unilateral-nodes",
        "4096",
        "--friction-mu",
        "0.3",
        "--friction-walls",
        "1",
        "--newton-soil",
        "1",
        "--native-machine-prediction",
        "1",
        "--soil-transfer",
        "fixed-order:3200000",
    ];
    for grid in ["dense", "active", "compact:1024"] {
        let mut reuse_base = base;
        *reuse_base.last_mut().unwrap() = "grid-reuse:14075712";
        let reuse_args: Vec<_> = reuse_base
            .into_iter()
            .chain(["--soil-grid", grid])
            .collect();
        let reuse = parse(&reuse_args).unwrap();
        assert_eq!(
            transfer_json(reuse.transfer),
            json!({"mode":"grid-reuse","max_gather_bytes":14075712})
        );
        assert!(
            selected_soil_backend_label(reuse.grid_storage, reuse.transfer).contains("grid-reuse")
        );
        let args: Vec<_> = base.into_iter().chain(["--soil-grid", grid]).collect();
        let options = parse(&args).unwrap();
        assert_eq!(
            transfer_json(options.transfer),
            json!({"mode":"fixed-order","max_gather_bytes":3200000})
        );
        assert!(
            selected_soil_backend_label(options.grid_storage, options.transfer)
                .contains("fixed-order")
        );
        assert_eq!(
            selected_soil_backend_label(options.grid_storage, SoilTractorTransfer::Atomic),
            soil_backend_label(options.grid_storage)
        );
        let shared_args: Vec<_> = args
            .iter()
            .map(|&arg| {
                if arg == "fixed-order:3200000" {
                    "shared-chunk:3200000"
                } else {
                    arg
                }
            })
            .collect();
        let shared = parse(&shared_args).unwrap();
        assert_eq!(
            shared.transfer,
            SoilTractorTransfer::SharedChunk {
                max_gather_bytes: 3200000
            }
        );
        assert_eq!(
            transfer_json(shared.transfer),
            json!({"mode":"shared-chunk","max_gather_bytes":3200000})
        );
        assert!(
            selected_soil_backend_label(shared.grid_storage, shared.transfer)
                .contains("shared-chunk")
        );
    }
}

#[test]
fn grid_selection_is_explicit_bounded_and_manifested() {
    let parse = |extra: &[&str]| {
        parse_options(
            ["--output", "unused"]
                .into_iter()
                .chain(extra.iter().copied())
                .map(str::to_owned),
        )
    };
    assert_eq!(parse(&[]).unwrap().grid_storage, MpmGridStorage::Dense);
    assert_eq!(
        soil_backend_label(MpmGridStorage::Dense),
        "native-float-atomic-dense-explicit-global-p2g"
    );
    for value in ["active", "compact:1", "compact:512", "compact:4096"] {
        assert!(parse(&["--soil-grid", value]).is_err());
        let options = parse(&["--soil-grid", value, "--mass-coupling-nodes", "128"]).unwrap();
        assert_eq!(options.grid_storage, parse_grid_storage(value).unwrap());
        assert_ne!(
            soil_backend_label(options.grid_storage),
            soil_backend_label(MpmGridStorage::Dense)
        );
    }
    for value in [
        "",
        "sparse",
        "compact",
        "compact:0",
        "compact:4097",
        "compact:+1",
        "compact:01",
        "compact:-1",
        "compact:1 ",
        "compact:4294967296",
    ] {
        assert!(
            parse(&["--soil-grid", value, "--mass-coupling-nodes", "128"]).is_err(),
            "{value}"
        );
    }
    assert!(parse(&["--soil-grid"]).is_err());
    assert!(parse(&["--soil-grid", "dense", "--soil-grid", "dense"]).is_err());
    let options = parse(&[
        "--soil-grid",
        "compact:512",
        "--unilateral-nodes",
        "4096",
        "--friction-mu",
        "0.3",
        "--friction-walls",
        "1",
        "--paired-machine-rates",
        "1",
        "--newton-soil",
        "1",
    ])
    .unwrap();
    assert_eq!(
        grid_storage_json(options.grid_storage),
        json!({"layout":"compact", "tile_capacity":512})
    );
    assert!(options.newton && options.paired_rates);
}

#[test]
fn paired_machine_options_require_the_supported_coupling() {
    let parse = |extra: &[&str]| {
        parse_options(
            ["--output", "unused"]
                .into_iter()
                .chain(extra.iter().copied())
                .map(str::to_owned),
        )
    };
    assert!(!parse(&[]).unwrap().paired_rates);
    for args in [
        vec!["--paired-machine-rates"],
        vec!["--paired-machine-rates", "2"],
        vec!["--paired-machine-rates", "1"],
        vec!["--paired-machine-rates", "0", "--paired-machine-rates", "0"],
    ] {
        assert!(parse(&args).is_err());
    }
    let base = [
        "--paired-machine-rates",
        "1",
        "--unilateral-nodes",
        "4096",
        "--friction-mu",
        "0.3",
        "--friction-walls",
        "1",
    ];
    assert!(parse(&base).unwrap().paired_rates);
    let specialized = parse(
        &base
            .into_iter()
            .chain(["--machine-execution", "joint-types", "--newton-soil", "1"])
            .collect::<Vec<_>>(),
    )
    .unwrap();
    assert!(specialized.paired_rates && specialized.newton);
    assert_eq!(
        specialized.execution,
        molla_kernels::FeatherstoneExecution::ExperimentalJointTypes
    );
    assert!(!parse(&base).unwrap().newton);
    assert!(parse(&["--newton-soil", "1"]).is_err());
    assert!(
        parse(
            &base
                .into_iter()
                .chain(["--newton-soil", "1"])
                .collect::<Vec<_>>()
        )
        .unwrap()
        .newton
    );
    assert!(
        parse(
            &base
                .into_iter()
                .chain(["--newton-soil", "2"])
                .collect::<Vec<_>>()
        )
        .is_err()
    );
    assert!(
        parse(
            &base
                .into_iter()
                .chain(["--newton-soil", "0", "--newton-soil", "1"])
                .collect::<Vec<_>>()
        )
        .is_err()
    );
    for extra in [
        ["--machine-rounding", "shared"],
        ["--machine-execution", "scratch"],
    ] {
        assert!(parse(&base.into_iter().chain(extra).collect::<Vec<_>>()).is_err());
    }
    assert!(parse(&base[..base.len() - 2]).is_err());
}

#[test]
fn native_prediction_selection_is_explicit_and_admitted() {
    let parse = |extra: &[&str]| {
        parse_options(
            ["--output", "unused"]
                .into_iter()
                .chain(extra.iter().copied())
                .map(str::to_owned),
        )
    };
    assert!(!parse(&[]).unwrap().native_prediction);
    for args in [
        vec!["--native-machine-prediction"],
        vec!["--native-machine-prediction", "2"],
        vec!["--native-machine-prediction", "1"],
        vec![
            "--native-machine-prediction",
            "0",
            "--native-machine-prediction",
            "0",
        ],
    ] {
        assert!(parse(&args).is_err());
    }
    let base = [
        "--paired-machine-rates",
        "1",
        "--unilateral-nodes",
        "4096",
        "--friction-mu",
        "0.3",
        "--friction-walls",
        "1",
        "--newton-soil",
        "1",
    ];
    assert!(!parse(&base).unwrap().native_prediction);
    let enabled: Vec<_> = base
        .into_iter()
        .chain(["--native-machine-prediction", "1"])
        .collect();
    assert!(parse(&enabled).unwrap().native_prediction);
    let specialized: Vec<_> = enabled
        .into_iter()
        .chain(["--machine-execution", "joint-types"])
        .collect();
    assert!(parse(&specialized).is_err());
}

#[test]
fn preparation_profile_selection_is_explicit_and_requires_newton() {
    let parse = |extra: &[&str]| {
        parse_options(
            ["--output", "unused"]
                .into_iter()
                .chain(extra.iter().copied())
                .map(str::to_owned),
        )
    };
    assert!(!parse(&[]).unwrap().profile_preparation);
    for args in [
        vec!["--profile-preparation"],
        vec!["--profile-preparation", "2"],
        vec!["--profile-preparation", "1"],
        vec!["--profile-preparation", "0", "--profile-preparation", "0"],
    ] {
        assert!(parse(&args).is_err());
    }
    let mut args = vec![
        "--profile-preparation",
        "1",
        "--paired-machine-rates",
        "1",
        "--unilateral-nodes",
        "4096",
        "--friction-mu",
        "0.3",
        "--friction-walls",
        "1",
        "--newton-soil",
        "1",
    ];
    assert!(parse(&args).is_err());
    args.extend(["--profile-mpm", "1"]);
    assert!(parse(&args).unwrap().profile_preparation);
}

#[test]
fn wall_friction_options_are_explicit_and_require_coefficients() {
    let parse = |args: &[&str]| {
        parse_options(
            ["--output", "unused"]
                .into_iter()
                .chain(args.iter().copied())
                .map(str::to_owned),
        )
    };
    assert!(!parse(&[]).unwrap().friction_walls);
    for value in ["0", "1"] {
        let options = parse(&[
            "--unilateral-nodes",
            "4096",
            "--friction-mu",
            "0.3",
            "--friction-walls",
            value,
        ])
        .unwrap();
        assert_eq!(options.friction_walls, value == "1");
        let scope = unilateral_config_json(
            options.unilateral,
            options.friction_mu,
            options.friction_walls,
            options.newton,
            false,
        );
        assert_eq!(
            scope["scope"]
                .as_str()
                .unwrap()
                .contains("shared fixed-wall response"),
            value == "1"
        );
    }
    for args in [
        vec!["--friction-walls", "0"],
        vec!["--friction-walls", "1"],
        vec![
            "--unilateral-nodes",
            "4096",
            "--friction-mu",
            "0.3",
            "--friction-walls",
            "2",
        ],
        vec![
            "--unilateral-nodes",
            "4096",
            "--friction-mu",
            "0.3",
            "--friction-walls",
            "1",
            "--friction-walls",
            "1",
        ],
        vec![
            "--unilateral-nodes",
            "4096",
            "--friction-mu",
            "0.3",
            "--friction-walls",
        ],
    ] {
        assert!(parse(&args).is_err(), "{args:?}");
    }
}

#[test]
fn contact_motion_options_require_unilateral_and_reject_duplicates() {
    let parse = |args: &[&str]| parse_options(args.iter().map(|s| (*s).to_owned()));
    assert!(!parse(&["--output", "unused"]).unwrap().contact_motion);
    assert!(parse(&["--output", "unused", "--contact-motion", "1"]).is_err());
    assert!(parse(&["--output", "unused", "--contact-motion", "2"]).is_err());
    assert!(
        parse(&[
            "--output",
            "unused",
            "--contact-motion",
            "0",
            "--contact-motion",
            "0"
        ])
        .is_err()
    );
    assert!(
        parse(&[
            "--output",
            "unused",
            "--unilateral-nodes",
            "512",
            "--contact-motion",
            "1"
        ])
        .unwrap()
        .contact_motion
    );
}

fn fixed_grid_boundary_scope(options: &Options) -> &'static str {
    if options.friction_walls {
        "simultaneous shared-machine Coulomb contact and normal-only fixed-wall response; free-grid tangential damping unchanged"
    } else if options.friction_mu.is_some() {
        "fixed-wall friction contact unsupported; faults without frictionless fallback"
    } else if options.unilateral.is_some() {
        "coupled inward-normal inequalities; free-grid tangential damping unchanged; frictionless machine contact"
    } else if options.mass_coupling_nodes > 0 {
        "finite-mass sticking contact with boundary QP; free-grid tangential damping unchanged"
    } else {
        "prescribed-velocity contact and existing MPM grid boundary handling; no simultaneous finite-mass wall solve"
    }
}

#[test]
fn boundary_scope_matches_every_coupling_mode() {
    let parse = |args: &[&str]| {
        parse_options(
            [vec!["--output", "unused"], args.to_vec()]
                .concat()
                .into_iter()
                .map(str::to_owned),
        )
        .unwrap()
    };
    for (args, scope) in [
        (vec![], "prescribed-velocity"),
        (vec!["--mass-coupling-nodes", "64"], "boundary QP"),
        (
            vec!["--unilateral-nodes", "4096"],
            "frictionless machine contact",
        ),
        (
            vec!["--unilateral-nodes", "4096", "--friction-mu", "0.3"],
            "unsupported",
        ),
        (
            vec![
                "--unilateral-nodes",
                "4096",
                "--friction-mu",
                "0.3",
                "--friction-walls",
                "1",
            ],
            "simultaneous shared-machine Coulomb",
        ),
    ] {
        assert!(fixed_grid_boundary_scope(&parse(&args)).contains(scope));
    }
}

#[test]
fn newton_metadata_distinguishes_global_and_local_solvers() {
    let config = MpmUnilateralConfig {
        node_capacity: 4096,
        search_distance: 0.05,
        max_iterations: 32,
    };
    let global = unilateral_config_json(Some(config), Some(0.3), true, true, false);
    let local = unilateral_config_json(Some(config), Some(0.3), true, false, false);
    let protected = unilateral_config_json(Some(config), Some(0.3), true, true, true);
    assert_eq!(
        protected["newton_method"],
        if cfg!(feature = "soa-scratch-newton") {
            "fischer-conditioned-protected-stalled-schur-norm-publication-products-retained-local-shared-core-stream-block-stagnant-branch-grouped-eight-parallel-indices-soa-scratch-256-pairs"
        } else if cfg!(feature = "parallel-indices-newton") {
            "fischer-conditioned-protected-stalled-schur-norm-publication-products-retained-local-shared-core-stream-block-stagnant-branch-grouped-eight-parallel-indices-256-pairs"
        } else if cfg!(feature = "grouped-eight-newton") {
            "fischer-conditioned-protected-stalled-schur-norm-publication-products-retained-local-shared-core-stream-block-stagnant-branch-grouped-eight-256-pairs"
        } else if cfg!(feature = "grouped-sum-newton") {
            "fischer-conditioned-protected-stalled-schur-norm-publication-products-retained-local-shared-core-stream-block-stagnant-branch-grouped-sum-256-pairs"
        } else if cfg!(feature = "stagnant-branch-newton") {
            "fischer-conditioned-protected-stalled-schur-norm-publication-products-retained-local-shared-core-stream-block-stagnant-branch-256-pairs"
        } else if cfg!(feature = "stream-block-newton") {
            "fischer-conditioned-protected-stalled-schur-norm-publication-products-retained-local-shared-core-stream-block-256-pairs"
        } else if cfg!(feature = "shared-conditioned-core-newton") {
            "fischer-conditioned-protected-stalled-schur-norm-publication-products-retained-local-shared-core-256-pairs"
        } else if cfg!(feature = "conditioned-projection-cache-newton") {
            "fischer-conditioned-protected-stalled-schur-norm-publication-products-retained-local-projection-cache-256-pairs"
        } else if cfg!(feature = "retained-local-newton") {
            "fischer-conditioned-protected-stalled-schur-norm-publication-products-retained-local-256-pairs"
        } else if cfg!(feature = "publication-products-newton") {
            "fischer-conditioned-protected-stalled-schur-norm-publication-products-256-pairs"
        } else if cfg!(feature = "active-publication-newton") {
            "fischer-conditioned-protected-stalled-schur-norm-active-publication-256-pairs"
        } else if cfg!(feature = "norm-tree-newton") {
            "fischer-conditioned-protected-stalled-schur-norm-256-pairs"
        } else if cfg!(feature = "schur-newton") {
            "fischer-conditioned-protected-stalled-schur-256-pairs"
        } else if cfg!(feature = "volatile-newton") {
            "fischer-conditioned-protected-stalled-volatile-256-pairs"
        } else if cfg!(feature = "parallel-newton") {
            "fischer-conditioned-protected-stalled-parallel-256-pairs"
        } else if cfg!(feature = "cached-newton") {
            "fischer-conditioned-protected-stalled-cached-256-pairs"
        } else if cfg!(feature = "wide-newton") {
            "fischer-conditioned-protected-stalled-256-pairs"
        } else {
            "fischer-conditioned-protected-stalled-pairs"
        }
    );
    assert_eq!(global["max_iterations"], 32);
    assert_eq!(global["newton_workgroup_lanes"], 64);
    assert_eq!(
        protected["newton_workgroup_lanes"],
        molla_solvers::soil_newton_gpu::execution_lanes(true)
    );
    assert!(
        global["scope"]
            .as_str()
            .unwrap()
            .contains("globally coupled")
    );
    assert!(!global["scope"].as_str().unwrap().contains("local Newton"));
    assert_eq!(global["newton_method"], "fischer-conditioned-stalled-pairs");
    assert!(local["newton_method"].is_null());
    assert!(local["scope"].as_str().unwrap().contains("local Newton"));
}

fn unilateral_config_json(
    config: Option<MpmUnilateralConfig>,
    friction: Option<f32>,
    walls: bool,
    newton: bool,
    protected: bool,
) -> Value {
    config.map_or(Value::Null, |c| {
        json!({
            "node_capacity_total": c.node_capacity,
            "search_distance_m": c.search_distance,
            "max_iterations": c.max_iterations,
            "newton_method": if newton { Some(if protected && cfg!(feature="soa-scratch-newton") {"fischer-conditioned-protected-stalled-schur-norm-publication-products-retained-local-shared-core-stream-block-stagnant-branch-grouped-eight-parallel-indices-soa-scratch-256-pairs"} else if protected && cfg!(feature="parallel-indices-newton") {"fischer-conditioned-protected-stalled-schur-norm-publication-products-retained-local-shared-core-stream-block-stagnant-branch-grouped-eight-parallel-indices-256-pairs"} else if protected && cfg!(feature="grouped-eight-newton") {"fischer-conditioned-protected-stalled-schur-norm-publication-products-retained-local-shared-core-stream-block-stagnant-branch-grouped-eight-256-pairs"} else if protected && cfg!(feature="grouped-sum-newton") {"fischer-conditioned-protected-stalled-schur-norm-publication-products-retained-local-shared-core-stream-block-stagnant-branch-grouped-sum-256-pairs"} else if protected && cfg!(feature="stagnant-branch-newton") {"fischer-conditioned-protected-stalled-schur-norm-publication-products-retained-local-shared-core-stream-block-stagnant-branch-256-pairs"} else if protected && cfg!(feature="stream-block-newton") {"fischer-conditioned-protected-stalled-schur-norm-publication-products-retained-local-shared-core-stream-block-256-pairs"} else if protected && cfg!(feature="shared-conditioned-core-newton") {"fischer-conditioned-protected-stalled-schur-norm-publication-products-retained-local-shared-core-256-pairs"} else if protected && cfg!(feature="conditioned-projection-cache-newton") {"fischer-conditioned-protected-stalled-schur-norm-publication-products-retained-local-projection-cache-256-pairs"} else if protected && cfg!(feature="retained-local-newton") {"fischer-conditioned-protected-stalled-schur-norm-publication-products-retained-local-256-pairs"} else if protected && cfg!(feature="publication-products-newton") {"fischer-conditioned-protected-stalled-schur-norm-publication-products-256-pairs"} else if protected && cfg!(feature="active-publication-newton") {"fischer-conditioned-protected-stalled-schur-norm-active-publication-256-pairs"} else if protected && cfg!(feature="norm-tree-newton") {"fischer-conditioned-protected-stalled-schur-norm-256-pairs"} else if protected && cfg!(feature="schur-newton") {"fischer-conditioned-protected-stalled-schur-256-pairs"} else if protected && cfg!(feature="volatile-newton") {"fischer-conditioned-protected-stalled-volatile-256-pairs"} else if protected && cfg!(feature="parallel-newton") {"fischer-conditioned-protected-stalled-parallel-256-pairs"} else if protected && cfg!(feature="cached-newton") {"fischer-conditioned-protected-stalled-cached-256-pairs"} else if protected && cfg!(feature="wide-newton") {"fischer-conditioned-protected-stalled-256-pairs"} else if protected {"fischer-conditioned-protected-stalled-pairs"} else {"fischer-conditioned-stalled-pairs"}) } else { None },
            "newton_workgroup_lanes": if newton {Some(molla_solvers::soil_newton_gpu::execution_lanes(protected))} else {None},
            "newton_start": if newton { Some(if cfg!(feature = "adjacent-newton") {"matched-node-body-history-within-frame-adjacent-parity"} else if cfg!(feature = "warm-newton") {"matched-node-body-history-within-frame-per-parity"} else {"cold"}) } else { None },
            "scope": if newton {
                "frozen single-body normal and Coulomb friction per grid node; globally coupled machine and fixed-wall Newton solve; adaptive smoothing, conditioned local sliding release and iterative-first direct fallback; configured outer iterations and 64 total Krylov iterations per linear solve; no penetration repair"
            } else if friction.is_some() && walls {
                "frozen single-body normal and Coulomb friction per grid node; shared fixed-wall response; 64 local Newton iterations; no penetration repair"
            } else if friction.is_some() {
                "frozen single-body normal and Coulomb friction per grid node; fixed-wall contact rejects; no penetration repair"
            } else {
                "frozen single-body normal per grid node; frictionless; no penetration repair"
            }
        })
    })
}

fn parse_options(mut args: impl Iterator<Item = String>) -> Result<Options, String> {
    let mut output = None;
    let mut warmup = None;
    let mut frames = None;
    let mut profile_mpm = None;
    let mut profile_preparation = None;
    let mut frame_loads = None;
    let mut contact_motion = None;
    let mut mass_coupling_nodes = None;
    let mut unilateral_nodes = None;
    let mut unilateral_margin = None;
    let mut unilateral_iterations = None;
    let mut friction_mu = None;
    let mut friction_walls = None;
    let mut rounding = None;
    let mut execution = None;
    let mut paired_rates = None;
    let mut native_prediction = None;
    let mut newton = None;
    let mut boundary_scan = None;
    let mut capture_every = None;
    let mut grid_storage = None;
    let mut transfer = None;
    let mut final_soil_state = None;
    let mut motion_bound = None;
    let mut motion_fusion = None;
    while let Some(key) = args.next() {
        let value = args
            .next()
            .ok_or_else(|| format!("missing value for {key}"))?;
        match key.as_str() {
            "--soil-motion-fusion" if motion_fusion.is_none() => {
                motion_fusion = Some(motion::parse_fusion(&value)?);
            }
            "--soil-motion-bound" if motion_bound.is_none() => {
                motion_bound = Some(motion::parse(&value)?);
            }
            "--soil-transfer" if transfer.is_none() => {
                transfer = Some(parse_transfer(&value)?);
            }
            "--profile-preparation" if profile_preparation.is_none() => {
                profile_preparation = Some(match value.as_str() {
                    "0" => false,
                    "1" => true,
                    _ => return Err("--profile-preparation must be 0 or 1".into()),
                });
            }
            "--native-machine-prediction" if native_prediction.is_none() => {
                native_prediction = Some(match value.as_str() {
                    "0" => false,
                    "1" => true,
                    _ => return Err("--native-machine-prediction must be 0 or 1".into()),
                });
            }
            "--soil-grid" if grid_storage.is_none() => {
                grid_storage = Some(parse_grid_storage(&value)?);
            }
            "--newton-soil" if newton.is_none() => {
                newton = Some(match value.as_str() {
                    "0" => false,
                    "1" => true,
                    _ => return Err("--newton-soil must be 0 or 1".into()),
                });
            }
            "--paired-machine-rates" if paired_rates.is_none() => {
                paired_rates = Some(match value.as_str() {
                    "0" => false,
                    "1" => true,
                    _ => return Err("--paired-machine-rates must be 0 or 1".into()),
                });
            }
            "--friction-walls" if friction_walls.is_none() => {
                friction_walls = Some(match value.as_str() {
                    "0" => false,
                    "1" => true,
                    _ => return Err("--friction-walls must be 0 or 1".into()),
                });
            }
            "--friction-mu" if friction_mu.is_none() => {
                let mu = value.parse::<f32>().map_err(|e| e.to_string())?;
                if !mu.is_finite() || mu < 0.0 {
                    return Err("--friction-mu must be finite and nonnegative".into());
                }
                friction_mu = Some(mu);
            }
            "--contact-motion" if contact_motion.is_none() => {
                contact_motion = Some(match value.as_str() {
                    "0" => false,
                    "1" => true,
                    _ => return Err("--contact-motion must be 0 or 1".into()),
                });
            }
            "--capture-every" if capture_every.is_none() => {
                capture_every = Some(value.parse::<usize>().map_err(|e| e.to_string())?);
            }
            "--final-soil-state" if final_soil_state.is_none() => {
                final_soil_state = Some(match value.as_str() {
                    "0" => false,
                    "1" => true,
                    _ => return Err("--final-soil-state must be 0 or 1".into()),
                });
            }
            "--output" if output.is_none() => output = Some(PathBuf::from(value)),
            "--warmup" if warmup.is_none() => {
                warmup = Some(value.parse::<usize>().map_err(|e| e.to_string())?)
            }
            "--frames" if frames.is_none() => {
                frames = Some(value.parse::<usize>().map_err(|e| e.to_string())?)
            }
            "--profile-mpm" if profile_mpm.is_none() => {
                profile_mpm = Some(match value.as_str() {
                    "0" => false,
                    "1" => true,
                    _ => return Err("--profile-mpm must be 0 or 1".into()),
                });
            }
            "--frame-loads" if frame_loads.is_none() => {
                frame_loads = Some(match value.as_str() {
                    "0" => false,
                    "1" => true,
                    _ => return Err("--frame-loads must be 0 or 1".into()),
                });
            }
            "--mass-coupling-nodes" if mass_coupling_nodes.is_none() => {
                mass_coupling_nodes = Some(value.parse::<u32>().map_err(|e| e.to_string())?);
            }
            "--unilateral-nodes" if unilateral_nodes.is_none() => {
                unilateral_nodes = Some(value.parse::<u32>().map_err(|e| e.to_string())?);
            }
            "--unilateral-margin" if unilateral_margin.is_none() => {
                unilateral_margin = Some(value.parse::<f32>().map_err(|e| e.to_string())?);
            }
            "--unilateral-iterations" if unilateral_iterations.is_none() => {
                unilateral_iterations = Some(value.parse::<u32>().map_err(|e| e.to_string())?);
            }
            "--machine-rounding" if rounding.is_none() => {
                rounding = Some(match value.as_str() {
                    "storage" => molla_kernels::FeatherstoneRounding::Storage,
                    "shared" => molla_kernels::FeatherstoneRounding::ExperimentalShared,
                    _ => return Err("--machine-rounding must be storage or shared".into()),
                });
            }
            "--machine-execution" if execution.is_none() => {
                execution = Some(match value.as_str() {
                    "generic" => molla_kernels::FeatherstoneExecution::Generic,
                    "scratch" => molla_kernels::FeatherstoneExecution::ExperimentalWorkgroupScratch,
                    "single-lane" => molla_kernels::FeatherstoneExecution::ExperimentalSingleLane,
                    "joint-types" => molla_kernels::FeatherstoneExecution::ExperimentalJointTypes,
                    "constraint-scratch" => {
                        molla_kernels::FeatherstoneExecution::ExperimentalConstraintScratch
                    }
                    _ => {
                        return Err(
                            "--machine-execution must be generic, scratch, constraint-scratch, single-lane or joint-types"
                                .into(),
                        );
                    }
                });
            }
            "--boundary-scan" if boundary_scan.is_none() => {
                boundary_scan = Some(match value.as_str() {
                    "full" => SoilBoundaryScan::Full,
                    "cached" => SoilBoundaryScan::ExperimentalCached,
                    _ => return Err("--boundary-scan must be full or cached".into()),
                });
            }
            _ => return Err(format!("unknown or duplicate option {key}")),
        }
    }
    let unilateral = if let Some(node_capacity) = unilateral_nodes {
        let config = MpmUnilateralConfig {
            node_capacity,
            search_distance: unilateral_margin.unwrap_or(0.05),
            max_iterations: unilateral_iterations.unwrap_or(64),
        };
        config.validate().map_err(|e| e.to_string())?;
        Some(config)
    } else {
        if unilateral_margin.is_some() || unilateral_iterations.is_some() {
            return Err("unilateral tuning requires --unilateral-nodes".into());
        }
        None
    };
    let result = Options {
        output: output.ok_or("--output is required")?,
        warmup: warmup.unwrap_or(3),
        frames: frames.unwrap_or(30),
        profile_mpm: profile_mpm.unwrap_or(false),
        profile_preparation: profile_preparation.unwrap_or(false),
        frame_loads: frame_loads.unwrap_or(false),
        contact_motion: contact_motion.unwrap_or(false),
        mass_coupling_nodes: mass_coupling_nodes.unwrap_or(0),
        unilateral,
        friction_mu,
        friction_walls: friction_walls.unwrap_or(false),
        rounding: rounding.unwrap_or_default(),
        execution: execution.unwrap_or_default(),
        paired_rates: paired_rates.unwrap_or(false),
        native_prediction: native_prediction.unwrap_or(false),
        newton: newton.unwrap_or(false),
        boundary_scan: boundary_scan.unwrap_or_default(),
        capture_every: capture_every.unwrap_or(0),
        grid_storage: grid_storage.unwrap_or_default(),
        transfer: transfer.unwrap_or_default(),
        final_soil_state: final_soil_state.unwrap_or(false),
        motion_bound: motion_bound.unwrap_or(None),
        motion_fusion: motion_fusion.unwrap_or(false),
    };
    if result.motion_fusion && result.motion_bound.is_none() {
        return Err("--soil-motion-fusion requires an active --soil-motion-bound".into());
    }
    if result.grid_storage != MpmGridStorage::Dense
        && result.mass_coupling_nodes == 0
        && result.unilateral.is_none()
    {
        return Err("active/compact soil requires explicit projected coupling".into());
    }
    if !(1..=30).contains(&result.warmup) || !(2..=300).contains(&result.frames) {
        return Err("warmup must be 1..30 and measured frames 2..300".into());
    }
    if result.boundary_scan != SoilBoundaryScan::Full && result.mass_coupling_nodes == 0 {
        return Err("cached boundary scan requires nonzero mass-coupling-nodes".into());
    }
    if result.unilateral.is_some() && result.mass_coupling_nodes != 0 {
        return Err("unilateral and sticking coupling are mutually exclusive".into());
    }
    if result.contact_motion && result.unilateral.is_none() {
        return Err("contact motion requires --unilateral-nodes".into());
    }
    if result.friction_mu.is_some() && result.unilateral.is_none() {
        return Err("--friction-mu requires --unilateral-nodes".into());
    }
    if friction_walls.is_some() && result.friction_mu.is_none() {
        return Err("--friction-walls requires --friction-mu".into());
    }
    if result.paired_rates
        && (!result.friction_walls
            || result.friction_mu.is_none()
            || result.rounding != molla_kernels::FeatherstoneRounding::Storage
            || !matches!(
                result.execution,
                molla_kernels::FeatherstoneExecution::Generic
                    | molla_kernels::FeatherstoneExecution::ExperimentalJointTypes
            ))
    {
        return Err("paired machine rates require shared-wall friction, storage rounding and generic or joint-types execution".into());
    }
    if result.newton && !result.paired_rates {
        return Err("--newton-soil requires --paired-machine-rates 1".into());
    }
    if result.profile_preparation && (!result.newton || !result.profile_mpm) {
        return Err("preparation profiling requires MPM profiling and Newton coupling".into());
    }
    if result.native_prediction
        && (!result.newton || result.execution != molla_kernels::FeatherstoneExecution::Generic)
    {
        return Err("native prediction requires paired Newton soil and generic execution".into());
    }
    visual::frames(result.warmup + result.frames, result.capture_every)?;
    if result.transfer != SoilTractorTransfer::Atomic && !result.native_prediction {
        return Err("fixed-order transfer requires native machine prediction".into());
    }
    Ok(result)
}

fn run(options: &Options) -> Result<(), String> {
    let diagnostic_inner_limit =
        molla_solvers::soil_newton_gpu::diagnostic_inner_limit().map_err(|e| e.to_string())?;
    if diagnostic_inner_limit.is_some() && !options.newton {
        return Err("Newton diagnostic inner limit requires --newton-soil 1".into());
    }
    let visual_frames = visual::frames(options.warmup + options.frames, options.capture_every)?;
    let mut features = wgpu::Features::SHADER_FLOAT32_ATOMIC;
    if options.native_prediction {
        features |= wgpu::Features::SHADER_F64;
    }
    if options.paired_rates {
        features |= molla_kernels::FeatherstoneKernel::paired_drift_required_features();
    }
    #[cfg(all(feature = "protected-newton", not(target_arch = "wasm32")))]
    if options.newton {
        features |= wgpu::Features::PASSTHROUGH_SHADERS;
    }
    if options.profile_mpm {
        features |= wgpu::Features::TIMESTAMP_QUERY
            | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS
            | wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES;
    }
    let engine = Arc::new(Engine::new_gpu_with_features(features).map_err(|e| e.to_string())?);
    let protected = molla_solvers::soil_newton_gpu::protected_enabled(&engine);
    let adapter = engine.adapter_info().ok_or("missing adapter identity")?;
    if adapter.is_software {
        return Err("software adapters are not a machine benchmark".into());
    }
    let cfg = SoilTractorConfig::default();
    let substeps = (DT / cfg.max_substep_s).ceil() as usize;
    let mut manifest = json!({
        "schema": 1, "case": "native-tractor-50k", "mode": "diagnostic",
        "scope": "completed machine plus soil; renderer and terrain export excluded",
        "physical_calibration": "uncalibrated", "acceptance": false,
        "adapter": {"name": adapter.name, "backend": adapter.backend,
            "driver": adapter.driver, "driver_info": adapter.driver_info,
            "vendor": adapter.vendor, "is_software": adapter.is_software},
        "backend": selected_soil_backend_label(options.grid_storage, options.transfer),
        "machine_addition_backend": options.rounding.label(),
        "particles": cfg.particle_count().map_err(|e| e.to_string())?,
        "seed": cfg.seed, "grid_size": cfg.grid_size, "cell_size_m": cfg.cell_size,
        "grid_origin_m": [-3.0, -0.5, -3.0], "soil_bed_size_m": [5.6, 0.25, 3.2],
        "youngs_pa": cfg.youngs_pa, "density_kg_m3": cfg.density_kg_m3,
        "bodies": 6, "soil_bindings": 5, "machine_mass_kg": 860,
        "wheel_torque_nm": [300, 300, 300, 300], "frame_simulated_seconds": DT,
        "substeps_per_frame": substeps, "substep_seconds": DT / substeps as f64,
        "warmup_frames": options.warmup, "measured_frames": options.frames,
        "coupling_budget_bytes": cfg.coupling_budget_bytes,
        "soil_only_ms": null, "renderer_enabled": false, "terrain_export_enabled": false,
        "mpm_kernel_profile_enabled": options.profile_mpm,
        "mpm_profile_scope": "kernel GPU durations and relative offsets; GPU span includes intervening work/idle, not a soil-only budget; host submission/completion durations are separate",
        "machine_profile_scope": "exact refresh/dynamics/momentum and optional soil-corrected-drift GPU passes; offsets use the first machine timestamp as their own origin; bridge/CRBA/Jacobian/guard/publication kernels and queue gaps excluded",
        "energy_scope": "machine kinetic and gravitational potential only; excludes soil energy and motor/contact work",
        "optional_observations": "final soil positions and last-substep loads, outside measured windows",
        "state_artifacts": "machine-states.jsonl contains existing host mirrors after each committed frame; final-particles.bin is particle-order xyz little-endian f32 from final observation; all artifact writes are outside measured windows",
        "resource_counters": "host-thread instrumented dispatches/transfers and buffer-allocation observations",
        "queue_submissions": null, "pipeline_creations": null, "bind_group_creations": null,
        "percentiles": "nearest-rank; diagnostic sample count, not release acceptance"
    });
    manifest["requested_soil_grid"] = grid_storage_json(options.grid_storage);
    manifest["requested_soil_transfer"] = transfer_json(options.transfer);
    manifest["requested_soil_motion_bound"] = motion::requested(options.motion_bound);
    manifest["live_soil_motion_bound"] = Value::Null;
    manifest["requested_soil_motion_fusion"] = json!(options.motion_fusion);
    manifest["live_soil_motion_fusion"] = Value::Null;
    manifest["soil_motion_bound_scope"] = json!(
        "per-axis metres per soil substep; retained proposal storage budget separate from coupling, ordering and driver storage"
    );
    manifest["final_soil_state_enabled"] = json!(options.final_soil_state);
    manifest["final_soil_state_scope"] = json!(history::SCOPE);
    manifest["live_soil_transfer"] = Value::Null;
    manifest["soil_transfer_budget_scope"] = json!(
        "gather stress scratch plus selected grid-reuse keys, logical nodes, tiles and header; ordering and driver storage excluded; retained ordering bytes reported separately"
    );
    manifest["live_soil_grid"] = Value::Null;
    manifest["physical_grid_and_metadata_bytes"] = Value::Null;
    manifest["newton_diagnostic_inner_limit"] = json!(diagnostic_inner_limit);
    manifest["newton_diagnostic_limit_scope"] = json!(
        "checked between outer iterations and damping attempts; may overshoot by one Krylov solve; not performance acceptance"
    );
    manifest["machine_rate_encoding"] = json!(if options.paired_rates { "f32x2" } else { "f32" });
    manifest["machine_paired_drift_build_variant"] = json!(
        options
            .paired_rates
            .then_some(molla_kernels::FeatherstoneKernel::paired_drift_build_variant())
    );
    manifest["soil_newton_enabled"] = json!(options.newton);
    manifest["soil_newton_preparation_backend"] = json!(options.newton.then_some(
        molla_solvers::soil_newton_prepare_gpu::ExperimentalSoilNewtonPrepareGpu::backend_label()
    ));
    manifest["machine_prediction_stages"] = json!(if options.native_prediction {
        5
    } else if options.paired_rates {
        2
    } else {
        1
    });
    manifest["preparation_profile_enabled"] = json!(options.profile_preparation);
    manifest["machine_prediction_backend"] = json!(if options.native_prediction {
        "native-controls-bias-gram-solve-paired-drift"
    } else {
        "featherstone-aba"
    });
    manifest["machine_prediction_mass_encoding"] = json!(if options.native_prediction {
        "f32x2"
    } else {
        "f32"
    });
    manifest["soil_mass_operator"] = json!(if options.native_prediction {
        "borrowed-native-input-configuration-per-parity"
    } else {
        "separate-crba"
    });
    manifest["soil_jacobian_operator"] = json!(if options.native_prediction {
        "native-input-com-row-copies-per-parity"
    } else {
        "separate-jacobian-dispatch"
    });
    manifest["requested_boundary_scan_backend"] = json!(options.boundary_scan.label());
    manifest["coupling_backend"] = json!(if options.newton {
        "experimental-device-prepared-newton-wall-friction"
    } else if options.friction_walls {
        "experimental-shared-wall-coulomb-friction"
    } else if options.friction_mu.is_some() {
        "experimental-joint-normal-coulomb-friction-no-fixed-walls"
    } else if options.unilateral.is_some() {
        "experimental-unilateral-normal-certified-floor-projection"
    } else if options.mass_coupling_nodes > 0 {
        "finite-mass-sticking-projection-with-boundary-qp"
    } else {
        "lagged-prescribed-velocity-force"
    });
    manifest["mass_coupling_nodes_per_body"] = json!(options.mass_coupling_nodes);
    manifest["unilateral_config"] = unilateral_config_json(
        options.unilateral,
        options.friction_mu,
        options.friction_walls,
        options.newton,
        protected,
    );
    manifest["friction_fixed_walls"] = json!(options.friction_mu.map(|_| options.friction_walls));
    manifest["friction_config"] = options.friction_mu.map_or(Value::Null, |mu| json!({
        "coefficient": mu, "assignment": "all four wheels and tine; chassis has no soil binding",
        "physical_calibration": "uncalibrated explicit diagnostic input",
        "fixed_wall_contact": if options.friction_walls {"simultaneous shared-machine wall/Coulomb solve; failures roll back"} else {"unsupported; faults and rolls back, never frictionless fallback"}
    }));
    manifest["machine_drift_backend"] = json!("unrestricted-specialized-with-generic-fallback");
    manifest["machine_dynamics_backend"] = json!(options.execution.label());
    manifest["visual_capture_frames"] = json!(visual_frames);
    manifest["visual_capture_scope"] = json!(
        "opt-in full-position readbacks after committed frames, outside wall/resource timing; not performance acceptance or restart checkpoints"
    );
    manifest["fixed_grid_boundary_scope"] = json!(fixed_grid_boundary_scope(&options));
    manifest["frame_load_observation_enabled"] = json!(options.frame_loads);
    manifest["contact_motion_observation_enabled"] = json!(options.contact_motion);
    manifest["contact_motion_scope"] = json!(
        "last-substep positive-normal-impulse-weighted corrected world soil and tangential machine-point-minus-soil velocities at contact configuration before drift; signed mean may cancel, mean local speed magnitude does not; no support is null; not frame-averaged or a normalized slip ratio; explicit GPU reduction/readback outside physics timing"
    );
    manifest["frame_load_scope"] = json!(
        "GPU sum of substep force, instantaneous-COM torque and fixed-world-origin torque times dt at contact configuration before drift; one additional dispatch per substep, explicit readback outside wall timing; soil reaction only, not total external impulse, work or a conservation verdict"
    );
    write_json(&options.output.join("manifest.json"), &manifest)?;
    eprintln!(
        "building native 50k tractor, {} substeps per 1/60 s frame",
        substeps
    );
    let started = Instant::now();
    let mut tractor = if options.native_prediction {
        SoilTractor::new_native_prediction_with_newton_transfer(
            engine.clone(),
            cfg,
            options.unilateral.ok_or("missing Newton configuration")?,
            options.friction_mu.ok_or("missing Newton friction")?,
            options.grid_storage,
            DT,
            options.transfer,
        )
    } else if options.newton {
        SoilTractor::new_native_explicit_with_paired_newton_grid(
            engine.clone(),
            cfg,
            options.unilateral.ok_or("missing Newton configuration")?,
            options.friction_mu.ok_or("missing Newton friction")?,
            options.execution,
            options.grid_storage,
        )
    } else if options.paired_rates {
        SoilTractor::new_native_explicit_with_paired_wall_friction_grid(
            engine.clone(),
            cfg,
            options
                .unilateral
                .ok_or("missing paired wall configuration")?,
            options.friction_mu.ok_or("missing paired wall friction")?,
            options.execution,
            options.grid_storage,
        )
    } else {
        SoilTractor::new_native_explicit_with_grid_options(
            engine.clone(),
            cfg,
            options.grid_storage,
            options.rounding,
            options.execution,
        )
    }
    .map_err(|e| e.to_string())?;
    if let Some(bound) = options.motion_bound {
        tractor
            .set_soil_motion_bound(Some([bound.metres; 3]), bound.max_bytes)
            .map_err(|e| e.to_string())?;
        tractor
            .set_soil_motion_projection_fused(options.motion_fusion)
            .map_err(|e| e.to_string())?;
        tractor.prepare_soil_coupling().map_err(|e| e.to_string())?;
    }
    manifest["live_soil_motion_fusion"] = motion::verified_fusion(
        options.motion_fusion,
        tractor.world().mpm_soil_motion_projection_fused(),
    )?;
    manifest["live_soil_motion_bound"] = motion::verified_live(
        options.motion_bound,
        tractor
            .world()
            .mpm_soil_motion_info()
            .ok_or("missing live motion guard info")?,
    )?;
    let (live_grid, grid_bytes) = tractor
        .world()
        .mpm_soil_grid_storage()
        .ok_or("missing live soil grid")?;
    if live_grid != options.grid_storage {
        return Err("live soil grid does not match requested layout/capacity".into());
    }
    manifest["live_soil_grid"] = grid_storage_json(live_grid);
    manifest["physical_grid_and_metadata_bytes"] = json!(grid_bytes);
    let transfer = tractor
        .world()
        .mpm_soil_transfer_info()
        .ok_or("missing live soil transfer")?;
    let fixed = options.transfer != SoilTractorTransfer::Atomic;
    if transfer.fixed_order != fixed
        || transfer.shared_chunks
            != matches!(options.transfer, SoilTractorTransfer::SharedChunk { .. })
        || transfer.grid_reuse != matches!(options.transfer, SoilTractorTransfer::GridReuse { .. })
        || transfer.particle_ordering != fixed
        || transfer.parallel_sort != fixed
        || transfer.gather_bytes
            != options
                .transfer
                .required_gather_bytes(
                    cfg.particle_count().map_err(|e| e.to_string())?,
                    cfg.grid_size,
                )
                .map_err(|e| e.to_string())?
        || (transfer.ordering_bytes > 0) != fixed
    {
        return Err("live soil transfer does not match requested selection/resources".into());
    }
    manifest["live_soil_transfer"] = json!({"mode":transfer_json(options.transfer)["mode"],
        "shared_chunks":transfer.shared_chunks,
        "grid_reuse":transfer.grid_reuse,
        "particle_ordering":transfer.particle_ordering,"parallel_sort":transfer.parallel_sort,
        "gather_bytes":transfer.gather_bytes,"ordering_bytes":transfer.ordering_bytes});
    write_json(&options.output.join("manifest.json"), &manifest)?;
    if tractor.world().mpm_soil_newton_enabled() != options.newton {
        return Err("live soil solver does not match requested Newton selection".into());
    }
    if tractor.world().solver.gpu_uses_paired_rates() != options.paired_rates {
        return Err("live machine rate encoding does not match the requested benchmark".into());
    }
    if tractor.world().gpu_uses_native_prediction() != options.native_prediction {
        return Err("live native machine predictor does not match request".into());
    }
    manifest["live_native_machine_prediction"] =
        json!(tractor.world().gpu_uses_native_prediction());
    if tractor.world().solver.gpu_execution() != Some(options.execution) {
        return Err("live machine execution does not match the requested benchmark".into());
    }
    if let Some(config) = options.unilateral.filter(|_| !options.paired_rates) {
        if let Some(mu) = options.friction_mu {
            let handles = tractor.handles();
            let coefficients = handles
                .wheels
                .into_iter()
                .chain([handles.tine])
                .map(|body| (body, mu))
                .collect::<Vec<_>>();
            if options.friction_walls {
                tractor.enable_wall_friction_coupling(config, &coefficients)
            } else {
                tractor.enable_friction_coupling(config, &coefficients)
            }
            .map_err(|e| e.to_string())?;
            manifest["friction_body_coefficients"] = json!(coefficients.iter()
                .map(|(body, coefficient)| json!({"body": format!("{body:?}"), "coefficient": coefficient}))
                .collect::<Vec<_>>());
        } else {
            tractor
                .enable_unilateral_coupling(config)
                .map_err(|e| e.to_string())?;
        }
    } else if options.mass_coupling_nodes > 0 {
        tractor
            .enable_mass_coupling_with_boundary_scan(
                options.mass_coupling_nodes,
                options.boundary_scan,
            )
            .map_err(|e| e.to_string())?;
    }
    let construction_ms = started.elapsed().as_secs_f64() * 1000.0;
    let handles = tractor.handles();
    let mut geometry = vec![json!({
        "body": format!("{:?}", handles.chassis), "role": "chassis", "shape": "box",
        "half_extents": molla_solvers::soil_tractor::CHASSIS_HALF_EXTENTS.to_array()
    })];
    for wheel in handles.wheels {
        geometry.push(
            json!({"body": format!("{wheel:?}"), "role": "wheel", "shape": "cylinder-y",
            "radius": molla_solvers::soil_tractor::WHEEL_RADIUS,
            "half_width": molla_solvers::soil_tractor::WHEEL_HALF_WIDTH}),
        );
    }
    geometry.push(
        json!({"body": format!("{:?}", handles.tine), "role": "tine", "shape": "box",
        "half_extents": molla_solvers::soil_tractor::TINE_HALF_EXTENTS.to_array()}),
    );
    write_json(
        &options.output.join("render-geometry.json"),
        &json!({
            "schema": 1, "scope": "analytic fixture shapes in body-local coordinates, not asset meshes", "bodies": geometry
        }),
    )?;
    if options.frame_loads {
        tractor
            .prepare_soil_frame_loads()
            .map_err(|e| e.to_string())?;
    }
    let mut frame_loads = if options.frame_loads {
        Some(File::create(options.output.join("frame-loads.jsonl")).map_err(|e| e.to_string())?)
    } else {
        None
    };
    let mut contact_motion = if options.contact_motion {
        Some(File::create(options.output.join("contact-motion.jsonl")).map_err(|e| e.to_string())?)
    } else {
        None
    };
    let expected_index =
        if options.unilateral.is_some() && options.grid_storage != MpmGridStorage::Dense {
            Some(if cfg!(feature = "parallel-candidate-index") {
                "parallel-stable-radix"
            } else {
                "serial-stable-radix"
            })
        } else {
            None
        };
    let live_index = tractor.world().mpm_soil_candidate_index_backend();
    if live_index != expected_index {
        return Err("live candidate index backend does not match grid/build selection".into());
    }
    manifest["live_candidate_index_backend"] = json!(live_index);
    let live_newton = tractor.world().mpm_soil_newton_backend();
    if json!(live_newton) != manifest["unilateral_config"]["newton_method"] {
        return Err("live Newton backend does not match requested coupling".into());
    }
    manifest["live_soil_newton_backend"] = json!(live_newton);
    let deferred_drift = tractor.world().mpm_soil_defers_native_prediction_drift();
    if deferred_drift
        != (options.native_prediction
            && options.unilateral.is_some()
            && cfg!(feature = "deferred-native-drift"))
    {
        return Err("live native prediction drift schedule mismatch".into());
    }
    manifest["live_deferred_native_prediction_drift"] = json!(deferred_drift);
    let drift_backend = tractor.world().solver.gpu_unrestricted_drift_backend();
    if options.native_prediction
        && cfg!(feature = "native-lazy-geometry-drift")
        && drift_backend != Some("native-geometry-lazy-anchors")
    {
        return Err("live native lazy geometry drift backend mismatch".into());
    }
    manifest["live_machine_unrestricted_drift_backend"] = json!(drift_backend);
    if deferred_drift {
        manifest["machine_prediction_stages"] = json!(4);
        manifest["machine_prediction_backend"] =
            json!("native-controls-bias-gram-solve-deferred-drift");
    }
    manifest["parallel_candidate_index_build"] = json!(cfg!(feature = "parallel-candidate-index"));
    if options.profile_mpm {
        if options.profile_preparation {
            tractor.prepare_soil_frame_profiling_with_preparation(substeps)
        } else {
            tractor.prepare_soil_frame_profiling(substeps)
        }
        .map_err(|e| e.to_string())?;
    }
    let mut profiles = if options.profile_mpm {
        Some(File::create(options.output.join("mpm-profiles.jsonl")).map_err(|e| e.to_string())?)
    } else {
        None
    };
    let initial_energy = machine_energy(&tractor.world().scene)?;
    let initial_axles = axle_angles(&tractor.world().scene, tractor.handles().axles)?;
    let solver = &tractor.world().solver;
    if solver.gpu_execution() != Some(options.execution) {
        return Err("machine execution profile differs from requested backend".into());
    }
    let observed_config = if options.friction_mu.is_some() {
        tractor.world().mpm_soil_friction_config()
    } else {
        tractor.world().mpm_soil_unilateral_config()
    };
    if observed_config != options.unilateral {
        return Err("contact configuration differs from requested backend".into());
    }
    let observed_friction = tractor.world().mpm_soil_friction_coefficients();
    let observed_walls = tractor.world().mpm_soil_friction_fixed_walls();
    if observed_walls != options.friction_mu.map(|_| options.friction_walls) {
        return Err("friction wall selection differs from requested backend".into());
    }
    let coefficients_match = match (options.friction_mu, &observed_friction) {
        (None, None) => true,
        (Some(mu), Some(values)) => {
            values.len() == 5
                && handles
                    .wheels
                    .into_iter()
                    .chain([handles.tine])
                    .all(|body| {
                        values
                            .iter()
                            .filter(|(handle, value)| *handle == body && *value == mu)
                            .count()
                            == 1
                    })
        }
        _ => false,
    };
    if !coefficients_match {
        return Err("friction coefficients differ from requested body assignment".into());
    }
    write_json(&options.output.join("manifest.json"), &manifest)?;
    write_json(
        &options.output.join("solver.json"),
        &json!({
            "joint_damping": solver.joint_damping,
            "stable_motor_feedback": solver.stable_motor_feedback,
            "rigorous_d_solve": solver.rigorous_d_solve,
            "rigorous_distance": solver.rigorous_distance,
            "regularized_friction": solver.regularized_friction,
            "momentum_transport": solver.momentum_transport,
            "contact_impacts": solver.contact_impacts,
            "construction_ms": construction_ms,
            "unilateral_config": unilateral_config_json(observed_config, options.friction_mu, options.friction_walls, options.newton, protected),
            "friction_fixed_walls": observed_walls,
            "friction_body_coefficients": observed_friction.map(|values| values.into_iter()
                .map(|(body, coefficient)| json!({"body": format!("{body:?}"), "coefficient": coefficient})).collect::<Vec<_>>()),
            "boundary_scan_backend": tractor.world().mpm_soil_boundary_scan().map(SoilBoundaryScan::label)
        }),
    )?;
    let mut csv = File::create(options.output.join("frames.csv")).map_err(|e| e.to_string())?;
    let mut states =
        File::create(options.output.join("machine-states.jsonl")).map_err(|e| e.to_string())?;
    writeln!(csv, "frame,warmup,wall_ms,frame_simulated_seconds,total_simulated_seconds,committed_frames,dispatches,upload_bytes,readback_bytes,allocation_observations,chassis_x,chassis_y,chassis_z,machine_kinetic_j,machine_gravity_potential_j,commanded_motor_work_j,machine_mechanical_change_minus_commanded_work_j")
        .map_err(|e| e.to_string())?;
    let mut times = Vec::new();
    let mut missed = 0;
    let mut totals = [0u64; 4];
    let mut stage_samples: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let mut contact_samples: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let mut preparation_samples: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let mut kernel_samples = Vec::new();
    let mut span_samples = Vec::new();
    let mut host_samples: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let mut machine_samples: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let mut profile_readback_bytes = 0u64;
    let mut load_readback_bytes = 0u64;
    let mut motion_readback_bytes = 0u64;
    let mut motion_wall_ms = Vec::new();
    let mut visual_records = Vec::new();
    let mut final_visual_positions = None;
    if !visual_frames.is_empty() {
        let initial = visual::machine(&tractor.world().scene, 0, 0.0)?;
        let (record, _) = visual::capture(&mut tractor, &options.output, &initial)?;
        visual_records.push(record);
    }
    for frame in 0..options.warmup + options.frames {
        let allocations = DeviceScope::start().map_err(|e| e.to_string())?;
        let transfers = TransferScope::start().map_err(|e| e.to_string())?;
        let dispatches = DispatchScope::start().map_err(|e| e.to_string())?;
        let start = Instant::now();
        let result = tractor.step(DT, DRIVE_TORQUE_NM);
        let wall_ms = start.elapsed().as_secs_f64() * 1000.0;
        let allocations = allocations.finish();
        let transfers = transfers.finish();
        let dispatches = dispatches.finish();
        if let Err(error) = result {
            if engine.check_device().is_err() {
                let _ = write_json(
                    &options.output.join("result.json"),
                    &json!({
                        "status": "failed", "acceptance": false, "frame": frame,
                        "error": error.to_string(), "device_quarantined": true
                    }),
                );
                eprintln!("terminal GPU failure: {error}");
                std::process::exit(76);
            }
            return Err(format!("frame {frame}: {error}"));
        }
        if allocations.invalid || transfers.overflowed || dispatches.overflowed {
            return Err("resource counters invalid or overflowed".into());
        }
        let telemetry = tractor.telemetry().map_err(|e| e.to_string())?;
        let stats = tractor
            .world()
            .mpm_soil_stats()
            .ok_or("soil stats missing")?;
        if stats.faulted
            || stats.rolled_back_frames != 0
            || stats.committed_frames != frame as u64 + 1
            || stats.live_bindings != 5
            || !telemetry.chassis_pose.position.is_finite()
            || (telemetry.simulated_seconds - (frame + 1) as f64 * DT).abs() > 1e-10
        {
            return Err(format!(
                "frame {frame}: completion/physical state invariant failed"
            ));
        }
        for (body, _) in tractor.world().scene.bodies() {
            let velocity = tractor
                .world()
                .scene
                .body_velocity(body)
                .map_err(|e| e.to_string())?;
            if !velocity.linear.is_finite() || !velocity.angular.is_finite() {
                return Err(format!("frame {frame}: nonfinite body velocity"));
            }
        }
        let warmup = frame < options.warmup;
        let energy = machine_energy(&tractor.world().scene)?;
        if let Some(file) = &mut contact_motion {
            let transfers = TransferScope::start().map_err(|e| e.to_string())?;
            let start = Instant::now();
            let sample = tractor
                .read_soil_contact_motion()
                .map_err(|e| e.to_string())?;
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;
            let transfers = transfers.finish();
            if transfers.overflowed
                || transfers.upload_bytes != 0
                || transfers.readback_bytes != 240
                || sample.committed_frame != stats.committed_frames
                || sample.bodies.len() != 5
            {
                return Err("contact motion identity/counter mismatch".into());
            }
            if !warmup {
                motion_readback_bytes += transfers.readback_bytes;
                motion_wall_ms.push(elapsed);
            }
            let bodies:Vec<_>=sample.bodies.iter().map(|body| json!({
                "body":format!("{:?}",body.body),"supporting_nodes":body.motion.supporting_nodes,
                "normal_impulse_n_s":body.motion.normal_impulse,"soil_velocity_world_m_s":body.motion.soil_velocity,
                "tangential_slip_velocity_world_m_s":body.motion.tangential_slip_velocity,
                "mean_tangential_slip_speed_m_s":body.motion.mean_tangential_slip_speed,
            })).collect();
            writeln!(file,"{}",json!({"schema":1,"frame":sample.committed_frame,"warmup":warmup,"substep_dt_s":sample.substep_dt,
                "observation_wall_ms":elapsed,"readback_bytes":transfers.readback_bytes,"bodies":bodies})).map_err(|e| e.to_string())?;
        }
        if let Some(file) = &mut frame_loads {
            let transfers = TransferScope::start().map_err(|e| e.to_string())?;
            let loads = tractor.read_soil_frame_loads().map_err(|e| e.to_string())?;
            let transfers = transfers.finish();
            if transfers.overflowed
                || loads.committed_frame != stats.committed_frames
                || loads.bodies.len() != 5
                || (loads.simulated_seconds
                    - f64::from((DT / substeps as f64) as f32) * substeps as f64)
                    .abs()
                    > 1e-12
            {
                return Err("frame load identity/duration/counter mismatch".into());
            }
            if !warmup {
                load_readback_bytes += transfers.readback_bytes;
            }
            let bodies: Vec<_> = loads.bodies.iter().map(|b| json!({
                "body": format!("{:?}", b.body), "linear_impulse_n_s": b.linear,
                "com_angular_impulse_nm_s": b.angular,
                "world_origin_angular_impulse_nm_s": b.world_origin_angular,
                "mean_force_n": b.linear.map(|v| f64::from(v) / loads.simulated_seconds),
                "mean_com_torque_nm": b.angular.map(|v| f64::from(v) / loads.simulated_seconds),
            })).collect();
            writeln!(file, "{}", json!({"frame": frame, "warmup": warmup,
                "committed_frame": loads.committed_frame, "simulated_seconds": loads.simulated_seconds,
                "bodies": bodies})).map_err(|e| e.to_string())?;
        }
        let motor_work = commanded_motor_work(
            initial_axles,
            axle_angles(&tractor.world().scene, tractor.handles().axles)?,
            DRIVE_TORQUE_NM,
        );
        if let Some(profiles) = &mut profiles {
            let captured = tractor
                .read_soil_frame_profile()
                .map_err(|e| e.to_string())?
                .ok_or("missing committed MPM frame profile")?;
            let profile = captured.profile;
            let stages_per_step = if options.mass_coupling_nodes > 0 || options.unilateral.is_some()
            {
                3
            } else {
                2
            };
            if captured.machine_stages.len() != 1 + substeps * stages_per_step
                || captured
                    .machine_stages
                    .iter()
                    .any(|s| !s.elapsed_ms.is_finite() || s.elapsed_ms <= 0.0)
                || captured.machine_stages[0].stage != "refresh"
                || captured.machine_stages[0].substep.is_some()
                || captured.machine_stages[1..]
                    .chunks_exact(stages_per_step)
                    .enumerate()
                    .any(|(i, pair)| {
                        pair[0].stage != "dynamics"
                            || pair[stages_per_step - 1].stage != "momentum"
                            || (stages_per_step == 3 && pair[1].stage != "soil-corrected-drift")
                            || pair.iter().any(|s| {
                                s.substep != Some(i)
                                    || !s.elapsed_ms.is_finite()
                                    || s.elapsed_ms <= 0.0
                            })
                    })
            {
                return Err("machine profile pass identity or duration mismatch".into());
            }
            if captured.committed_frame != stats.committed_frames {
                return Err("MPM frame profile committed-frame mismatch".into());
            }
            profiling::validate(
                &profile,
                substeps,
                options.grid_storage,
                options.mass_coupling_nodes > 0 || options.unilateral.is_some(),
                stats.committed_frames == 1,
                options.transfer != SoilTractorTransfer::Atomic,
                options.motion_fusion,
            )?;
            let mut sums: BTreeMap<String, f64> = BTreeMap::new();
            if profile.contact_stages.len() != if options.newton { substeps * 3 } else { 0 } {
                return Err("contact profile child stage count mismatch".into());
            }
            if profile.preparation_details_enabled != options.profile_preparation
                || profile.preparation_stages.len()
                    != if options.profile_preparation {
                        substeps * 8
                    } else {
                        0
                    }
                || profile.readback_bytes
                    != 16
                        * (profile.stages.len()
                            + profile.contact_stages.len()
                            + profile.preparation_stages.len()) as u64
            {
                return Err("preparation profile identity, count or byte mismatch".into());
            }
            let mut preparation_sums: BTreeMap<String, f64> = BTreeMap::new();
            let preparation_rows: Vec<_> = profile.preparation_stages.iter().map(|stage| {
                *preparation_sums.entry(stage.stage.name().to_owned()).or_default() += stage.elapsed_ms;
                json!({"stage": stage.stage.name(), "substep": stage.substep, "gpu_ms": stage.elapsed_ms, "start_ms": stage.start_ms})
            }).collect();
            let mut contact_sums: BTreeMap<String, f64> = BTreeMap::new();
            let contact_rows: Vec<_> = profile.contact_stages.iter().map(|stage| {
                *contact_sums.entry(stage.stage.name().to_owned()).or_default() += stage.elapsed_ms;
                json!({"stage": stage.stage.name(), "substep": stage.substep, "gpu_ms": stage.elapsed_ms, "start_ms": stage.start_ms})
            }).collect();
            let rows: Vec<_> = profile.stages.iter().map(|stage| {
                *sums.entry(stage.stage.name().to_owned()).or_default() += stage.elapsed_ms;
                json!({"stage": stage.stage.name(), "substep": stage.substep, "gpu_ms": stage.elapsed_ms,
                    "start_ms": stage.start_ms})
            }).collect();
            let kernel_ms: f64 = sums.values().sum();
            let mut machine_sums: BTreeMap<String, f64> = BTreeMap::new();
            let machine_rows: Vec<_> = captured.machine_stages.iter().map(|s| {
                *machine_sums.entry(s.stage.to_owned()).or_default() += s.elapsed_ms;
                json!({"stage": s.stage, "substep": s.substep, "gpu_ms": s.elapsed_ms, "start_ms": s.start_ms})
            }).collect();
            let machine_readback_bytes = captured.machine_profile_readback_bytes;
            let span_ms = profile
                .stages
                .iter()
                .map(|s| s.start_ms + s.elapsed_ms)
                .fold(0.0, f64::max);
            writeln!(
                profiles,
                "{}",
                json!({
                    "committed_frame": captured.committed_frame, "warmup": warmup,
                    "requested_substeps": profile.requested_substeps,
                    "active_grid": profile.active_grid, "compact_grid": profile.compact_grid,
                    "requested_simulated_seconds": profile.requested_simulated_seconds,
                    "readback_bytes": profile.readback_bytes, "mpm_kernel_gpu_ms": kernel_ms,
                    "timestamp_period_ns": profile.timestamp_period_ns,
                    "mpm_capture_span_ms": span_ms,
                    "host_stage_ms": captured.host_stage_ms.into_iter().collect::<BTreeMap<_, _>>(),
                    "machine_stages": machine_rows, "machine_stage_gpu_ms": machine_sums,
                    "machine_profile_readback_bytes": machine_readback_bytes,
                    "newton_work": captured.newton_work.iter().map(|w|json!({"substep":w.substep,"faults":w.faults,"steps":w.steps,"linear_iterations":w.linear_iterations,"trials":w.trials})).collect::<Vec<_>>(),
                    "newton_work_scope":"recovery counts; linear iterations include Krylov iterations and condensed solves; report bytes already included in machine_profile_readback_bytes",
                    "stages": rows, "by_stage_gpu_ms": sums,
                    "contact_stages": contact_rows, "contact_stage_gpu_ms": contact_sums,
                    "preparation_details_enabled": profile.preparation_details_enabled,
                    "preparation_stages": preparation_rows, "preparation_stage_gpu_ms": preparation_sums,
                    "preparation_stage_scope": "nested inside contact-preparation; do not add to parent totals",
                    "contact_stage_scope": "nested inside contact-projection; do not add to MPM totals"
                })
            )
            .map_err(|e| e.to_string())?;
            profiles.flush().map_err(|e| e.to_string())?;
            if !warmup {
                kernel_samples.push(kernel_ms);
                span_samples.push(span_ms);
                for (stage, ms) in captured.host_stage_ms {
                    host_samples.entry(stage.to_owned()).or_default().push(ms);
                }
                for (stage, ms) in machine_sums {
                    machine_samples.entry(stage).or_default().push(ms);
                }
                for (stage, sum) in sums {
                    stage_samples.entry(stage).or_default().push(sum);
                }
                for (stage, sum) in contact_sums {
                    contact_samples.entry(stage).or_default().push(sum);
                }
                for (stage, sum) in preparation_sums {
                    preparation_samples.entry(stage).or_default().push(sum);
                }
                profile_readback_bytes += profile.readback_bytes + machine_readback_bytes;
            }
        }
        if !warmup {
            times.push(wall_ms);
            missed += usize::from(wall_ms > DT * 1000.0);
            for (total, count) in totals.iter_mut().zip([
                dispatches.encoded_dispatches,
                transfers.upload_bytes,
                transfers.readback_bytes,
                allocations.observations,
            ]) {
                *total = total
                    .checked_add(count)
                    .ok_or("aggregate counter overflow")?;
            }
        }
        let p = telemetry.chassis_pose.position;
        writeln!(
            csv,
            "{frame},{warmup},{wall_ms:.9},{DT:.15},{:.15},{},{},{},{},{},{:.12},{:.12},{:.12},{:.9},{:.9},{motor_work:.9},{:.9}",
            telemetry.simulated_seconds,
            stats.committed_frames,
            dispatches.encoded_dispatches,
            transfers.upload_bytes,
            transfers.readback_bytes,
            allocations.observations,
            p.x,
            p.y,
            p.z,
            energy[0],
            energy[1],
            energy.iter().sum::<f64>() - initial_energy.iter().sum::<f64>() - motor_work
        )
        .map_err(|e| e.to_string())?;
        csv.flush().map_err(|e| e.to_string())?;
        let state = visual::machine(
            &tractor.world().scene,
            stats.committed_frames,
            telemetry.simulated_seconds,
        )?;
        writeln!(states, "{}", state).map_err(|e| e.to_string())?;
        states.flush().map_err(|e| e.to_string())?;
        if visual_frames.contains(&(frame + 1)) {
            let (record, positions) = visual::capture(&mut tractor, &options.output, &state)?;
            visual_records.push(record);
            if frame + 1 == options.warmup + options.frames {
                final_visual_positions = Some(positions);
            }
        }
        eprintln!("frame={frame} warmup={warmup} completed_wall_ms={wall_ms:.3}");
    }
    let positions = match final_visual_positions {
        Some(positions) => positions,
        None => tractor.read_soil_positions().map_err(|e| e.to_string())?,
    };
    if positions.len() != 50_000 || positions.iter().flatten().any(|v| !v.is_finite()) {
        return Err("final particle count or finite-position check failed".into());
    }
    let bytes: Vec<_> = positions
        .iter()
        .flatten()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    fs::write(options.output.join("final-particles.bin"), bytes).map_err(|e| e.to_string())?;
    let final_soil_state = if options.final_soil_state {
        history::capture(
            &mut tractor,
            &options.output,
            (options.warmup + options.frames) as u64,
            50_000,
        )?
    } else {
        Value::Null
    };
    let loads = tractor.read_soil_loads().map_err(|e| e.to_string())?;
    if loads.bodies.len() != 5
        || loads
            .bodies
            .iter()
            .any(|b| b.force.iter().chain(&b.torque).any(|v| !v.is_finite()))
    {
        return Err("final wrench observation invalid".into());
    }
    let load_values: Vec<_> = loads
        .bodies
        .iter()
        .map(|b| {
            json!({
                "body": format!("{:?}", b.body), "force_n": b.force, "com_torque_nm": b.torque
            })
        })
        .collect();
    let final_energy = machine_energy(&tractor.world().scene)?;
    let final_axles = axle_angles(&tractor.world().scene, tractor.handles().axles)?;
    let motor_work = commanded_motor_work(initial_axles, final_axles, DRIVE_TORQUE_NM);
    let mut report = json!({
        "schema": 1, "status": "diagnostic_complete", "acceptance": false,
        "scope": manifest["scope"], "physical_calibration": "uncalibrated",
        "samples": times.len(), "frame_simulated_seconds": DT,
        "measured_simulated_seconds": times.len() as f64 * DT,
        "total_simulated_seconds": tractor.telemetry().map_err(|e| e.to_string())?.simulated_seconds,
        "completed_wall_ms_sum": times.iter().sum::<f64>(),
        "wall_over_simulated_ratio": times.iter().sum::<f64>() / 1000.0 / (times.len() as f64 * DT),
        "wall_ms": {"median": percentile(&times, 0.5), "p95": percentile(&times, 0.95),
            "p99": percentile(&times, 0.99), "max": percentile(&times, 1.0)},
        "missed_60hz_deadlines": missed, "soil_only_ms": null,
        "dispatches": totals[0], "upload_bytes": totals[1], "readback_bytes": totals[2],
        "allocation_observations": totals[3], "construction_ms": construction_ms,
        "mpm_kernel_gpu_ms": timing_summary(&kernel_samples),
        "mpm_capture_span_ms": timing_summary(&span_samples),
        "host_stage_ms": host_samples.iter().map(|(k, v)| (k.clone(), timing_summary(v))).collect::<BTreeMap<_, _>>(),
        "machine_stage_gpu_ms": machine_samples.iter().map(|(k, v)| (k.clone(), timing_summary(v))).collect::<BTreeMap<_, _>>(),
        "mpm_stage_gpu_ms": stage_samples.iter().map(|(k, v)| (k.clone(), timing_summary(v))).collect::<BTreeMap<_, _>>(),
        "profile_readback_bytes_outside_timing": profile_readback_bytes,
        "initial_machine_energy_j": initial_energy,
        "final_machine_energy_j": final_energy,
        "commanded_motor_work_j": motor_work,
        "initial_axle_angles_rad": initial_axles, "final_axle_angles_rad": final_axles,
        "machine_mechanical_change_minus_commanded_work_j": final_energy.iter().sum::<f64>() - initial_energy.iter().sum::<f64>() - motor_work,
        "work_scope": "constant commanded axle effort times unwrapped reported coordinate change; machine-only energy difference excludes soil energy and contact dissipation, not a conservation verdict",
        "final_particles": positions.len(), "final_committed_frame": loads.committed_frame,
        "final_soil_state": final_soil_state,
        "final_load_substep_seconds": loads.substep_dt, "last_substep_loads": load_values,
        "remaining_acceptance": ["soil-only wall budget", "required terrain export", "rendering",
            "physical calibration", "frozen acceptance sampling", "long-horizon accuracy"]
    });
    report["frame_load_observation_enabled"] = json!(options.frame_loads);
    report["preparation_profile_enabled"] = json!(options.profile_preparation);
    report["preparation_stage_gpu_ms"] = json!(
        preparation_samples
            .iter()
            .map(|(k, v)| (k.clone(), timing_summary(v)))
            .collect::<BTreeMap<_, _>>()
    );
    report["preparation_stage_scope"] =
        json!("nested inside contact-preparation; do not add to parent totals");
    report["boundary_scan_backend"] = json!(
        tractor
            .world()
            .mpm_soil_boundary_scan()
            .map(SoilBoundaryScan::label)
    );
    report["contact_stage_gpu_ms"] = json!(
        contact_samples
            .iter()
            .map(|(k, v)| (k.clone(), timing_summary(v)))
            .collect::<BTreeMap<_, _>>()
    );
    report["contact_stage_scope"] =
        json!("nested inside contact-projection; do not add to MPM totals");
    report["load_readback_bytes_outside_timing"] = json!(load_readback_bytes);
    report["contact_motion_observation_enabled"] = json!(options.contact_motion);
    report["contact_motion_readback_bytes_outside_timing"] = json!(motion_readback_bytes);
    report["contact_motion_wall_ms_outside_timing"] = if motion_wall_ms.is_empty() {
        Value::Null
    } else {
        timing_summary(&motion_wall_ms)
    };
    report["visual_capture_frames"] = json!(visual_frames);
    report["visual_capture_wall_ms_outside_timing"] = json!(
        visual_records
            .iter()
            .map(|record| record["capture_wall_ms_outside_timing"].as_f64().unwrap())
            .sum::<f64>()
    );
    report["visual_readback_bytes_outside_timing"] = json!(
        visual_records
            .iter()
            .map(|record| record["readback_bytes_outside_timing"].as_u64().unwrap())
            .sum::<u64>()
    );
    if !visual_records.is_empty() {
        write_json(
            &options.output.join("visual-captures.json"),
            &json!({"schema": 1, "scope": "time-aligned display snapshots, not restart checkpoints or performance acceptance", "frames": visual_records}),
        )?;
    }
    write_json(&options.output.join("result.json"), &report)?;
    println!("{report}");
    Ok(())
}

fn timing_summary(values: &[f64]) -> Value {
    if values.is_empty() {
        return Value::Null;
    }
    json!({"samples": values.len(), "median": percentile(values, 0.5),
        "p95": percentile(values, 0.95), "p99": percentile(values, 0.99),
        "max": percentile(values, 1.0)})
}

fn machine_energy(scene: &molla_sim::runtime::RigidScene) -> Result<[f64; 2], String> {
    let mut energy = [0.0, 0.0];
    for (body, _) in scene.bodies() {
        let mass = scene
            .body_mass_properties(body)
            .map_err(|e| e.to_string())?;
        let center = scene.body_world_com(body).map_err(|e| e.to_string())?;
        let linear = scene
            .body_velocity_at_point(body, center)
            .map_err(|e| e.to_string())?;
        let pose = scene.body_pose(body).map_err(|e| e.to_string())?;
        let angular = pose.rotation.inverse()
            * scene
                .body_velocity(body)
                .map_err(|e| e.to_string())?
                .angular;
        energy[0] +=
            0.5 * (mass.mass * linear.length_squared() + angular.dot(mass.inertia * angular));
        energy[1] += 9.81 * mass.mass * center.y;
    }
    if energy.iter().any(|e| !e.is_finite()) {
        return Err("nonfinite machine energy".into());
    }
    Ok(energy)
}

fn main() -> std::process::ExitCode {
    let options = match options() {
        Ok(options) => options,
        Err(error) => {
            eprintln!("{error}");
            return std::process::ExitCode::from(2);
        }
    };
    if let Err(error) = fs::create_dir(&options.output) {
        eprintln!("refusing to reuse or unable to create output directory: {error}");
        return std::process::ExitCode::from(2);
    }
    match run(&options) {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            let _ = write_json(
                &options.output.join("result.json"),
                &json!({
                    "status": "failed", "acceptance": false, "error": error
                }),
            );
            eprintln!("{error}");
            std::process::ExitCode::from(1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn friction_is_explicit_finite_and_requires_contact_configuration() {
        let parse = |args: &[&str]| {
            parse_options(
                ["--output", "unused"]
                    .into_iter()
                    .chain(args.iter().copied())
                    .map(str::to_owned),
            )
        };
        assert!(parse(&[]).unwrap().friction_mu.is_none());
        for value in ["0", "0.3", "1", "2"] {
            let options = parse(&["--unilateral-nodes", "4096", "--friction-mu", value]).unwrap();
            assert_eq!(options.friction_mu, Some(value.parse().unwrap()));
            let recorded = unilateral_config_json(
                options.unilateral,
                options.friction_mu,
                options.friction_walls,
                options.newton,
                false,
            );
            assert!(
                recorded["scope"]
                    .as_str()
                    .unwrap()
                    .contains("fixed-wall contact rejects")
            );
            assert!(!recorded["scope"].as_str().unwrap().contains("frictionless"));
        }
        for args in [
            vec!["--friction-mu", "0.3"],
            vec!["--friction-mu"],
            vec!["--unilateral-nodes", "4096", "--friction-mu", "NaN"],
            vec!["--unilateral-nodes", "4096", "--friction-mu", "inf"],
            vec!["--unilateral-nodes", "4096", "--friction-mu", "-0.1"],
            vec!["--unilateral-nodes", "4096", "--friction-mu", "bad"],
            vec![
                "--unilateral-nodes",
                "4096",
                "--friction-mu",
                "0",
                "--friction-mu",
                "0",
            ],
            vec![
                "--unilateral-nodes",
                "4096",
                "--friction-mu",
                "0.3",
                "--mass-coupling-nodes",
                "64",
            ],
        ] {
            assert!(parse(&args).is_err(), "{args:?}");
        }
    }

    #[test]
    fn visual_capture_is_opt_in_bounded_and_strictly_parsed() {
        let parse = |args: &[&str]| parse_options(args.iter().map(|arg| (*arg).to_owned()));
        assert_eq!(parse(&["--output", "unused"]).unwrap().capture_every, 0);
        assert_eq!(
            parse(&["--output", "unused", "--capture-every", "11"])
                .unwrap()
                .capture_every,
            11
        );
        for args in [
            vec!["--output", "unused", "--capture-every"],
            vec!["--output", "unused", "--capture-every", "-1"],
            vec!["--output", "unused", "--capture-every", "331"],
            vec![
                "--output",
                "unused",
                "--capture-every",
                "1",
                "--frames",
                "300",
            ],
            vec![
                "--output",
                "unused",
                "--capture-every",
                "1",
                "--capture-every",
                "2",
            ],
        ] {
            assert!(parse(&args).is_err());
        }
    }

    #[test]
    fn unilateral_selection_is_explicit_exclusive_and_recorded() {
        let parse = |args: &[&str]| {
            parse_options(
                [vec!["--output", "unused"], args.to_vec()]
                    .concat()
                    .into_iter()
                    .map(str::to_owned),
            )
        };
        assert!(parse(&[]).unwrap().unilateral.is_none());
        assert_eq!(
            unilateral_config_json(None, None, false, false, false),
            Value::Null
        );
        let cfg = parse(&["--unilateral-nodes", "4096"])
            .unwrap()
            .unilateral
            .unwrap();
        assert_eq!(cfg.node_capacity, 4096);
        assert_eq!(cfg.search_distance, 0.05);
        assert_eq!(cfg.max_iterations, 64);
        let cfg = parse(&[
            "--unilateral-nodes",
            "16384",
            "--unilateral-margin",
            "0.1",
            "--unilateral-iterations",
            "32",
        ])
        .unwrap()
        .unilateral
        .unwrap();
        let recorded = unilateral_config_json(Some(cfg), None, false, false, false);
        assert_eq!(recorded["node_capacity_total"], 16384);
        assert_eq!(recorded["search_distance_m"], json!(0.1f32));
        assert_eq!(recorded["max_iterations"], 32);
        for args in [
            vec!["--unilateral-nodes"],
            vec!["--unilateral-nodes", "0"],
            vec!["--unilateral-nodes", "16385"],
            vec![
                "--unilateral-nodes",
                "4096",
                "--mass-coupling-nodes",
                "4096",
            ],
            vec!["--unilateral-nodes", "4096", "--boundary-scan", "cached"],
            vec!["--unilateral-nodes", "4096", "--unilateral-nodes", "4096"],
            vec!["--unilateral-margin", "0.05"],
            vec!["--unilateral-iterations", "64"],
            vec!["--unilateral-nodes", "4096", "--unilateral-margin", "NaN"],
            vec!["--unilateral-nodes", "4096", "--unilateral-margin", "inf"],
            vec!["--unilateral-nodes", "4096", "--unilateral-margin", "-0.1"],
            vec!["--unilateral-nodes", "4096", "--unilateral-iterations", "0"],
            vec![
                "--unilateral-nodes",
                "4096",
                "--unilateral-iterations",
                "65",
            ],
            vec![
                "--unilateral-nodes",
                "4096",
                "--unilateral-margin",
                "0.1",
                "--unilateral-margin",
                "0.2",
            ],
            vec![
                "--unilateral-nodes",
                "4096",
                "--unilateral-iterations",
                "32",
                "--unilateral-iterations",
                "64",
            ],
        ] {
            assert!(parse(&args).is_err(), "accepted {args:?}");
        }
    }

    #[test]
    fn boundary_cache_requires_coupling_and_explicit_valid_selection() {
        let parse = |args: &[&str]| parse_options(args.iter().map(|arg| (*arg).to_owned()));
        assert_eq!(
            parse(&["--output", "unused"]).unwrap().boundary_scan,
            SoilBoundaryScan::Full
        );
        assert_eq!(
            parse(&[
                "--output",
                "unused",
                "--mass-coupling-nodes",
                "4096",
                "--boundary-scan",
                "cached"
            ])
            .unwrap()
            .boundary_scan,
            SoilBoundaryScan::ExperimentalCached
        );
        for args in [
            vec!["--output", "unused", "--boundary-scan", "cached"],
            vec!["--output", "unused", "--boundary-scan", "invalid"],
            vec!["--output", "unused", "--boundary-scan"],
            vec![
                "--output",
                "unused",
                "--boundary-scan",
                "full",
                "--boundary-scan",
                "cached",
            ],
        ] {
            assert!(parse(&args).is_err());
        }
    }

    #[test]
    fn rounding_selection_is_explicit_and_rejects_invalid_options() {
        let parse = |args: &[&str]| parse_options(args.iter().map(|arg| (*arg).to_owned()));
        assert_eq!(
            parse(&["--output", "unused"]).unwrap().rounding,
            molla_kernels::FeatherstoneRounding::Storage
        );
        for (name, expected) in [
            ("storage", molla_kernels::FeatherstoneRounding::Storage),
            (
                "shared",
                molla_kernels::FeatherstoneRounding::ExperimentalShared,
            ),
        ] {
            assert_eq!(
                parse(&["--output", "unused", "--machine-rounding", name])
                    .unwrap()
                    .rounding,
                expected
            );
        }
        assert!(parse(&["--output", "unused", "--machine-rounding", "native"]).is_err());
        assert!(parse(&["--output", "unused", "--machine-rounding"]).is_err());
        assert!(
            parse(&[
                "--output",
                "unused",
                "--machine-rounding",
                "shared",
                "--machine-rounding",
                "storage"
            ])
            .is_err()
        );
    }

    #[test]
    fn execution_selection_is_separate_default_generic_and_strict() {
        use molla_kernels::FeatherstoneExecution;
        let parse = |args: &[&str]| parse_options(args.iter().map(|arg| (*arg).to_owned()));
        assert_eq!(
            parse(&["--output", "unused"]).unwrap().execution,
            FeatherstoneExecution::Generic
        );
        for (name, expected) in [
            ("generic", FeatherstoneExecution::Generic),
            ("single-lane", FeatherstoneExecution::ExperimentalSingleLane),
            ("joint-types", FeatherstoneExecution::ExperimentalJointTypes),
            (
                "scratch",
                FeatherstoneExecution::ExperimentalWorkgroupScratch,
            ),
            (
                "constraint-scratch",
                FeatherstoneExecution::ExperimentalConstraintScratch,
            ),
        ] {
            let options = parse(&["--output", "unused", "--machine-execution", name]).unwrap();
            assert_eq!(options.execution, expected);
            assert_eq!(
                options.rounding,
                molla_kernels::FeatherstoneRounding::Storage
            );
        }
        for args in [
            vec!["--machine-execution"],
            vec!["--machine-execution", "shared"],
            vec![
                "--machine-execution",
                "scratch",
                "--machine-execution",
                "generic",
            ],
        ] {
            assert!(parse(&[vec!["--output", "unused"], args].concat()).is_err());
        }
    }

    #[test]
    fn nearest_rank_keeps_tail_samples() {
        let values = [4.0, 1.0, 3.0, 100.0, 2.0];
        assert_eq!(percentile(&values, 0.5), 3.0);
        assert_eq!(percentile(&values, 0.95), 100.0);
        assert_eq!(percentile(&values, 0.99), 100.0);
        assert_eq!(percentile(&values, 1.0), 100.0);
    }

    #[test]
    fn motor_work_uses_unwrapped_axles_and_drive_sign() {
        let (mut scene, handles) = molla_solvers::soil_tractor::build_soil_tractor_scene().unwrap();
        let initial = axle_angles(&scene, handles.axles).unwrap();
        assert_eq!(initial, [0.0; 4]);
        for (handle, angle) in handles.axles.into_iter().zip([1.0, -2.0, 3.0, -4.0]) {
            let joint = scene.joint_index(handle).unwrap().index();
            let offset = scene.model().joint_coord_offset.host().unwrap()[joint] as usize;
            scene.state_mut().joint_q.host_mut().unwrap()[offset] = angle;
        }
        assert_eq!(
            commanded_motor_work(
                initial,
                axle_angles(&scene, handles.axles).unwrap(),
                DRIVE_TORQUE_NM
            ),
            600.0
        );
        let turns = [-10.0 * std::f64::consts::TAU; 4];
        assert!(
            (commanded_motor_work(initial, turns, DRIVE_TORQUE_NM)
                - 12000.0 * std::f64::consts::TAU)
                .abs()
                < 1e-8
        );
        assert_eq!(
            commanded_motor_work(turns, initial, DRIVE_TORQUE_NM),
            -commanded_motor_work(initial, turns, DRIVE_TORQUE_NM)
        );
    }

    #[test]
    fn machine_energy_uses_mass_com_and_world_velocity() {
        let (mut scene, _) = molla_solvers::soil_tractor::build_soil_tractor_scene().unwrap();
        let rest = machine_energy(&scene).unwrap();
        assert_eq!(rest[0], 0.0);
        assert!((rest[1] - 725.6 * 9.81).abs() < 1e-8);
        let bodies: Vec<_> = scene.bodies().map(|(body, _)| body).collect();
        for body in bodies {
            scene
                .set_body_velocity(
                    body,
                    molla_math::SpatialVector::new(
                        molla_math::Vec3::X * 2.0,
                        molla_math::Vec3::ZERO,
                    ),
                )
                .unwrap();
        }
        let moving = machine_energy(&scene).unwrap();
        assert!((moving[0] - 1720.0).abs() < 1e-8);
        assert_eq!(rest[1], moving[1]);
        assert_eq!(timing_summary(&[]), Value::Null);
    }
}
