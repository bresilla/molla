#[path = "soil_support/frontier.rs"]
mod frontier;
#[path = "soil_support/results.rs"]
mod results;
#[path = "soil_support/smoke.rs"]
mod smoke;
#[cfg(feature = "gpu")]
#[path = "soil_support/smoke_counters.rs"]
mod smoke_counters;
#[cfg(feature = "gpu")]
#[path = "soil_support/smoke_dem_gpu.rs"]
mod smoke_dem_gpu;
#[cfg(feature = "gpu")]
#[path = "soil_support/smoke_gpu.rs"]
mod smoke_gpu;

use serde_json::json;
use std::path::PathBuf;

fn run() -> Result<u8, String> {
    let mut args = std::env::args().skip(1);
    let command = args.next();
    if command.as_deref() == Some("frontier") {
        return frontier::run(args);
    }
    if command.as_deref() == Some("smoke") {
        let backend = smoke::parse_backend(args)?;
        #[cfg(feature = "gpu")]
        {
            match backend {
                smoke::Backend::Mpm => smoke_gpu::run(),
                smoke::Backend::Dem => smoke_dem_gpu::run(),
            }
            .map_err(|e| e.to_string())?;
            return Ok(0);
        }
        #[cfg(not(feature = "gpu"))]
        return Err("smoke requires --features gpu or resource-counters".into());
    }
    if command.as_deref() != Some("check") {
        return Err(
            "usage: soil_benchmark smoke [--backend mpm|dem] | check --input DIR [--baseline DIR | --instrumentation-baseline DIR] [--output DIR] | frontier --reference DATA_DIR --input DATA_DIR [--input DATA_DIR] --output NEW_DIR"
                .into(),
        );
    }
    let mut input = None;
    let mut output = None;
    let mut baseline = None;
    let mut instrumentation_baseline = None;
    while let Some(arg) = args.next() {
        let slot = match arg.as_str() {
            "--input" => &mut input,
            "--output" => &mut output,
            "--baseline" => &mut baseline,
            "--instrumentation-baseline" => &mut instrumentation_baseline,
            _ => return Err(format!("unknown option {arg}")),
        };
        if slot.is_some() {
            return Err(format!("duplicate option {arg}"));
        }
        *slot = Some(PathBuf::from(
            args.next()
                .ok_or_else(|| format!("missing value for {arg}"))?,
        ));
    }
    let input = input.ok_or("--input is required")?;
    if baseline.is_some() && instrumentation_baseline.is_some() {
        return Err("choose baseline or instrumentation-baseline, not both".into());
    }
    let checked = (|| {
        let mut checked = results::check_directory(&input)?;
        if let Some(dir) = instrumentation_baseline {
            let previous = results::check_directory(&dir)?;
            checked.report["instrumentation_comparison"] =
                smoke::compare_instrumentation(&checked, &previous)?;
        }
        if let Some(dir) = baseline {
            let previous = results::check_directory(&dir)?;
            results::compatible(&checked, &previous)?;
            if !previous.failures.is_empty() {
                checked
                    .failures
                    .push("baseline failed its own correctness/completion gates".into());
            }
            checked.report["baseline_provenance"] = previous.report["provenance"].clone();
            checked.report["failures"] = json!(checked.failures);
            if !checked.failures.is_empty() {
                checked.report["status"] = json!("failed");
            }
        }
        Ok(checked)
    })()
    .unwrap_or_else(|error| results::invalid_for_directory(error, &input));
    let encoded = serde_json::to_string_pretty(&checked.report).map_err(|e| e.to_string())?;
    if let Some(output) = output {
        results::write_outputs(&checked, &output)?;
    }
    println!("{encoded}");
    Ok(if checked.report["complete_records"] != true {
        2
    } else {
        u8::from(!checked.failures.is_empty())
    })
}

fn main() -> std::process::ExitCode {
    match run() {
        Ok(code) => std::process::ExitCode::from(code),
        Err(error) => {
            if std::env::args().nth(1).as_deref() == Some("smoke") {
                println!(
                    "{}",
                    json!({"schema":1,"record":"error","case":"smoke","error":error,"complete_records":false,"performance_accepted":false})
                );
            } else if std::env::args().nth(1).as_deref() == Some("frontier") {
                println!(
                    "{}",
                    json!({"schema":1,"report":"tractor-frontier-v1","complete":false,"acceptance":false,"error":error})
                );
            } else {
                println!("{}", results::invalid(error).report);
            }
            std::process::ExitCode::from(2)
        }
    }
}
