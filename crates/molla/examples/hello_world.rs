//! Stage 0 smoke example: build an `Engine`, print the chosen adapter info.
//!
//! ```bash
//! cargo run -p molla --example hello_world
//! ```

#[cfg(feature = "gpu")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use molla::compute::{Backend, Engine};

    println!("molla {} — stage 0 hello", env!("CARGO_PKG_VERSION"));

    println!();
    println!("CPU backend:");
    let cpu = Engine::new_cpu();
    println!("  backend = {:?}", cpu.backend());

    println!();
    println!("GPU backend (auto-select):");
    match Engine::new(Backend::AutoGpu) {
        Ok(gpu) => {
            if let Some(info) = gpu.adapter_info() {
                println!("  vendor   = {}", info.vendor);
                println!("  device   = {}", info.name);
                println!("  backend  = {}", info.backend);
                println!("  software = {}", info.is_software);
            }
        }
        Err(e) => {
            println!("  no GPU available: {e}");
            println!("  (this is fine on headless / pure-CPU hosts)");
        }
    }

    Ok(())
}

#[cfg(not(feature = "gpu"))]
fn main() {
    println!(
        "molla {} — built without `gpu` feature.",
        env!("CARGO_PKG_VERSION")
    );
    println!("Re-run with `--features gpu` (default) to see adapter info.");
}
