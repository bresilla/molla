use molla_compute::Engine;
use std::sync::{Arc, OnceLock};

pub fn engine() -> Arc<Engine> {
    static ENGINE: OnceLock<Arc<Engine>> = OnceLock::new();
    ENGINE
        .get_or_init(|| {
            let engine = Arc::new(Engine::new_gpu().unwrap());
            let adapter = engine.adapter_info().expect("requires a GPU adapter");
            assert!(!adapter.is_software, "requires a hardware GPU");
            println!("adapter={adapter:?}");
            engine
        })
        .clone()
}

#[test]
#[ignore = "requires a hardware GPU through make test-gpu"]
fn hardware_engine_identity_is_shared_within_the_test_executable() {
    assert!(Arc::ptr_eq(&engine(), &engine()));
}
