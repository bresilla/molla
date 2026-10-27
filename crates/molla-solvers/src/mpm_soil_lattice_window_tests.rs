use super::*;

fn state(count: u32) -> MpmSoilCheckpoint {
    let mut state = super::super::tests::fixture(count);
    state.cfg.grid_origin = [-3.0, -0.5, -3.0];
    state.cfg.grid_size = 64;
    state.cfg.cell_size = 0.1;
    state
}

#[test]
fn lattice_admission_decimal_windows_preserve_words_and_roundtrip() {
    for count in [0, 1, 129] {
        for time_step in [
            MpmTimeStep::Explicit,
            MpmTimeStep::Implicit { jacobi_iters: 2 },
        ] {
            let mut before = state(count);
            before.cfg.time_step = time_step;
            let bytes = before.to_bytes();
            for cells in [[-1, 0, 1], [1, 0, -1], [0; 3]] {
                let moved = before.with_grid_window_cells(cells).unwrap();
                assert_eq!(moved.words, before.words);
                assert_eq!(moved.cfg, before.cfg);
                assert_eq!(moved.grid_window_cells(), cells);
                moved.validate_lattice_window().unwrap();
                let decoded = MpmSoilCheckpoint::from_bytes(&moved.to_bytes(), count).unwrap();
                assert_eq!(decoded.to_bytes(), moved.to_bytes());
                assert_eq!(
                    decoded.with_grid_window_cells([0; 3]).unwrap().to_bytes(),
                    bytes
                );
            }
            assert_eq!(before.to_bytes(), bytes);
        }
    }
}

#[test]
fn lattice_admission_floor_support_ranges_and_no_mutation() {
    let mut before = state(1);
    before.words[1] = (-0.3_f32).to_bits();
    let bytes = before.to_bytes();
    assert!(before.with_grid_window_cells([-1, 0, 1]).is_ok());
    for cells in [
        [0, 1, 0],
        [30, 0, 0],
        [65, 0, 0],
        [i32::MIN, 0, 0],
        [0, 0, i32::MAX],
    ] {
        assert!(before.with_grid_window_cells(cells).is_err());
        assert_eq!(before.to_bytes(), bytes);
    }
    before.cfg.time_step = MpmTimeStep::Implicit { jacobi_iters: 64 };
    assert!(before.with_grid_window_cells([0; 3]).is_ok());
    assert!(before.with_grid_window_cells([1, 0, 0]).is_err());
    for position in [f32::MAX, f32::MIN, f32::INFINITY, f32::NAN, -9.0] {
        let mut bad = state(1);
        bad.words[0] = position.to_bits();
        assert!(bad.with_grid_window_cells([1, 0, 0]).is_err());
    }
    let mut bad = state(1);
    bad.grid_window_cells = [i32::MAX; 3];
    assert!(bad.validate_lattice_window().is_err());
    assert!(state(0).with_grid_window_cells([65, 0, 0]).is_err());
    let mut tiny = state(0);
    tiny.cfg.cell_size = f32::from_bits(1);
    assert!(tiny.with_grid_window_cells([1, 0, 0]).is_err());
}
