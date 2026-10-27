override METADATA_WORD: u32;
override FAULT_BIT: u32 = 0x80000000u;
@group(0) @binding(0) var<storage, read_write> grid: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read_write> status: atomic<u32>;

@compute @workgroup_size(1)
fn propagate() {
    if atomicLoad(&grid[METADATA_WORD + 2u]) != 0u {
        atomicOr(&status, FAULT_BIT);
    }
}
