/// Copy bytes into a write-only mapped buffer in bounded 256-byte blocks.
pub fn copy_to_mapped(destination: &mut wgpu::BufferViewMut, source: &[u8]) {
    assert_eq!(destination.len(), source.len());
    let (blocks, tail) = source.as_chunks::<256>();
    let (mapped, mut remainder) = destination.slice(..).into_chunks::<256>();
    for (block, bytes) in mapped.into_iter().zip(blocks) {
        block.write(*bytes);
    }
    remainder.copy_from_slice(tail);
}
