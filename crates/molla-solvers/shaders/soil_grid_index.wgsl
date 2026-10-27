override GRID_SIZE:u32;
override CAPACITY:u32;
override METADATA_WORD:u32;
override TILE_IDS_WORD:u32;
override SPARSE:bool;
@group(0) @binding(0) var<storage,read> metadata:array<u32>;
@group(0) @binding(1) var<storage,read_write> worlds:array<u32>;
@group(0) @binding(2) var<storage,read_write> physical:array<u32>;
@group(0) @binding(3) var<storage,read_write> status:atomic<u32>;
@compute @workgroup_size(64)
fn index_cells(@builtin(global_invocation_id) id:vec3<u32>) {
    let i=id.x;if i>=CAPACITY {return;}
    worlds[i]=0xffffffffu;physical[i]=0xffffffffu;
    if metadata[METADATA_WORD+2u]!=0u || metadata[METADATA_WORD]>CAPACITY/64u {
        atomicOr(&status,0x80000000u);return;
    }
    if atomicLoad(&status)!=0u || i/64u>=metadata[METADATA_WORD] {return;}
    let width=(GRID_SIZE+3u)/4u;
    let tile=metadata[TILE_IDS_WORD+i/64u];
    if tile>=width*width*width {atomicOr(&status,256u);return;}
    let local=i%64u;
    let xyz=vec3<u32>(tile%width,(tile/width)%width,tile/(width*width))*4u
        +vec3<u32>(local%4u,(local/4u)%4u,local/16u);
    if any(xyz>=vec3<u32>(GRID_SIZE)) {return;}
    let world=xyz.x+GRID_SIZE*(xyz.y+GRID_SIZE*xyz.z);
    worlds[i]=world;physical[i]=select(world,i,SPARSE);
}
