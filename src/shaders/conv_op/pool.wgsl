struct PoolDir{
    i_layer_dim: vec4<u32>,
    o_layer_dim: vec4<u32>,

    read_start: u32,
    write_start: u32,
    pool_k: u32,
    batch_swap_buffer_size: u32,

    storage_write_start: u32,
    storage_write_skip: u32,
}

@group(0) @binding(0) var<storage, read_write> swap_buffer: array<f32>;
@group(0) @binding(1) var<storage, read_write> o_storage_buffer: array<f32>;
@group(0) @binding(2) var<storage, read_write> pool_idx_storage_buffer: array<u32>;
@group(0) @binding(3) var <uniform> pool_dir: PoolDir;

const T_N : u32 = 8;
const T_M : u32 = 8;

@compute @workgroup_size(T_N, T_M)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {

    let n_channels = pool_dir.o_layer_dim.z;

    let channels_i = gid.z % n_channels;
    let batch_i = gid.z / n_channels; 

    let batch_read_offset = pool_dir.batch_swap_buffer_size * batch_i;
    let storage_batch_offset = pool_dir.storage_write_skip * batch_i;

    let kernal_coord = vec3u(gid.xy * pool_dir.pool_k, channels_i);

    if !in_bounds(kernal_coord, pool_dir.i_layer_dim.xyz){
        return;
    }
    
    var max_val = 0.0;
    var max_idx: u32 = 0;

    for (var ty:u32 = 0; ty < pool_dir.pool_k; ty += 1){
        for (var tx:u32 = 0; tx < pool_dir.pool_k; tx += 1){
            let read_coord = kernal_coord + vec3u(tx, ty, 0);

            if in_bounds(read_coord, pool_dir.i_layer_dim.xyz){
                let temp_val = swap_buffer[pool_dir.read_start + batch_read_offset + flatten(read_coord, pool_dir.i_layer_dim.xyz)];

                if temp_val > max_val{
                    max_val = temp_val;
                    max_idx = ty * pool_dir.pool_k + tx;
                }
            }
        }
    }

    let write_idx = flatten(vec3u(gid.xy, channels_i), pool_dir.o_layer_dim.xyz);
    // swap_buffer[pool_dir.write_start + batch_read_offset + write_idx] = max_val;
    // o_storage_buffer[pool_dir.storage_write_start + pool_dir.storage_write_skip * batch_i + write_idx] = 1.0;
    o_storage_buffer[pool_dir.storage_write_start + storage_batch_offset + write_idx] = max_val;
    pool_idx_storage_buffer[pool_dir.storage_write_start + storage_batch_offset + write_idx] = max_idx;
    swap_buffer[pool_dir.write_start + batch_read_offset + write_idx] = max_val;
}

fn flatten(v: vec3u, k: vec3u) -> u32{
    let val = v.z * k.x * k.y + v.y * k.x + v.x;
    
    return val;
}

fn in_bounds(v: vec3u, dim: vec3u) -> bool{
    return !(v.x >= dim.x || v.y >= dim.y || v.z >= dim.z);
}