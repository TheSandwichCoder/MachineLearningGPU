struct PoolDir{
    i_layer_dim: vec4<u32>,
    o_layer_dim: vec4<u32>,

    read_start: u32,
    write_start: u32,
    pool_k: u32,
}

@group(0) @binding(0) var<storage, read_write> act_buffer: array<f32>;

@group(0) @binding(1) var <uniform> pool_dir: PoolDir;

const T_N : u32 = 8;
const T_M : u32 = 8;

@compute @workgroup_size(T_N, T_M)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {

    let kernal_coord = vec3u(gid.xy * pool_dir.pool_k, gid.z);

    if !in_bounds(kernal_coord, pool_dir.i_layer_dim.xyz){
        return;
    }
    
    var max_val = 0.0;

    for (var ty:u32 = 0; ty < pool_dir.pool_k; ty += 1){
        for (var tx:u32 = 0; tx < pool_dir.pool_k; tx += 1){
            let read_coord = kernal_coord + vec3u(tx, ty, 0);

            if in_bounds(read_coord, pool_dir.i_layer_dim.xyz){
                let temp_val = act_buffer[pool_dir.read_start + flatten(read_coord, pool_dir.i_layer_dim.xyz)];

                if temp_val > max_val{
                    max_val = temp_val;
                }
            }
        }
    }

    act_buffer[pool_dir.write_start + flatten(gid, pool_dir.o_layer_dim.xyz)] = max_val;
}

fn flatten(v: vec3u, k: vec3u) -> u32{
    let val = v.z * k.x * k.y + v.y * k.x + v.x;
    
    return val;
}

fn in_bounds(v: vec3u, dim: vec3u) -> bool{
    return !(v.x >= dim.x || v.y >= dim.y || v.z >= dim.z);
}