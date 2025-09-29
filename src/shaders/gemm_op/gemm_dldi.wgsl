struct MatrixDir{
    kernal_dim: vec4<u32>,
    layer_dim: vec4<u32>,

    kernal_read_start: u32,
    layer_read_start: u32,
    write_start: u32,

    c_start: u32,

    n_outputs: u32, // number of outputs for a single batch
    batch_swap_buffer_size: u32, // size of the swap buffer for a single input

    kernal_size: u32, // number of values in the 2d kernal

    n: u32,
    m: u32,
    k: u32
}

@group(0) @binding(0) var<storage, read> read_buffer1: array<f32>;
@group(0) @binding(1) var<storage, read_write> read_buffer2: array<f32>;

@group(0) @binding(2) var <uniform> mat_dir: MatrixDir;

// all the same (I suck at coding)
const T_K : u32 = 16;
const T_N : u32 = 16;
const T_M : u32 = 16;

const N_TRANSPOSE: u32 = 0;
const M_TRANSPOSE: u32 = 1;
const O_TRANSPOSE: u32 = 2;

var<workgroup> a_sub : array<array<f32, T_K>, T_M>; 
var<workgroup> b_sub : array<array<f32, T_N>, T_K>; 

@compute @workgroup_size(T_N, T_M)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    
    var k_i : u32 = 0;

    let w_n = wg.x;
    let w_m = wg.y;

    let t_n = lid.x;
    let t_m = lid.y;

    let g_n = w_n * T_N + t_n;
    let g_m = w_m * T_M + t_m;

    let batch_g_m = g_m % mat_dir.n_outputs;
    let batch_i = g_m / mat_dir.n_outputs;

    let batch_buffer_offset = u32(batch_i * mat_dir.batch_swap_buffer_size);

    var v = 0.0;


    let is_dead : bool = (g_n >= mat_dir.n || g_m >= mat_dir.m);
    
    let glob_kernal_pos = expand(batch_g_m, mat_dir.layer_dim.xyz);

    let kernal_i_offset = g_n * mat_dir.k;
    
    loop{
        if k_i >= mat_dir.k{
            break;
        }

        let l_n_i = k_i + t_n;
        let l_m_i = k_i + t_m;

        // loading
        // assignment for forward

        if (g_n < mat_dir.n && l_m_i < mat_dir.k){
            let read_idx = kernal_i_offset + (mat_dir.kernal_size - l_m_i); // rotate kernal by 180

            a_sub[t_n][t_m] = read_buffer1[mat_dir.kernal_read_start + read_idx];
        }
        if (g_m < mat_dir.m && l_n_i < mat_dir.k){
            let rel_kernal_pos = expand(l_n_i, mat_dir.kernal_dim.xyz);
            
            let read_pos = rel_kernal_pos + glob_kernal_pos;

            let read_idx = flatten_safe(read_pos, vec3i(mat_dir.layer_dim.xyz));

            if read_idx == -1{
                b_sub[t_n][t_m] = 0.0;
            }
            else{
                b_sub[t_n][t_m] = read_buffer2[mat_dir.layer_read_start + u32(read_idx) + batch_buffer_offset];
            }
        }    
        

        workgroupBarrier();
        
        if !is_dead{
            let limit = min(T_K, mat_dir.k - k_i);

            for (var i: u32 = 0; i < limit; i++) {
                v += a_sub[t_n][i] * b_sub[i][t_m];
            }
        }
        
        workgroupBarrier();

        k_i += T_K;
    }


    if !is_dead{
        // v += read_buffer1[mat_dir.c_start + g_n];

        // v = ReLu(v);

        let write_idx = g_n * mat_dir.n_outputs + batch_g_m;

        read_buffer2[mat_dir.write_start + u32(write_idx) + batch_buffer_offset] = v;
    }
}

fn expand(i: u32, k: vec3u) -> vec3i{
    let z = i / (k.y * k.x);

    let temp_val = (i % (k.y * k.x));

    let y = temp_val / k.y;
    let x = temp_val % k.y;

    return vec3i(i32(x), i32(y), i32(z));
}

fn flatten_safe(v: vec3i, k: vec3i) -> i32{
    // safety check (out of bounds)
    if (v.x < 0 || v.x >= k.x) || (v.y < 0 || v.y >= k.y) || (v.z < 0 || v.z >= k.z){
        return -1;
    }

    let val = v.z * k.x * k.y + v.y * k.x + v.x;
    
    return val;
}

fn flatten(v: vec3i, k: vec3i) -> i32{
    let val = v.z * k.x * k.y + v.y * k.x + v.x;
    
    return val;
}

fn read_transpose(x: u32, i: u32) -> bool{
    return (x & u32(1 << i)) != 0;
}

fn ReLu(x: f32) -> f32{
    if x > 0{
        return x;
    }
    return 0;
}

fn dReLu(x: f32) -> f32{
    if (x > 0){
        return 1;
    }

    return 0;
}