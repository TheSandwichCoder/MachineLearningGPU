struct MatrixDir{
    kernal_dim: vec4<u32>,
    o_layer_offset: vec4<i32>,
    o_layer_dim: vec4<u32>,
    i_layer_dim: vec4<u32>,

    deriv_read_start: u32,
    input_read_start: u32,
    write_start: u32,

    c_start: u32,

    n_outputs: u32, // number of outputs for a single batch
    batch_swap_buffer_size: u32, // size of the swap buffer for a single batch
    batch_input_buffer_size: u32, // size of the input buffer for a single batch
    acc_buffer_batch_length: u32, // size of the accumulate buffer for a batch

    split_k: u32, // num of values in k slice
    n_k_splits: u32, // num of k slices
    batch_k_length: u32, // length of a k batch length

    n: u32,
    m: u32,
    k: u32
}


@group(0) @binding(0) var<storage, read> input_buffer: array<f32>;
@group(0) @binding(1) var<storage, read> deriv_buffer: array<f32>;
@group(0) @binding(2) var<storage, read_write> accumulate_buffer: array<f32>;

@group(0) @binding(3) var <uniform> mat_dir: MatrixDir;

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
    

    let w_n = wg.x;
    let w_m = wg.y;

    let t_n = lid.x;
    let t_m = lid.y;

    let g_n = w_n * T_N + t_n;
    let g_m = w_m * T_M + t_m;

    var k_i : u32 = mat_dir.split_k * wg.z;
    let k_stop: u32 = mat_dir.split_k * (wg.z + 1);

    // let batch_g_m = g_m % mat_dir.n_outputs;
    // let batch_i = g_m / mat_dir.n_outputs;

    // let batch_buffer_offset = u32(batch_i * mat_dir.batch_swap_buffer_size);
    // let acc_buffer_offset = u32(batch_i * mat_dir.acc_buffer_batch_length);

    var v = 0.0;


    let is_dead : bool = (g_n >= mat_dir.n || g_m >= mat_dir.m);
    
    let glob_kernal_pos = expand(g_m, mat_dir.kernal_dim.xyz);

    let layer_i_offset = g_n * mat_dir.batch_k_length;
    
    loop{
        if k_i >= mat_dir.k || k_i >= k_stop{
            break;
        }

        let batch_k_i = k_i % mat_dir.batch_k_length;
        let batch_i = k_i / mat_dir.batch_k_length;

        let l_n_i = k_i + t_n;
        let l_m_i = k_i + t_m;

        let batch_l_n_i = batch_k_i + t_n;
        let batch_l_m_i = batch_k_i + t_m;

        // TODO check if this is the way to calculate the offset
        let batch_deriv_read_offset = batch_i * mat_dir.batch_swap_buffer_size;
        let batch_input_read_offset = batch_i * mat_dir.batch_input_buffer_size;

        // loading
        // assignment for forward

        if (g_n < mat_dir.n && l_m_i < mat_dir.k){
            let read_idx = layer_i_offset + batch_l_m_i;

            // a_sub[t_n][t_m] = f32(l_m_i) / 100.0 + 1.0;
            a_sub[t_n][t_m] = deriv_buffer[mat_dir.deriv_read_start + read_idx + batch_deriv_read_offset];
            // a_sub[t_n][t_m] = 1.0;
        }
        
        if (g_m < mat_dir.m && l_n_i < mat_dir.k){
            let rel_kernal_pos = expand(batch_l_n_i, mat_dir.o_layer_dim.xyz);
            
            let read_pos = glob_kernal_pos + rel_kernal_pos + mat_dir.o_layer_offset.xyz;

            let read_idx = flatten_safe(read_pos, vec3i(mat_dir.i_layer_dim.xyz));

            if read_idx == -1{
                b_sub[t_n][t_m] = 0.0;
            }
            else{
                // b_sub[t_n][t_m] = input_buffer[mat_dir.input_read_start + u32(read_idx) + batch_input_read_offset];
                b_sub[t_n][t_m] = 1.0;
            }
        }    
        

        workgroupBarrier();
        
        if !is_dead{
            let limit = min(T_K, mat_dir.k - k_i);

            for (var i: u32 = 0; i < limit; i++) {
                // v += 1.0;
                v += a_sub[t_n][i] * b_sub[i][t_m];
                // v += a_sub[t_n][i];
            }
        }
        
        workgroupBarrier();

        k_i += T_K;
    }

    if !is_dead{
        // v += read_buffer1[mat_dir.c_start + g_n];

        // v = ReLu(v);

        let write_idx = g_n * mat_dir.n_outputs * mat_dir.n_k_splits + g_m * mat_dir.n_k_splits + wg.z;
        // let write_idx = g_n * mat_dir.n_k_splits + batch_g_m * mat_dir.n_k_splits + wg.z;

        // accumulate_buffer[write_idx] = 1.0;
        accumulate_buffer[mat_dir.write_start + u32(write_idx)] = v;
        // accumulate_buffer[mat_dir.write_start + acc_buffer_offset + u32(write_idx)] = 1.0;
    }}

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