struct MatrixDir{
    n_read_start: u32,
    m_read_start: u32,

    n_stride_length: u32,
    m_stride_length: u32,
    
    w_start: u32,
    w_stride_length: u32,

    add_const: u32,
    c_read_start: u32,
    c_stride_length: u32,
    a_func_type: u32,

    n: u32,
    m: u32,
    k: u32
}

@group(0) @binding(0) var<storage, read> read_buffer1: array<f32>;
@group(0) @binding(1) var<storage, read_write> read_buffer2: array<f32>;

@group(0) @binding(2) var <uniform> mat_dir: MatrixDir;
// @group(0) @binding(2) var <uniform> nn_dir: NNDir;

// all the same (I suck at coding)
const T_K : u32 = 16;
const T_N : u32 = 16;
const T_M : u32 = 16;

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

    var v = 0.0;

    let is_dead : bool = (g_n >= mat_dir.n || g_m >= mat_dir.m);

    loop{
        if k_i >= mat_dir.k{
            break;
        }

        let l_n_i = k_i + t_n;
        let l_m_i = k_i + t_m;

        // loading
        // assignment for forward

        if (g_n < mat_dir.n && l_m_i < mat_dir.k){
            a_sub[t_n][t_m] = read_buffer1[mat_dir.n_read_start + mat_dir.n_stride_length * g_n + l_m_i];            
        }
        if (g_m < mat_dir.m && l_n_i < mat_dir.k){
            b_sub[t_n][t_m] = read_buffer2[mat_dir.m_read_start + mat_dir.m_stride_length * g_m + l_n_i];
        }    
        

        workgroupBarrier();
        
        if !is_dead{
            let limit = min(T_K, mat_dir.k - k_i);

            for (var i: u32 = 0; i < limit; i++) {
                v += a_sub[t_n][i] * b_sub[i][t_m];
                // v = a_sub[t_n][i];
            }
        }


        workgroupBarrier();

        k_i += T_K;
    }


    if !is_dead{
        if (mat_dir.add_const == 1){
            v += read_buffer1[mat_dir.c_read_start + mat_dir.c_stride_length * g_n];
        }

        if (mat_dir.a_func_type == 0){

        }
        else if (mat_dir.a_func_type == 1){
            v = ReLu(v);
        }

        read_buffer2[mat_dir.w_start + mat_dir.w_stride_length * g_m + g_n] = v;    
    }
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