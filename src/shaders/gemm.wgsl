struct MatrixDir{
    n_read_start: u32,
    m_read_start: u32,

    n_stride_length: u32,
    m_stride_length: u32,
    
    w_start: u32,
    w_stride_length: u32,

    add_const: bool,
    c_start: u32,
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

    let wx = wg.x;
    let wy = wg.y;

    let tx = lid.x;
    let ty = lid.y;

    let gx = wx * T_N + tx;
    let gy = wy * T_M + ty;

    var v = 0.0;

    if (gx > mat_dir.n || gy > mat_dir.m){
        return;
    }

    loop{
        if k_i >= mat_dir.k{
            break;
        }

        let l_x_i = k_i + tx;
        let l_y_i = k_i + ty;

        // loading
        if (l_y_i < mat_dir.k){
            // assignment for forward

            if (l_x_i < mat_dir.n){
                a_sub[tx][ty] = read_buffer1[mat_dir.n_read_start + mat_dir.n_stride_length * gy + l_x_i];
            }
            if (l_y_i < mat_dir.m){
                b_sub[tx][ty] = read_buffer2[mat_dir.m_read_start + mat_dir.m_stride_length * gx + l_y_i];
            }
            
        }

        workgroupBarrier();

        // get partial derivs
        let limit = min(T_K, mat_dir.k - k_i);

        for (var i: u32 = 0; i < limit; i++) {
            v += a_sub[tx][i] * b_sub[i][ty];
        }

        workgroupBarrier();

        k_i += T_K;
    }

    read_buffer2[mat_dir.w_start + mat_dir.w_stride_length * gy + gx] = v;
    
}
