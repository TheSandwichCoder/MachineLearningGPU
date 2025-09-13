// struct NNDir{
//     read_layer_start: u32,
//     read_layer_length: u32,

//     param_layer_start: u32,
//     param_layer_length:u32,

//     write_layer_start: u32,
//     write_layer_length: u32,

//     n_batches: u32, 
//     batch_act_size: u32,

//     activation_type: u32,
// };

struct MatrixDir{
    layer_length: u32,

    n: u32,
    m: u32,
    k: u32
}

@group(0) @binding(0) var<storage, read> params: array<f32>;
@group(0) @binding(1) var<storage, read_write> activities: array<f32>; 
@group(0) @binding(2) var <uniform> mat_dir: MatrixDir;
// @group(0) @binding(2) var <uniform> nn_dir: NNDir;

// all the same (I suck at coding)
const T_K = 16;
const T_N = 16;
const T_M = 16;

var<workgroup> a_sub : array<array<f32, T_K>, T_M>; 
var<workgroup> b_sub : array<array<f32, T_N>, T_K>; 

@compute @workgroup_size(T_N, T_M)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    
    var k_i = 0;

    let wx = wg.x;
    let wy = wg.y;

    let tx = lid.x;
    let ty = lid.y;

    let gx = wx * T_N + tx;
    let gy = wy * T_M + ty;

    var v = 0.0;

    loop{
        if k_i >= mat_dir.k{
            break;
        }

        // loading
        if (ty + k_i < mat_dir.k){
            a_sub[tx][ty] = ;
            
            b_sub[tx][ty] = ;
        }

        workgroupBarrier();

        // get partial derivs
        let limit = min(T_K, mat_dir.k - k_i);

        for (var i: u32 = 0; i < limit; i++) {
            v += a_sub[tx][i] * b_sub[i][ty];
        }

        workgroupBarrier()        

        k_i += T_K;
    }


}
