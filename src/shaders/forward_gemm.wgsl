struct MatrixDir{
    act_length: u32,
    n_start: u32,
    m_start: u32,
    w_start: u32,

    n: u32,
    m: u32,
    k: u32
}

@group(0) @binding(0) var<storage, read> params: array<f32>;
@group(0) @binding(1) var<storage, read_write> activities: array<f32>; 
@group(0) @binding(2) var <uniform> mat_dir: MatrixDir;

// all the same (I suck at coding)
const T_K : u32 = 64;
const T_M : u32 = 64;
const T_N : u32 = 64;

var<workgroup> a_sub : array<f32, T_K>; 
var<workgroup> b_sub : array<array<f32, T_M>, T_K>; 

@compute @workgroup_size(T_M)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    
    var k_i : u32 = 0;

    let wy = wg.x;

    let ty = lid.x;

    let gy = wy * T_M + ty;

    let act_start = wg.y * mat_dir.act_length;
    
    var v = 0.0;

    loop{
        if (k_i >= mat_dir.k){
            break;
        }

        // loading

        if (k_i + ty < mat_dir.k){
            a_sub[ty] = activities[act_start + mat_dir.n_start + k_i + ty];
            
            for (var i: u32 = 0; i < T_K; i++) {
                b_sub[i][ty] = params[mat_dir.m_start + (mat_dir.k + 1) * gy + k_i + i];
            }
        }

        workgroupBarrier();

        let limit = min(T_K, mat_dir.k - k_i);

        for (var i: u32 = 0; i < limit; i++) {
            v += a_sub[i] * b_sub[i][ty];
        }

        workgroupBarrier();        

        k_i += T_K;
    }

    // bias and activation
    v += params[mat_dir.m_start + (mat_dir.k + 1) * gy + mat_dir.k];

    v = ReLu(v);

    activities[act_start + mat_dir.w_start + gy] = v;
}

fn ReLu(x: f32) -> f32{
  if (x > 0.0){
    return x;
  }

  return 0;
}


