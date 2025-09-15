struct MatrixDir{
    assign_type: u32,

    n_read_start: u32,
    m_read_start: u32,

    w_start: u32,

    n_stride_length: u32,
    m_stride_length: u32,
    w_stride_length: u32,

    n: u32,
    m: u32,
    k: u32
}

@group(0) @binding(0) var<storage, read> read_buffer1: array<f32>;
@group(0) @binding(1) var<storage, read_write> read_buffer2: array<f32>;

@group(0) @binding(2) var <uniform> mat_dir: MatrixDir;

const T_K = 16;

@compute @workgroup_size(T_K, T_K)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    
    var k_i = 0;

    let wx = wg.x;
    let wy = wg.y;

    let tx = lid.x;
    let ty = lid.y;

    let gx = wx * T_K + tx;
    let gy = wy * T_K + ty;

    var v = 0.0;

    if (gx > mat_dir.n || gy > mat_dir.m){
        return;
    }

    let v1 = read_buffer1[mat_dir.n_read_start + mat_dir.n_stride_length * gy + gx];
    let v2 = read_buffer2[mat_dir.m_read_start + mat_dir.m_stride_length * gy + gx];

    read_buffer2[mat_dir.w_start + mat_dir.w_stride_length * gy + gx] = v1 + v2;
}
