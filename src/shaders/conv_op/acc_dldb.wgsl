struct MatrixDir{
    layer_size_2d: u32,
    deriv_read_start: u32,
    write_start: u32,
    batch_offset: u32,

    split_k: u32,
    acc_length: u32,

    n_batches: u32,
    n_kernals: u32,
}


@group(0) @binding(0) var<storage, read> deriv_buffer: array<f32>;
@group(0) @binding(1) var<storage, read_write> accumulate_buffer: array<f32>;
@group(0) @binding(2) var <uniform> mat_dir: MatrixDir;


const T_N = 16;

// k - number of derivs

// x - layer i
// y - acc i
// z - batch i

@compute @workgroup_size(T_N, T_N)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let layer_i = gid.x;
    let acc_i = gid.y;

    var k_i : u32 = mat_dir.split_k * acc_i;
    let k_stop: u32 = mat_dir.split_k * (acc_i + 1);
    
    let layer_i_offset = layer_i * mat_dir.layer_size_2d;

    if layer_i > mat_dir.n_kernals{
        return;
    }

    var v = 0.0;

    loop{
        if k_i >= mat_dir.layer_size_2d * mat_dir.n_batches || k_i >= k_stop{
            break;
        }
        
        let batch_i = k_i / mat_dir.layer_size_2d;

        let w_k_i = k_i % mat_dir.layer_size_2d;

        let batch_i_offset = batch_i * mat_dir.batch_offset;

        v += deriv_buffer[mat_dir.deriv_read_start + batch_i_offset + layer_i_offset + w_k_i];

        k_i++;
    }

    accumulate_buffer[mat_dir.write_start + layer_i * mat_dir.acc_length + acc_i] = v;
}
