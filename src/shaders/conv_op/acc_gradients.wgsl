struct AccDir{
    read_start: u32,
    write_start: u32,
    acc_length: u32,
    n_weights: u32,
}

@group(0) @binding(0) var<storage, read> read_buffer: array<f32>;
@group(0) @binding(1) var<storage, read_write> write_buffer: array<f32>;

@group(0) @binding(2) var <uniform> acc_dir: AccDir;

@compute @workgroup_size(32)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= acc_dir.n_weights){
        return;
    }

    let start_idx = gid.x * acc_dir.acc_length;

    var v = 0.0;
    for (var i:u32 = 0; i < acc_dir.acc_length; i += 1){
        let idx = start_idx + i;

        v += read_buffer[acc_dir.read_start + idx];
        // v += 1.0;
    }

    write_buffer[acc_dir.write_start + gid.x] = v;
    // write_buffer[acc_dir.write_start + gid.x] = 1.0;
}