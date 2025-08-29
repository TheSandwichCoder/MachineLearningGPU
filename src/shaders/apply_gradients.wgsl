
struct NNDir{
    batch_start_i: u32,
    gradient_start_i: u32,
    batch_contribution: f32,
    lr: f32,
};

@group(0) @binding(0) var<storage, read_write> params: array<f32>;
@group(0) @binding(1) var<storage, read> activities: array<f32>; 
@group(0) @binding(2) var <uniform> nn_dir: NNDir;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let gradient_i = gid.x;

    let batch_start = nn_dir.batch_start_i;

    params[gradient_i] += activities[batch_start + nn_dir.gradient_start_i + gradient_i] * nn_dir.batch_contribution * nn_dir.lr;
}
