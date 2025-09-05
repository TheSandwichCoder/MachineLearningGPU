
struct NNDir{
    batch_start_i: u32,
    batch_i: u32,
    gradient_start_i: u32,
    batch_contribution: f32,
    n_params: u32,
    lr: f32,
    mr: f32,
};

@group(0) @binding(0) var<storage, read_write> gradients: array<f32>;
@group(0) @binding(1) var<storage, read> activities: array<f32>; 
@group(0) @binding(2) var <uniform> nn_dir: NNDir;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let gradient_i = gid.x;

    if (gradient_i >= nn_dir.n_params){
        return;
    }

    // reset gradients
    if nn_dir.batch_i == 0{
        gradients[gradient_i] = 0.0;
    }

    let batch_start = nn_dir.batch_start_i;

    gradients[gradient_i] += activities[batch_start + nn_dir.gradient_start_i + gradient_i] * nn_dir.batch_contribution;
}
