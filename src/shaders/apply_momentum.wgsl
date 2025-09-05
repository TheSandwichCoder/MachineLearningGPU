struct NNDir{
    batch_start_i: u32,
    batch_i: u32,
    momentum_start_i: u32,
    batch_contribution: f32,
    n_params: u32,
    lr: f32,
    mr: f32,
};

@group(0) @binding(0) var<storage, read_write> params: array<f32>;
@group(0) @binding(1) var<storage, read_write> momentum: array<f32>; 
@group(0) @binding(2) var<storage, read> gradients: array<f32>; 
@group(0) @binding(3) var <uniform> nn_dir: NNDir;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let param_i = gid.x;

    if (param_i >= nn_dir.n_params){
        return;
    }

    let momentum_value = momentum[param_i] * nn_dir.mr + gradients[param_i] * nn_dir.lr;

    params[param_i] += momentum_value;
    momentum[param_i] = momentum_value;
}
