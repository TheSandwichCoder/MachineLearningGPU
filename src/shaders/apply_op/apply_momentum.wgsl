struct NNDir{
    batch_start_i: u32,
    batch_i: u32,
    momentum_start_i: u32,
    batch_contribution: f32,
    n_params: u32,
    lr: f32,
    mr: f32,
    vr: f32,
};

@group(0) @binding(0) var<storage, read_write> params: array<f32>;
@group(0) @binding(1) var<storage, read_write> momentum: array<f32>; 
@group(0) @binding(2) var<storage, read_write> variance: array<f32>; 
@group(0) @binding(3) var<storage, read> gradients: array<f32>; 
@group(0) @binding(4) var <uniform> nn_dir: NNDir;


const EPS : f32= 0.00000001;

struct PC {
  mr_dec : f32,
  vr_dec : f32,
  _pad2 : u32,
  _pad3 : u32, // keep 16B alignment friendly; total 16 bytes here
};

var<push_constant> pc: PC;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let param_i = gid.x;

    if (param_i >= nn_dir.n_params){
        return;
    }

    let gradient_corrected = gradients[param_i] * nn_dir.batch_contribution;

    let momentum_value = momentum[param_i] * nn_dir.mr + gradient_corrected * (1.0 - nn_dir.mr);
    let variance_value = variance[param_i] * nn_dir.vr + gradient_corrected * gradient_corrected * (1.0 - nn_dir.vr);

    let momentum_corrected = momentum_value / (1.0 - pc.mr_dec);
    let variance_corrected = variance_value / (1.0 - pc.vr_dec);

    let gradient_update = momentum_corrected / (sqrt(variance_corrected) + EPS);

    // params[param_i] -= gradient_corrected * nn_dir.lr;
    params[param_i] -= gradient_update * nn_dir.lr;
    momentum[param_i] = momentum_value;
    variance[param_i] = variance_value;
}
