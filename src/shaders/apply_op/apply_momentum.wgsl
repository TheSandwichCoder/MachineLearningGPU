struct ApplyDir{
    n_params: u32,
    n_weights: u32,
    batch_contribution: f32,
    lr: f32,
    mr: f32,
    vr: f32,
    wd: f32,
};

@group(0) @binding(0) var<storage, read_write> params: array<f32>;
@group(0) @binding(1) var<storage, read_write> momentum: array<f32>; 
@group(0) @binding(2) var<storage, read_write> variance: array<f32>; 
@group(0) @binding(3) var<storage, read> gradients: array<f32>; 
@group(0) @binding(4) var <uniform> app_dir: ApplyDir;


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

    if (param_i >= app_dir.n_params){
        return;
    }

    let gradient_corrected = gradients[param_i] * app_dir.batch_contribution;

    let momentum_value = momentum[param_i] * app_dir.mr + gradient_corrected * (1.0 - app_dir.mr);
    let variance_value = variance[param_i] * app_dir.vr + gradient_corrected * gradient_corrected * (1.0 - app_dir.vr);

    let momentum_corrected = momentum_value / (1.0 - pc.mr_dec);
    let variance_corrected = variance_value / (1.0 - pc.vr_dec);

    let gradient_update = momentum_corrected / (sqrt(variance_corrected) + EPS);

    let w = params[param_i];
    params[param_i] = w - gradient_update * app_dir.lr;

    // not for nn and is not a bias
    if (param_i < app_dir.n_weights){
        params[param_i] -= app_dir.lr * app_dir.wd * w;
    }

    momentum[param_i] = momentum_value;
    variance[param_i] = variance_value;
}
