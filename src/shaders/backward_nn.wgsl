struct NNDir{
    prev_layer_start: u32,
    prev_layer_length: u32,

    param_layer_start: u32,
    param_layer_length:u32,

    curr_layer_start: u32,
    curr_layer_length: u32,

    next_layer_start: u32,
    next_layer_length: u32,

    ping_start: u32,
    pong_start: u32,

    gradients_start: u32,
    gradient_length: u32,

    n_batches: u32, 
    batch_act_size: u32,

    activation_type: u32,
};

@group(0) @binding(0) var<storage, read> params: array<f32>;
@group(0) @binding(1) var<storage, read_write> activities: array<f32>; 
@group(0) @binding(2) var <uniform> nn_dir: NNDir;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let output_i = gid.x;
    let input_i = gid.y;
    let batch_i = gid.z;
    let act_start = nn_dir.batch_act_size * batch_i;

    if (output_i >= nn_dir.curr_layer_length){
        return;
    }

    let tens_start = nn_dir.param_layer_start + nn_dir.param_layer_length * output_i;

    let output_gradient_start = nn_dir.gradients_start + nn_dir.gradient_length * output_i;

    let input_deriv_start = nn_dir.ping_start;
    let output_deriv_start = nn_dir.pong_start + nn_dir.prev_layer_length * output_i;

    let z = activities[act_start + nn_dir.curr_layer_start + output_i];
    
    var act_z = 1.0;
    
    if (nn_dir.activation_type != 0){
        act_z = derivReLu(z);
    }

    // reset the gradients and derivatives
    activities[act_start + output_gradient_start + input_i] = 0.0;

    if (input_i < nn_dir.prev_layer_length){
        activities[act_start + output_deriv_start + input_i] = 0.0;
    }
    

    for (var der_i: u32 = 0; der_i < nn_dir.next_layer_length; der_i++){
        let part_deriv = activities[act_start + input_deriv_start + der_i * nn_dir.curr_layer_length + output_i];
        
        // weights
        if (input_i < nn_dir.prev_layer_length){
            let prev_act_info = activities[act_start + nn_dir.prev_layer_start + input_i];
            let curr_act_info = params[tens_start + input_i];

            let gradient_value = act_z * part_deriv * prev_act_info;
            let deriv_value = act_z * part_deriv * curr_act_info;

            activities[act_start + output_gradient_start + input_i] += gradient_value;

            activities[act_start + output_deriv_start + input_i] += deriv_value;
        }

        // bias
        else{
            activities[act_start + output_gradient_start + input_i] += part_deriv;
        }
    }
}

fn derivReLu(x: f32) -> f32{
  if (x > 0.0){
    return 1;
  }

  return 0;
}