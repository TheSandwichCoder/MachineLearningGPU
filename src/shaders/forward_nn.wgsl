struct NNDir{
    read_layer_start: u32,
    read_layer_length: u32,

    param_layer_start: u32,
    param_layer_length:u32,

    write_layer_start: u32,
    write_layer_length: u32,

    n_batches: u32, 
    batch_act_size: u32,

    activation_type: u32,
};


@group(0) @binding(0) var<storage, read> params: array<f32>;
@group(0) @binding(1) var<storage, read_write> activities: array<f32>; 
@group(0) @binding(2) var <uniform> nn_dir: NNDir;

@compute @workgroup_size(64, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let output_i = gid.x;
  let batch_i = gid.y;

  if (output_i >= nn_dir.write_layer_length){
    return;
  }

  let tens_start = nn_dir.param_layer_start + nn_dir.param_layer_length * output_i;
  let act_start = nn_dir.batch_act_size * batch_i;

  var z : f32 = 0.0;

  for (var input_i: u32 = 0; input_i < nn_dir.read_layer_length; input_i++) {
    z += params[tens_start + input_i] * activities[act_start + nn_dir.read_layer_start + input_i];
  }

  z += params[tens_start + nn_dir.read_layer_length];

  if (nn_dir.activation_type == 1){
    z = ReLu(z);
  }

  activities[act_start + nn_dir.write_layer_start + output_i] = z;
}

fn ReLu(x: f32) -> f32{
  if (x > 0.0){
    return x;
  }

  return 0;
}