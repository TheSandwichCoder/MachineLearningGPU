struct NNDir{
    act_size: u32,
    outputs_offset: u32,
    n_outputs: u32,
    ping_start: u32,
    data_size: u32,
};

struct PC {
  batch_start_idx : u32,
  _pad1 : u32,
  _pad2 : u32,
  _pad3 : u32, // keep 16B alignment friendly; total 16 bytes here
};

@group(0) @binding(0) var<storage, read_write> activities: array<f32>; 
@group(0) @binding(1) var<storage, read> data: array<f32>; 
@group(0) @binding(2) var <uniform> nn_dir: NNDir;


var<push_constant> pc: PC;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let batch_i = gid.x;
    
    if (batch_i >= 16){
        return;
    }

    let act_start = batch_i * nn_dir.act_size;

    let label = u32(data[pc.batch_start_idx + batch_i * (nn_dir.data_size + 1)]);

    // softmax
    let output_start_i = act_start + nn_dir.outputs_offset;

    let outputs = softmax(output_start_i);

    for (var output_i:u32 = 0; output_i < nn_dir.n_outputs; output_i += 1){
        if (output_i == label){
            activities[act_start + nn_dir.ping_start + output_i] = 1.0 - outputs[output_i];
        }
        else{
            activities[act_start + nn_dir.ping_start + output_i] = -outputs[output_i];
        }
    }
}

fn get_max(start_i: u32) -> f32{
    var largest = activities[start_i];

    for (var output_i:u32 = 1; output_i < nn_dir.n_outputs; output_i += 1){
        let value = activities[start_i + output_i];

        if (value > largest){
            largest = value;
        }
    }

    return largest;
}

fn softmax(start_i: u32) -> array<f32, 32>{
    var softmax_prob: array<f32, 32>;

    var total_sum = 0.0;

    let max_val = get_max(start_i);

    for (var output_i:u32 = 0; output_i < nn_dir.n_outputs; output_i += 1){
        let value = exp(activities[start_i + output_i] - max_val);

        total_sum += value;

        softmax_prob[output_i] = value;
    }

    for (var output_i:u32 = 0; output_i < nn_dir.n_outputs; output_i += 1){
        softmax_prob[output_i] /= total_sum;
    }

    return softmax_prob;
}