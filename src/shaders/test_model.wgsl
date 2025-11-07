struct NNDir{
    act_size: u32,
    outputs_offset: u32,
    n_outputs: u32,
    ping_start: u32,
    data_size: u32,
};

struct PC {
  batch_start_idx : u32,
  n_batches : u32,
  _pad2 : u32,
  _pad3 : u32, // keep 16B alignment friendly; total 16 bytes here
};

struct Metrics {
  correct : atomic<u32>,
  incorrect : atomic<u32>,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<storage, read> activities: array<f32>; 
@group(0) @binding(1) var<storage, read> data: array<f32>; 
@group(0) @binding(2) var<storage, read_write> metrics : Metrics;
@group(0) @binding(3) var<uniform> nn_dir: NNDir;

var<push_constant> pc: PC;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let batch_i = gid.x;
    
    if (batch_i >= pc.n_batches){
        return;
    }

    let act_start = batch_i * nn_dir.act_size;

    let label = u32(data[pc.batch_start_idx + batch_i * (nn_dir.data_size + 1)]);

    // softmax
    let output_start_i = act_start + nn_dir.outputs_offset;

    let predicted_label = get_max(output_start_i);
    
    if (predicted_label == label){
        atomicAdd(&metrics.correct, 1u);
    }
    else{
        atomicAdd(&metrics.incorrect, 1u);
    }
}

fn get_max(start_i: u32) -> u32{
    var highest_i: u32 = 0;
    var highest_value: f32 = activities[start_i];

    for (var output_i:u32 = 1; output_i < nn_dir.n_outputs; output_i += 1){
        let value = activities[start_i + output_i];

        if value > highest_value{
            highest_i = output_i;
            highest_value = value;
        }
    }

    return highest_i;
} 

fn softmax(start_i: u32) -> array<f32, 32>{
    var softmax_prob: array<f32, 32>;

    var total_sum = 0.0;

    for (var output_i:u32 = 0; output_i < nn_dir.n_outputs; output_i += 1){
        let value = exp(activities[start_i + output_i]);

        total_sum += value;

        softmax_prob[output_i] = value;
    }

    for (var output_i:u32 = 0; output_i < nn_dir.n_outputs; output_i += 1){
        softmax_prob[output_i] /= total_sum;
    }

    return softmax_prob;
}