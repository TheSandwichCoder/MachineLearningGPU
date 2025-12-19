use crate::data_reader::DataReader;
use crate::functions::get_mb;
use crate::model::ModelConstructor;
use crate::tensor::*;
use itertools::Itertools;
use rand::Rng;
use rand::distr::uniform::UniformUsize;

fn activation(z: f32) -> f32 {
    return z;
}

fn deriv_activation(z: f32) -> f32 {
    return 1.0;
}

fn get_activity_strides(a_dim: &Vec<usize>) -> Vec<usize> {
    let n_dim = a_dim.len();
    let mut strides: Vec<usize> = vec![0; n_dim];

    let mut curr_counter = 0;

    for i in 1..(n_dim) {
        curr_counter += a_dim[i - 1];

        strides[i] = curr_counter;
    }

    return strides;
}

fn get_activity_length(a_strides: &Vec<usize>) -> usize {
    let mut l = 0;

    for stride in a_strides {
        l += stride;
    }

    return l;
}

// finds highest consecutive product
fn get_activity_deriv_buffer_size(a_dim: &Vec<usize>) -> usize {
    let n_dim = a_dim.len();

    let mut highest_product = 0;

    for i in 1..n_dim {
        let p = a_dim[i - 1] * a_dim[i];

        if p > highest_product {
            highest_product = p;
        }
    }

    return highest_product;
}

fn get_vec_max(vec: &Vec<usize>) -> usize {
    let mut largest = 0;

    for n in vec {
        if *n > largest {
            largest = *n;
        }
    }

    return largest;
}

pub struct NNConstructor {
    pub nn_dim: Vec<usize>,
    pub n_batches: usize,
    pub lr: f32,
    pub mr: f32,
    pub vr: f32,
}

impl NNConstructor {
    pub fn default() -> Self {
        return NNConstructor {
            nn_dim: Vec::new(),
            n_batches: 0,
            lr: 0.0,
            mr: 0.0,
            vr: 0.0,
        };
    }

    pub fn from_model_constructor(model_constructor: &ModelConstructor) -> Self {
        let mut new_nn_dim = model_constructor.nn_dim.clone();

        return NNConstructor {
            nn_dim: new_nn_dim,
            n_batches: model_constructor.n_batches,
            lr: model_constructor.lr,
            mr: model_constructor.mr,
            vr: model_constructor.vr,
        };
    }

    pub fn update_input_n_dim(&mut self, input_dim: usize) {
        self.nn_dim[0] = input_dim;
    }

    pub fn set_dim(&mut self, dim: Vec<usize>) {
        self.nn_dim = dim;
    }

    pub fn set_n_batches(&mut self, n_batches: usize) {
        self.n_batches = n_batches;
    }

    pub fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }

    pub fn set_mr(&mut self, mr: f32) {
        self.mr = mr;
    }
}
// Activity Info
// Consists of:
// 1. activity ping pong buffer 1 (array)
// 2. activity ping pong buffer 2 (array)
// 3. pre-activition storage (array)
// 4. deriv ping pong buffer 1 (array)
// 5. deriv ping pong buffer 2 (array)

#[derive(Clone)]
pub struct NNActivityInfo {
    pub offset: usize,
    pub a_dim: Vec<usize>,
    pub a_strides: Vec<usize>,

    pub a_swap_buffer_size: usize,
    pub a_deriv_buffer_size: usize,
    pub a_length: usize,

    pub s_start: usize,
    pub d_start: usize,
}

impl NNActivityInfo {
    pub fn null() -> Self {
        return NNActivityInfo {
            offset: 0,
            a_dim: Vec::new(),
            a_strides: Vec::new(),

            a_swap_buffer_size: 0,
            a_deriv_buffer_size: 0,
            a_length: 0,

            s_start: 0, // activition storage
            d_start: 0, // derivative ping pong
        };
    }

    pub fn new(a_dim: &Vec<usize>) -> Self {
        let largest_layer = get_vec_max(a_dim);

        let mut t_length = 0;

        t_length += largest_layer * 2;

        let s_start = t_length;

        let a_strides = get_activity_strides(a_dim);
        t_length += get_activity_length(a_dim);

        let d_start = t_length;

        t_length += largest_layer * 2;

        return NNActivityInfo {
            offset: 0,
            a_dim: a_dim.clone(),
            a_strides: a_strides.clone(),

            a_swap_buffer_size: largest_layer,
            a_deriv_buffer_size: largest_layer,
            a_length: t_length,

            s_start: s_start,
            d_start: d_start,
        };
    }

    pub fn get_index(&self, idx_coord: &Vec<usize>) -> usize {
        return self.a_strides[idx_coord[0]] + idx_coord[1] + self.offset;
    }

    pub fn get_slice(&self, start_coord: &Vec<usize>, slice_length: usize) -> (usize, usize) {
        let start_i = self.get_index(start_coord);

        return (start_i, start_i + slice_length);
    }
}

pub struct NeuralNetworkInfo {
    pub layer_dim: Vec<usize>,
    pub n_layers: usize,
    pub activity_info: NNActivityInfo,
    pub layer_info: Vec<TensorInfo>,
    pub nn_length: usize,
    pub p_length: usize,
    pub n_batches: usize,
    pub lr: f32,
    pub mr: f32,
    pub vr: f32,
}

impl NeuralNetworkInfo {
    pub fn construct(nn_constructor: &NNConstructor) -> Self {
        return NeuralNetworkInfo::new(
            &nn_constructor.nn_dim,
            nn_constructor.n_batches,
            nn_constructor.lr,
            nn_constructor.mr,
            nn_constructor.vr,
        );
    }

    pub fn null() -> Self {
        return NeuralNetworkInfo {
            layer_dim: Vec::new(),
            n_layers: 0,
            activity_info: NNActivityInfo::null(),
            layer_info: Vec::new(),
            nn_length: 0,
            p_length: 0,
            n_batches: 0,
            lr: 0.0,
            mr: 0.0,
            vr: 0.0,
        };
    }

    pub fn show_all_specs(&self) {
        println!("\nNEURAL NETWORK");
        println!("Layer Dim: {:?}", self.layer_dim);
        println!("N Batches: {:?}", self.n_batches);

        println!("\nACTIVITY:");
        println!("Activity Strides: {:?}", self.activity_info.a_strides);
        println!(
            "Ping Pong Buffer length: {} floats ({}MB)",
            self.activity_info.a_deriv_buffer_size,
            get_mb(self.activity_info.a_deriv_buffer_size)
        );
        println!(
            "Total Buffer Length: {} floats ({}MB)",
            self.activity_info.a_length,
            get_mb(self.activity_info.a_length)
        );

        println!("\nPARAMETERS:");
        println!("Param Dim + Offset:");
        for p in &self.layer_info {
            println!(" - {:?} {}", p.tens_dim, p.offset);
        }
        println!(
            "Total Buffer Length: {} floats ({}MB)",
            self.p_length,
            get_mb(self.p_length)
        );
        println!("");
    }

    pub fn get_n_layers(&self) -> usize {
        return self.n_layers;
    }

    pub fn get_dim_n(&self, layer_i: usize) -> usize {
        return self.layer_dim[layer_i];
    }

    pub fn get_nn_length(&self) -> usize {
        return self.nn_length;
    }

    pub fn get_n_batches(&self) -> usize {
        return self.n_batches;
    }

    pub fn create_buffer(&self) -> Vec<f32> {
        let mut buf = vec![1.0; self.nn_length];

        return buf;
    }

    pub fn load_param_buffer(&self, file_string: &str) -> Vec<f32> {
        let mut param_buffer: Vec<f32> = Vec::new();

        let file_lines: Vec<&str> = file_string.trim().split("\n").collect();
        let mut curr_line = 0;

        // skip nn dim
        curr_line += 1;

        for layer_i in 0..self.n_layers - 1 {
            let n_inputs = self.layer_dim[layer_i];
            let n_outputs = self.layer_dim[layer_i + 1];

            for _output_layer_i in 0..n_outputs {
                let layer_weights: Vec<&str> = file_lines[curr_line].trim().split(" ").collect();

                for w_i in 0..n_inputs {
                    param_buffer.push(layer_weights[w_i].parse().unwrap());
                }

                param_buffer.push(layer_weights[n_inputs].parse().unwrap());

                curr_line += 1;
            }
        }

        return param_buffer;
    }

    pub fn new(
        nn_dim: &Vec<usize>,
        n_batches: usize,
        learning_rate: f32,
        momentum_rate: f32,
        variance_rate: f32,
    ) -> Self {
        let n_layers = nn_dim.len();

        let mut offset = 0;
        let mut p_length = 0;

        // make activity layers
        let activity_info = NNActivityInfo::new(nn_dim);

        // make info layers
        let mut layer_info = Vec::new();

        for layer_i in 0..(n_layers - 1) {
            // create Tensor
            // idx 0: info / momentum
            // idx 1: prev input layer
            // idx 2: new input layer + 1 (extra to store bias)

            let mut tensor_info_i =
                TensorInfo::new(&vec![1, nn_dim[layer_i + 1], nn_dim[layer_i] + 1]);

            tensor_info_i.offset = offset;

            offset += tensor_info_i.tens_length;
            p_length += tensor_info_i.tens_length;

            layer_info.push(tensor_info_i);
        }

        // make derivative layers

        return NeuralNetworkInfo {
            layer_dim: nn_dim.clone(),
            n_layers: n_layers,
            activity_info: activity_info,
            layer_info: layer_info,
            nn_length: offset,
            p_length: p_length,
            n_batches: n_batches,
            lr: learning_rate,
            mr: momentum_rate,
            vr: variance_rate,
        };
    }

    pub fn create_dirs(&self) -> (ParamsDir, ActivityDir) {
        return (ParamsDir::new(self), ActivityDir::new(self));
    }

    pub fn read_readback(&self, param_slice: &[f32]) {
        let mut layer_i = 0;
        for layer_info_i in &self.layer_info {
            println!("l{}->l{}: ", layer_i, layer_i + 1);

            let input_layer_n = self.layer_dim[layer_i];
            let output_layer_n = self.layer_dim[layer_i + 1];

            for output_i in 0..output_layer_n {
                let start_i = layer_info_i.offset + output_i * (input_layer_n + 1);
                let end_i = start_i + input_layer_n;

                println!("{:?} {}", &param_slice[start_i..end_i], param_slice[end_i]);
            }
            layer_i += 1;
        }
    }

    pub fn get_save_string(&self, param_slice: &[f32]) -> String {
        let mut layer_i = 0;
        let mut save_string = String::from("");

        for layer_i in 0..self.n_layers {
            save_string += &format!("{} ", self.layer_dim[layer_i]);
        }

        save_string += "\n";

        for layer_info_i in &self.layer_info {
            let input_layer_n = self.layer_dim[layer_i];
            let output_layer_n = self.layer_dim[layer_i + 1];

            for output_i in 0..output_layer_n {
                let start_i = layer_info_i.offset + output_i * (input_layer_n + 1);
                let end_i = start_i + input_layer_n;

                // println!("{:?} {}", &param_slice[start_i..end_i], param_slice[end_i]);

                save_string += &format!(
                    "{} {}\n",
                    param_slice[start_i..end_i].iter().join(" "),
                    param_slice[end_i]
                );
            }
            layer_i += 1;
        }

        return save_string;
    }
}

pub struct ParamsDir {
    pub layer_dim: Vec<usize>,
    pub n_layers: usize,

    pub param_info: Vec<TensorInfo>,
    pub buffer_size: usize,
}

impl ParamsDir {
    pub fn new(nn_info: &NeuralNetworkInfo) -> Self {
        return ParamsDir {
            layer_dim: nn_info.layer_dim.clone(),
            n_layers: nn_info.layer_dim.len(),
            param_info: nn_info.layer_info.clone(),
            buffer_size: nn_info.p_length,
        };
    }

    pub fn create_buffer_empty(&self) -> Vec<f32> {
        return vec![0.0; self.buffer_size];
    }

    pub fn create_buffer(&self) -> Vec<f32> {
        // let mut out: Vec<f32> = Vec::new();

        // for i in 0..self.buffer_size {
        //     out.push(i as f32 / 1000.0);
        // }

        // return out;

        let mut rng = rand::thread_rng();

        let random_floats: Vec<f32> = (0..self.buffer_size)
            .map(|_| rng.gen_range(-1.0..=1.0))
            .collect();

        return random_floats;
    }
}

pub struct ActivityDir {
    layer_dim: Vec<usize>,
    n_layers: usize,
    n_batches: usize,

    activities_info: Vec<NNActivityInfo>,
    buffer_size: usize,
    single_buffer_size: usize,
}

impl ActivityDir {
    pub fn new(nn_info: &NeuralNetworkInfo) -> Self {
        let mut activities_info = Vec::new();

        for batch_i in 0..nn_info.n_batches {
            let mut a_info = nn_info.activity_info.clone();

            a_info.offset = a_info.a_length * batch_i;

            activities_info.push(a_info);
        }

        return ActivityDir {
            layer_dim: nn_info.layer_dim.clone(),
            n_layers: nn_info.layer_dim.len(),
            n_batches: nn_info.n_batches,
            activities_info: activities_info,
            buffer_size: nn_info.activity_info.a_length * nn_info.n_batches,
            single_buffer_size: nn_info.activity_info.a_length,
        };
    }

    pub fn create_buffer(&self) -> Vec<f32> {
        let mut v = vec![0.0; self.buffer_size];

        // for n in 0..self.n_batches{
        //     for i in 0..12{
        //         v[n*self.single_buffer_size + 24 + i] = -1.0;
        //     }
        // }

        return v;
    }
}

// "
// make nn struct

// (
// nn struct
//     - n layers
//     - using the layers I make:
//         - activity buffer (1d buffer? needs mapping)
//         - n tensors (info)
// )

// make directory

// (
// directory (nn struct)
//     - info
//         - mapping for the tensors
//             - takes in layer_n, gives the offset and length for the tensor

//         - get length

//     - activity
//         - mapping for the 1d buffer
//             - takes in layer_n, gives offset and length for activity buffer binding?

//         - get length
// )

// make info heap (directory.info_length)
// make activity heap (directory.activity_length)

// do all the bindings?

// forward prop

// set input layer

// for layer_i in layers:
//     layer1, layer2 = directory.get_layers(layer_i)
//     compute(layer1, layer2)

// backwards prop

// for layer_i in layers:
//     layer1, layer2 = directory.get_layers(layer_i)
//     compute(layer1, layer2)

// "
/*

Convolution layer
Params:
h - height
w - width
c - channels
k - convolution size
n - filters


Idea:

with w, h, c we want to make a process that dispatches 1 thread for each of the values in the hxwxn outputs.
although we have c channels for the input, each filter only produces a layer of h x w x 1. and since we have n filters,
our next layer will be w x h x n

so we dispatch w x h x n threads (each for one pixel of the output conv).
with these thread there will be a 8 x 8 x 4 thread group (meaning all these threads will read from the same cached buffer)
This means that we can have a cache buffer size of 8x8x4, which will also slide over the image with t_k. Then the kernals that actually use the values
in the cache will read from the cache to update their values.



workgroup_dim()


*/
