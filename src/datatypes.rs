use bytemuck::{Pod, Zeroable};

const lr: f32 = 0.1;

fn activation(z: f32) -> f32{
    return z;
}

fn deriv_activation(z: f32) -> f32{
    return 1.0;
}

fn get_tensor_size(tens_dim: &[usize]) -> usize{
    let mut size: usize = 1;

    for n in tens_dim{
        size *= n;
    } 

    return size;
}

fn get_tensor_strides(tens_dim: &Vec<usize>) -> Vec<usize>{
    let n_dim = tens_dim.len();
    
    let mut strides: Vec<usize> = vec![0; n_dim];

    for i in 0..(n_dim - 1){
        strides[i] = get_tensor_size(&tens_dim[(i + 1)..n_dim]);
    }

    strides[n_dim - 1] = 1;

    return strides;
}

fn get_activity_strides(a_dim: &Vec<usize>) -> Vec<usize>{
    let n_dim = a_dim.len();
    let mut strides: Vec<usize> = vec![0; n_dim];

    let mut curr_counter = 0;

    for i in 1..(n_dim){
        curr_counter += a_dim[i - 1];

        strides[i] = curr_counter;
    }

    return strides;
}

fn get_activity_length(a_strides: &Vec<usize>) -> usize{
    let mut l = 0;

    for stride in a_strides{
        l += stride;
    }

    return l;
}

// finds highest consecutive product
fn get_activity_deriv_buffer_size(a_dim: &Vec<usize>) -> usize{
    let n_dim = a_dim.len();

    let mut highest_product = 0;

    for i in 1..n_dim{
        let p = a_dim[i - 1] * a_dim[i];

        if p > highest_product{
            highest_product = p;
        }
    }

    return highest_product;
}

fn vec_dot(vec1: &Vec<usize>, vec2: &Vec<usize>) -> usize{
    let l1 = vec1.len();
    let l2 = vec2.len();

    let mut n : usize = 0;

    if l1 != l2{
        panic!("Vectors are not same length");
    }

    else{
        for i in 0..l1{
            n += vec1[i] * vec2[i];
        }
    }

    return n;
}

fn vec_dot_f(vec1: &[f32], vec2: &[f32]) -> f32{
    let l1 = vec1.len();
    let l2 = vec2.len();

    let mut n : f32 = 0.0;

    if l1 != l2{
        panic!("Vectors are not same length");
    }

    else{
        for i in 0..l1{
            n += vec1[i] * vec2[i];
        }
    }

    return n;
}

#[derive(Clone)]
struct TensorInfo{
    offset: usize,
    tens_dim: Vec<usize>,
    tens_n_dim: usize,
    tens_strides: Vec<usize>,
    tens_length: usize,
}

impl TensorInfo{
    pub fn null() -> Self{
        return TensorInfo{
            offset: 0,
            tens_dim: Vec::new(),
            tens_n_dim: 0,
            tens_strides: Vec::new(),
            tens_length: 0,
        };
    }
    
    pub fn new(tens_dim: &Vec<usize>) -> Self{
        return TensorInfo{
            offset: 0,
            tens_dim: tens_dim.clone(),
            tens_n_dim: tens_dim.len(),
            tens_strides: get_tensor_strides(tens_dim),
            tens_length: get_tensor_size(tens_dim),
        }
    }

    // start idx + slice length -> (mem_start, mem_end)
    pub fn get_slice(&self, idx_coord: &Vec<usize>, slice_length: usize) -> (usize, usize){
        let start_i = self.get_index(idx_coord);
        let end_i = start_i + slice_length;

        return (start_i, end_i);
    }

    pub fn get_index(&self, idx_coord: &Vec<usize>) -> usize{
        return vec_dot(&self.tens_strides, idx_coord) + self.offset;
    }
}

// Activity Info
// Consists of:
// 1. activity layers (array)
// 2. activity layer gradients (tensor)
// 3. deriv ping pong buffer 1 (array)
// 4. deriv ping pong buffer 2 (array)

#[derive(Clone)]
struct ActivityInfo{
    pub offset: usize,
    pub a_dim: Vec<usize>,
    pub a_strides: Vec<usize>,
    pub a_gradients: Vec<TensorInfo>,
    pub a_deriv_buffer_size: usize,
    pub a_length: usize,
    
    pub g_start: usize,
    pub d_start: usize,
}

impl ActivityInfo{
    pub fn null() -> Self{
        return ActivityInfo{
            offset: 0,
            a_dim: Vec::new(),
            a_strides: Vec::new(),
            a_gradients: Vec::new(),
            a_deriv_buffer_size: 0,
            a_length: 0,
            g_start: 0,
            d_start: 0,
        }
    }

    pub fn new(a_dim: &Vec<usize>) -> Self{
        let a_strides = get_activity_strides(a_dim);
        let mut a_length = get_activity_length(a_dim);
        let deriv_buffer_size = get_activity_deriv_buffer_size(a_dim);
        let mut gradient_info = Vec::new();

        let g_start = a_length;

        for layer_i in 0..(a_dim.len() - 1){

            let mut tensor_info_i = TensorInfo::new(&vec![1, a_dim[layer_i + 1], a_dim[layer_i] + 1]);

            tensor_info_i.offset = a_length;

            a_length += tensor_info_i.tens_length;

            gradient_info.push(tensor_info_i);
        }

        let d_start = a_length;

        a_length += deriv_buffer_size * 2;

        return ActivityInfo{
            offset: 0,
            a_dim: a_dim.clone(),
            a_strides: a_strides.clone(),
            a_gradients: gradient_info.clone(),
            a_deriv_buffer_size: deriv_buffer_size,
            a_length: a_length,
            g_start: g_start,
            d_start: d_start,
        }
    }

    pub fn get_index(&self, idx_coord: &Vec<usize>) -> usize{
        return self.a_strides[idx_coord[0]] + idx_coord[1] + self.offset;
    }

    pub fn get_slice(&self, start_coord: &Vec<usize>, slice_length: usize) -> (usize, usize) {
        let start_i = self.get_index(start_coord);

        return (start_i, start_i + slice_length);
    }
}


pub struct NeuralNetworkInfo{
    pub layer_dim: Vec<usize>,
    pub n_layers: usize,
    pub activity_info: ActivityInfo,
    pub layer_info: Vec<TensorInfo>,
    pub nn_length: usize,
    pub p_length: usize,
    pub n_batches: usize,
}

impl NeuralNetworkInfo{
    pub fn null() -> Self{
        return NeuralNetworkInfo{
            layer_dim: Vec::new(),
            n_layers: 0,
            activity_info: ActivityInfo::null(),
            layer_info: Vec::new(),
            nn_length: 0,
            p_length: 0,
            n_batches: 0,
        }
    }

    pub fn show_all_specs(&self){
        println!("\nMAIN MODEL:");
        println!("Layer Dim: {:?}", self.layer_dim);
        println!("N Batches: {:?}", self.n_batches);
        
        println!("\nACTIVITY:");
        println!("Activity Strides: {:?}", self.activity_info.a_strides);
        println!("Activity Gradients:");
        for g in &self.activity_info.a_gradients{
            println!(" - gradient dim: {:?}", g.tens_dim);
        }
        println!("Ping Pong Buffer length: {} floats", self.activity_info.a_deriv_buffer_size);
        println!("Total Buffer Length: {} floats", self.activity_info.a_length);

        println!("\nPARAMETERS:");
        println!("Param Dim + Offset:");
        for p in &self.layer_info{
            println!(" - {:?} {}", p.tens_dim, p.offset);
        }
        println!("Total Buffer Length: {} floats", self.p_length);
        println!("");
    }

    pub fn get_n_layers(&self) -> usize{
        return self.n_layers;
    }

    pub fn get_dim_n(&self, layer_i: usize) -> usize{
        return self.layer_dim[layer_i];
    }

    pub fn get_nn_length(&self) -> usize{
        return self.nn_length;
    }

    pub fn get_n_batches(&self) -> usize{
        return self.n_batches;
    }

    pub fn create_buffer(&self) -> Vec<f32>{
        let mut buf = vec![1.0; self.nn_length];
        
        return buf;
    }

    pub fn new(nn_dim: &Vec<usize>, n_batches: usize) -> Self{
        let n_layers = nn_dim.len();

        let mut offset = 0;
        let mut p_length = 0;

        // make activity layers
        let activity_info = ActivityInfo::new(nn_dim);

        // make info layers
        let mut layer_info = Vec::new();

        for layer_i in 0..(n_layers - 1){

            // create Tensor
            // idx 0: info / momentum 
            // idx 1: prev input layer
            // idx 2: new input layer + 1 (extra to store bias)

            let mut tensor_info_i = TensorInfo::new(&vec![1, nn_dim[layer_i + 1], nn_dim[layer_i] + 1]);

            tensor_info_i.offset = offset;

            offset += tensor_info_i.tens_length;
            p_length += tensor_info_i.tens_length;

            layer_info.push(tensor_info_i);
        }

        // make derivative layers

        return NeuralNetworkInfo{
            layer_dim: nn_dim.clone(),
            n_layers: n_layers,
            activity_info: activity_info,
            layer_info: layer_info,
            nn_length: offset,
            p_length: p_length,
            n_batches: n_batches,
        }
    }

    pub fn create_dirs(&self) -> (ParamsDir, ActivityDir){
        return (ParamsDir::new(self), ActivityDir::new(self));
    }

    pub fn read_readback(&self, param_slice: &[f32]){
        let mut layer_i = 0;
        for layer_info_i in &self.layer_info{
            
            println!("l{}->l{}: ", layer_i, layer_i + 1);

            let input_layer_n = self.layer_dim[layer_i];
            let output_layer_n = self.layer_dim[layer_i + 1];

            for output_i in 0..output_layer_n{
                let start_i = layer_info_i.offset + output_i * input_layer_n;
                let end_i = layer_info_i.offset + (output_i + 1)* input_layer_n;

                println!("{:?} {}", &param_slice[start_i..end_i], param_slice[end_i + 1]);

            }
            layer_i += 1;
        }
    }
}



#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ForwardDir {
    pub read_layer_start: u32,
    pub read_layer_length: u32,

    pub param_layer_start: u32,
    pub param_layer_length:u32,

    pub write_layer_start: u32,
    pub write_layer_length: u32,

    pub n_batches: u32,
    pub batch_act_size: u32,
}

impl ForwardDir{
    pub fn new(
        nn_info: &NeuralNetworkInfo,
        dir_i: usize,
    ) -> Self{
        ForwardDir{
            read_layer_start: nn_info.activity_info.a_strides[dir_i] as u32,
            read_layer_length: nn_info.activity_info.a_dim[dir_i] as u32,
            
            param_layer_start: nn_info.layer_info[dir_i].offset as u32,
            param_layer_length: nn_info.activity_info.a_dim[dir_i] as u32 + 1,

            write_layer_start: nn_info.activity_info.a_strides[dir_i + 1] as u32,
            write_layer_length: nn_info.activity_info.a_dim[dir_i + 1] as u32,

            n_batches: nn_info.n_batches as u32,
            batch_act_size: nn_info.activity_info.a_length as u32,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BackwardDir {
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
}

impl BackwardDir{
    pub fn new(
        nn_info: &NeuralNetworkInfo,
        dir_i: usize,
    ) -> Self{
        let next_layer_start: u32;
        let next_layer_length: u32;

        if dir_i == nn_info.n_layers - 2{
            next_layer_start = 0;
            next_layer_length = nn_info.activity_info.a_dim[dir_i + 1] as u32;
        }

        else{
            next_layer_start = nn_info.activity_info.a_strides[dir_i + 2] as u32;
            next_layer_length = nn_info.activity_info.a_dim[dir_i + 2] as u32;
        }

        let ping_switch = (dir_i + 1) % 2;
        let pong_switch = dir_i % 2;

        let ping_pong_default = nn_info.activity_info.a_length - nn_info.activity_info.a_deriv_buffer_size;

        BackwardDir{
            prev_layer_start: nn_info.activity_info.a_strides[dir_i] as u32,
            prev_layer_length: nn_info.activity_info.a_dim[dir_i] as u32,

            param_layer_start: nn_info.layer_info[dir_i].offset as u32,
            param_layer_length: nn_info.activity_info.a_dim[dir_i] as u32 + 1,

            curr_layer_start: nn_info.activity_info.a_strides[dir_i + 1] as u32,
            curr_layer_length: nn_info.activity_info.a_dim[dir_i + 1] as u32,

            next_layer_start: next_layer_start,
            next_layer_length: next_layer_length,

            ping_start : (ping_pong_default - nn_info.activity_info.a_deriv_buffer_size * ping_switch) as u32,
            pong_start : (ping_pong_default - nn_info.activity_info.a_deriv_buffer_size * pong_switch) as u32,

            gradients_start: nn_info.activity_info.g_start as u32,
            gradient_length: nn_info.activity_info.a_dim[dir_i] as u32 + 1,

            n_batches: nn_info.n_batches as u32,
            batch_act_size: nn_info.activity_info.a_length as u32,
        }
    }
}


pub struct ParamsDir{
    layer_dim: Vec<usize>,
    n_layers: usize,

    param_info: Vec<TensorInfo>,
    buffer_size: usize,
}

impl ParamsDir{
    pub fn new(nn_info: &NeuralNetworkInfo) -> Self{
        return ParamsDir{
            layer_dim: nn_info.layer_dim.clone(),
            n_layers: nn_info.layer_dim.len(),
            param_info: nn_info.layer_info.clone(),
            buffer_size: nn_info.p_length,
        }
    }

    pub fn create_buffer(&self) -> Vec<f32>{
        return vec![1.0; self.buffer_size];
    }
}

pub struct ActivityDir{
    layer_dim: Vec<usize>,
    n_layers: usize,
    n_batches: usize,

    activities_info: Vec<ActivityInfo>,
    buffer_size: usize,
}

impl ActivityDir{
    pub fn new(nn_info: &NeuralNetworkInfo) -> Self{
        let mut activities_info = Vec::new();

        for batch_i in 0..nn_info.n_batches{
            let mut a_info = nn_info.activity_info.clone();

            a_info.offset = a_info.a_length * batch_i;

            activities_info.push(a_info);
        }

        return ActivityDir{
            layer_dim: nn_info.layer_dim.clone(),
            n_layers: nn_info.layer_dim.len(),
            n_batches: nn_info.n_batches,
            activities_info: activities_info,
            buffer_size: nn_info.activity_info.a_length * nn_info.n_batches,
        }
    }

    pub fn create_buffer(&self) -> Vec<f32>{
        return vec![0.0; self.buffer_size];
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