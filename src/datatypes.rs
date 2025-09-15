use bytemuck::{Pod, Zeroable};
use itertools::Itertools;
use crate::data_reader::DataReader;
use rand::Rng;

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

fn get_vec_max(vec: &Vec<usize>) -> usize{
    let mut largest = 0;

    for n in vec{
        if *n > largest{
            largest = *n;
        }
    }

    return largest;
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
// 1. activity ping pong buffer 1 (array)
// 2. activity ping pong buffer 2 (array)
// 3. pre-activition storage (array)
// 4. deriv ping pong buffer 1 (array)
// 5. deriv ping pong buffer 2 (array)

#[derive(Clone)]
pub struct ActivityInfo{
    pub offset: usize,
    pub a_dim: Vec<usize>,
    pub a_strides: Vec<usize>,

    pub a_swap_buffer_size: usize,
    pub a_deriv_buffer_size: usize,
    pub a_length: usize,
    
    pub s_start: usize,
    pub d_start: usize,
}

impl ActivityInfo{
    pub fn null() -> Self{
        return ActivityInfo{
            offset: 0,
            a_dim: Vec::new(),
            a_strides: Vec::new(),

            a_swap_buffer_size: 0,
            a_deriv_buffer_size: 0,
            a_length: 0,

            s_start: 0, // activition storage
            d_start: 0, // derivative ping pong
        }
    }

    pub fn new(a_dim: &Vec<usize>) -> Self{
        let largest_layer = get_vec_max(a_dim);

        let mut t_length = 0;

        t_length += largest_layer * 2;

        let s_start = t_length;

        let a_strides = get_activity_strides(a_dim);
        t_length += get_activity_length(a_dim);

        let d_start = t_length;

        t_length += largest_layer * 2;

        return ActivityInfo{
            offset: 0,
            a_dim: a_dim.clone(),
            a_strides: a_strides.clone(),

            a_swap_buffer_size: largest_layer,
            a_deriv_buffer_size: largest_layer,
            a_length: t_length,

            s_start: s_start,
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
    pub lr: f32,
    pub mr: f32,
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
            lr: 0.0,
            mr: 0.0, 
        }
    }

    pub fn show_all_specs(&self){
        println!("\nMAIN MODEL:");
        println!("Layer Dim: {:?}", self.layer_dim);
        println!("N Batches: {:?}", self.n_batches);
        
        println!("\nACTIVITY:");
        println!("Activity Strides: {:?}", self.activity_info.a_strides);
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

    pub fn new(nn_dim: &Vec<usize>, n_batches: usize, learning_rate: f32, momentum_rate: f32) -> Self{
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
            lr: learning_rate,
            mr: momentum_rate,
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
                let start_i = layer_info_i.offset + output_i * (input_layer_n + 1);
                let end_i = start_i + input_layer_n;

                println!("{:?} {}", &param_slice[start_i..end_i], param_slice[end_i]);

            }
            layer_i += 1;
        }
    }

    pub fn get_save_string(&self, param_slice: &[f32]) -> String{
        let mut layer_i = 0;
        let mut save_string = String::from("");

        for layer_i in 0..self.n_layers{
            save_string += &format!("{} ", self.layer_dim[layer_i]);
        }

        save_string += "\n";

        for layer_info_i in &self.layer_info{
            

            let input_layer_n = self.layer_dim[layer_i];
            let output_layer_n = self.layer_dim[layer_i + 1];

            for output_i in 0..output_layer_n{
                let start_i = layer_info_i.offset + output_i * (input_layer_n + 1);
                let end_i = start_i + input_layer_n;

                // println!("{:?} {}", &param_slice[start_i..end_i], param_slice[end_i]);

                save_string += &format!("{} {}\n", param_slice[start_i..end_i].iter().join(" "), param_slice[end_i]);

            }
            layer_i += 1;
        }

        return save_string;
    }


}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct FlatApplyDir{
    batch_start_i: u32,
    batch_i: u32,
    gradient_start_i: u32,
    batch_contribution: f32,
    n_params: u32,
    lr: f32,
    mr: f32,
}

impl FlatApplyDir{
    pub fn new(nn_info : &NeuralNetworkInfo, dir_i: usize) -> Self{
        return FlatApplyDir{
            batch_start_i: nn_info.activity_info.a_length as u32 * dir_i as u32,
            batch_i: dir_i as u32,
            gradient_start_i: 0,
            batch_contribution: 1.0 / nn_info.n_batches as f32,
            n_params: nn_info.p_length as u32,
            lr: nn_info.lr,
            mr: nn_info.mr,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ErrorDir{
    act_size: u32,
    outputs_offset: u32,
    n_outputs: u32,
    ping_start: u32,
    data_size: u32,
}

impl ErrorDir{
    pub fn new(nn_info: &NeuralNetworkInfo) -> Self{
        let last_layer_i = nn_info.n_layers - 1;

        let ping_switch = (last_layer_i - 1) % 2;
        
        return ErrorDir{
            act_size: nn_info.activity_info.a_length as u32,
            outputs_offset: nn_info.activity_info.s_start as u32 + nn_info.activity_info.a_strides[last_layer_i] as u32,
            n_outputs: nn_info.activity_info.a_dim[last_layer_i] as u32,
            ping_start: (nn_info.activity_info.d_start + ping_switch * nn_info.activity_info.a_deriv_buffer_size) as u32,
            data_size: nn_info.layer_dim[0] as u32, // temporary
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ErrorPC{
    layer_idx: u32,
    n_batches: u32,
    _pad2: u32,
    _pad3: u32,
}

impl ErrorPC{
    pub fn new(dr: &DataReader, n_batches: usize) -> Self{
        return ErrorPC{
            layer_idx: (dr.load_batch_length * (dr.data_value_size + 1) * dr.load_batch_i + dr.sub_batch_length * (dr.data_value_size + 1) * dr.sub_batch_i) as u32,
            n_batches: n_batches as u32,
            _pad2: 0,
            _pad3: 0,   
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct TestMetrics{
    correct: u32,
    incorrect: u32,
    _pad1: u32,
    _pad2: u32,
}

impl TestMetrics{
    pub fn zero() -> Self{
        return TestMetrics{
             correct: 0,
             incorrect: 0,
             _pad1: 0,
             _pad2: 0,
        }
    }
}

fn make_transpose(n: bool, m: bool, o: bool) -> u32{
    let mut v : u32 = 0;

    if n{
        v |= 1 << 0;
    }
    if m{
        v |= 1 << 1;
    }
    if o{
        v |= 1 << 2;
    }

    return v;
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct MatrixDir{
    n_read_start: u32,
    m_read_start: u32,

    n_stride_length: u32,
    m_stride_length: u32,
    
    w_start: u32,
    w_stride_length: u32,

    add_const: u32,
    c_start: u32,
    c_stride_length: u32,
    a_func_type: u32,

    extra: u32,
    e_start: u32,
    e_stride_length: u32,    

    transpose: u32,

    n: u32,
    m: u32,
    k: u32
}

impl MatrixDir{
    /*
    new forward: 
    n: params/weights
    m: activity
    w: activity
    */
    pub fn null() -> Self{
        MatrixDir{
            n_read_start: 0,
            m_read_start: 0,

            n_stride_length: 0,
            m_stride_length: 0,
            
            w_start: 0,
            w_stride_length: 0,

            add_const: 0,
            c_start: 0,
            c_stride_length: 0,
            a_func_type: 0,

            extra: 0,
            e_start: 0,
            e_stride_length: 0,    

            transpose: 0,

            n: 0,
            m: 0,
            k: 0,
        }
    }

    pub fn new_forward(nn_info: &NeuralNetworkInfo, dir_i: usize) -> Self{
        let ping_switch = dir_i % 2;
        let pong_switch = (dir_i + 1) % 2;

        return MatrixDir{
            n_read_start: nn_info.layer_info[dir_i].offset as u32,
            m_read_start: nn_info.activity_info.a_swap_buffer_size as u32 * ping_switch as u32,

            n_stride_length: (nn_info.layer_dim[dir_i] + 1) as u32,
            m_stride_length: nn_info.activity_info.a_length as u32,

            w_start: nn_info.activity_info.a_swap_buffer_size as u32 * pong_switch as u32,
            w_stride_length: nn_info.activity_info.a_length as u32,

            add_const: 1,
            c_start: (nn_info.layer_info[dir_i].offset + nn_info.layer_dim[dir_i])  as u32,
            c_stride_length: (nn_info.layer_dim[dir_i] + 1) as u32,
            a_func_type: 1,

            extra: 1,
            e_start: nn_info.activity_info.s_start as u32 + nn_info.activity_info.a_strides[dir_i + 1] as u32,
            e_stride_length: nn_info.activity_info.a_length as u32,

            transpose: make_transpose(false, false, false),

            n: nn_info.layer_dim[dir_i + 1] as u32,
            m: nn_info.n_batches as u32,
            k: nn_info.layer_dim[dir_i] as u32,
        }
    }

    pub fn new_backward_deriv(nn_info: &NeuralNetworkInfo, dir_i: usize) -> Self{
        let ping_switch = dir_i % 2;
        let pong_switch = (dir_i + 1) % 2;

        let ping_pong_default = nn_info.activity_info.d_start;

        let n_start: u32;
        if dir_i >= nn_info.n_layers - 1{
            return MatrixDir::null();
        }


        return MatrixDir{            
            n_read_start: nn_info.layer_info[dir_i].offset as u32,
            m_read_start: (ping_pong_default + nn_info.activity_info.a_deriv_buffer_size * ping_switch) as u32,

            n_stride_length: (nn_info.layer_dim[dir_i] + 1) as u32,
            m_stride_length: nn_info.activity_info.a_length as u32,

            w_start : (ping_pong_default + nn_info.activity_info.a_deriv_buffer_size * pong_switch) as u32,
            w_stride_length: nn_info.activity_info.a_length as u32,

            add_const: 0,
            c_start: 0,
            c_stride_length: 0,
            a_func_type: 2,

            extra: 2,
            e_start: nn_info.activity_info.s_start as u32 + nn_info.activity_info.a_strides[dir_i] as u32,
            e_stride_length: nn_info.activity_info.a_length as u32,

            transpose: make_transpose(true, false, false),

            n: nn_info.layer_dim[dir_i] as u32,
            m: nn_info.n_batches as u32,
            k: nn_info.layer_dim[dir_i + 1] as u32,
        }
    }

    pub fn new_backward_gradients(nn_info: &NeuralNetworkInfo, dir_i: usize) -> Self{
        // have to add 1 to compensate
        let ping_switch = (dir_i + 1) % 2;
        let pong_switch = (dir_i + 1 + 1) % 2;

        let ping_pong_default = nn_info.activity_info.d_start;

        return MatrixDir{
            n_read_start: (ping_pong_default + nn_info.activity_info.a_deriv_buffer_size * pong_switch) as u32,
            m_read_start: nn_info.activity_info.s_start as u32 + nn_info.activity_info.a_strides[dir_i] as u32,

            n_stride_length: nn_info.activity_info.a_length as u32,
            m_stride_length: nn_info.activity_info.a_length as u32,

            w_start: nn_info.layer_info[dir_i].offset as u32,
            w_stride_length: (nn_info.layer_dim[dir_i] + 1) as u32,

            add_const: 0,
            c_start: 0,
            c_stride_length: 0,
            a_func_type: 0,

            extra: 2,
            e_start: 0,
            e_stride_length: 0,

            transpose: make_transpose(true, true, true),

            n: nn_info.layer_dim[dir_i + 1] as u32,
            m: nn_info.layer_dim[dir_i] as u32,
            k: nn_info.n_batches as u32,
        }
    }

    
    // pub fn new(nn_info : &NeuralNetworkInfo, dir_i: usize) -> Self{
    //     return MatrixDir{
    //         act_length: nn_info.activity_info.a_length as u32,
    //         n_start: nn_info.activity_info.a_strides[dir_i] as u32,
    //         m_start: nn_info.layer_info[dir_i].offset as u32,
    //         w_start: nn_info.activity_info.a_strides[dir_i + 1] as u32,
    //         n: 1,
    //         m: nn_info.layer_dim[dir_i + 1] as u32,
    //         k: nn_info.layer_dim[dir_i] as u32,
            
    //     }
    // }
}


pub struct ParamsDir{
    pub layer_dim: Vec<usize>,
    pub n_layers: usize,

    pub param_info: Vec<TensorInfo>,
    pub buffer_size: usize,
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

    pub fn create_buffer_empty(&self) -> Vec<f32>{
        return vec![0.0; self.buffer_size];
    }

    pub fn create_buffer(&self) -> Vec<f32>{
        // let mut out : Vec<f32> = Vec::new();

        // for i in 0..self.buffer_size{
        //     out.push(i as f32);
        // }

        // return out;

        let mut rng = rand::thread_rng();

        let random_floats: Vec<f32> = (0..self.buffer_size)
            .map(|_| rng.gen_range(-1.0..=1.0))
            .collect();
    

        return random_floats;
    }

}

pub struct ActivityDir{
    layer_dim: Vec<usize>,
    n_layers: usize,
    n_batches: usize,

    activities_info: Vec<ActivityInfo>,
    buffer_size: usize,
    single_buffer_size: usize,
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
            single_buffer_size: nn_info.activity_info.a_length,
        }
    }

    pub fn create_buffer(&self) -> Vec<f32>{
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