use rand::Rng;

use crate::functions::*;
use crate::range::*;
use crate::tensor::*;

/*
IMPORTANT TO NOTE
all info dimensions are
[x y z]
but in the tensor it needs to be REVERSED
[z y x]

layer Dim:
0 - l_x
1 - l_y
2 - number of input channels

Conv Dim:
0 - k_x
1 - k_y
2 - cin_in (number of input channels)
3 - c_out (number of conv layers / number of output channels)

Pool Dim:
0 - p_x
1 - p_y

*/

// CONSTRUCTOR
#[derive(Clone)]
pub struct ConvolutionConstructor {
    pub i_x: usize,
    pub i_y: usize,
    pub i_c: usize,
    pub n_layers: usize,
    pub n_batches: usize,
    pub split_k: usize,

    pub pooling_dim: Vec<usize>,
    pub kernal_dim: Vec<usize>,
    pub layer_output: Vec<usize>,
}

impl ConvolutionConstructor {
    pub fn default() -> Self {
        return ConvolutionConstructor {
            i_x: 0,
            i_y: 0,
            i_c: 0,
            n_layers: 0,
            n_batches: 0,
            split_k: 16,

            pooling_dim: Vec::new(),
            kernal_dim: Vec::new(),
            layer_output: Vec::new(),
        };
    }

    pub fn set_inputs(&mut self, input_dim: Vec<usize>, n_layers: usize) {
        self.i_x = input_dim[0];
        self.i_y = input_dim[1];
        self.i_c = input_dim[2];

        self.n_layers = n_layers;
    }

    pub fn set_pool(&mut self, pool_dim: Vec<usize>) {
        self.pooling_dim = pool_dim;
    }

    pub fn set_kernal(&mut self, kernal_dim: Vec<usize>) {
        self.kernal_dim = kernal_dim;
    }

    pub fn set_outputs(&mut self, layer_output: Vec<usize>) {
        self.layer_output = layer_output;
    }

    pub fn set_n_batches(&mut self, n_batches: usize) {
        self.n_batches = n_batches;
    }
}

#[derive(Clone)]
pub struct ConvolutionInfo {
    pub conv_layers: Vec<ConvLayerInfo>,
    pub param_info: ConvParamInfo,
    pub activity_info: ConvActivityInfo,
    pub input_layer_dim: Vec<usize>,
    pub output_layer_dim: Vec<usize>,
    pub n_layers: usize,
    pub n_batches: usize,
    pub split_k: usize,
}

impl ConvolutionInfo {
    pub fn construct(constructor: ConvolutionConstructor) -> Self {
        let mut conv_layers = Vec::new();

        let mut prev_x = constructor.i_x;
        let mut prev_y = constructor.i_y;
        let mut prev_c = constructor.i_c;

        let mut output_layer_x = 0;
        let mut output_layer_y = 0;
        let mut output_layer_c = 0;

        for layer_i in 0..constructor.n_layers {
            let pool_k: usize;
            let kernal_k: usize;
            let output_n: usize;

            let mut temp_x = 0;
            let mut temp_y = 0;
            let mut temp_c = 0;

            if layer_i == constructor.n_layers - 1 {
                kernal_k = 0;
                pool_k = 0;
                output_n = 0;

                output_layer_x = prev_x;
                output_layer_y = prev_y;
                output_layer_c = prev_c;
            } else {
                kernal_k = constructor.kernal_dim[layer_i];
                pool_k = constructor.pooling_dim[layer_i];
                output_n = constructor.layer_output[layer_i];
                temp_x = ceil_div(prev_x, pool_k);
                temp_y = ceil_div(prev_y, pool_k);
                temp_c = output_n;
            }

            let layer_info = ConvLayerInfo::new(
                vec![prev_x, prev_y, prev_c],
                vec![kernal_k, kernal_k, prev_c],
                output_n,
                pool_k,
                constructor.split_k,
                constructor.n_batches,
            );

            prev_x = temp_x;
            prev_y = temp_y;
            prev_c = temp_c;

            conv_layers.push(layer_info);
        }

        let activity_info = ConvActivityInfo::new(&conv_layers, constructor.n_batches);
        let param_info = ConvParamInfo::new(&conv_layers, constructor.split_k);

        return ConvolutionInfo {
            conv_layers: conv_layers,
            activity_info: activity_info,
            param_info: param_info,
            input_layer_dim: vec![constructor.i_x, constructor.i_y, constructor.i_c],
            output_layer_dim: vec![output_layer_x, output_layer_y, output_layer_c],
            n_layers: constructor.n_layers,
            n_batches: constructor.n_batches,
            split_k: constructor.split_k,
        };
    }

    pub fn show_all_specs(&self) {
        println!("\nCONVOLUTION LAYERS");

        println!("Input layer Dim");
        println!("{:?}", self.input_layer_dim);

        println!("Output layer Dim");
        println!("{:?} \n", self.output_layer_dim);

        println!("Convolutional layer Dim");
        for conv_layer_i in 0..self.n_layers {
            println!(
                "- {:?} {} floats",
                self.conv_layers[conv_layer_i].layer_dim,
                get_vec_product(&self.conv_layers[conv_layer_i].layer_dim)
            );
        }

        println!("Kernals");

        for kernal_layer_i in 0..self.n_layers - 1 {
            println!(
                "- {} {:?}",
                self.conv_layers[kernal_layer_i].n_kernals,
                self.conv_layers[kernal_layer_i].kernal_info.dim
            );
        }

        println!("\nACTIVITY INFO");
        println!("Activity Strides: {:?}", self.activity_info.strides);
        println!(
            "Deriv Swap Buffer length: {} floats",
            self.activity_info.deriv_buffer_size * 2
        );
        println!(
            "Output Swap Buffer length: {} floats",
            self.activity_info.swap_buffer_size * 2
        );
        println!(
            "Output Storage Buffer length: {} floats",
            self.activity_info.storage_buffer_size
        );
        println!(
            "Total Buffer Length: {} floats",
            self.activity_info.deriv_buffer_size * 2
                + self.activity_info.swap_buffer_size * 2
                + self.activity_info.storage_buffer_size
        );

        println!("\nPARAM INFO");

        println!("Param Kernal Strides: {:?}", self.param_info.k_strides);
        println!("Param Bias Strides: {:?}", self.param_info.b_strides);
        println!("Total Buffer Length: {} floats", self.param_info.size);
    }
}

// LAYER INFO
#[derive(Clone)]
pub struct ConvLayerInfo {
    pub layer_dim: Vec<usize>,
    pub layer_offset: Vec<i32>,

    pub layer_size_2d: usize,
    pub layer_size: usize,
    pub acc_length: usize,

    pub n_kernals: usize,
    pub kernal_info: KernalInfo,
    pub pooling_info: PoolingInfo,
}

impl ConvLayerInfo {
    pub fn new(
        layer_dim: Vec<usize>,
        kernal_dim: Vec<usize>,
        n_kernals: usize,
        pool_size: usize,
        split_k: usize,
        n_batches: usize,
    ) -> Self {
        let mut size = 0;

        let mut kernal_info = KernalInfo::new(kernal_dim);

        let pooling_info = PoolingInfo::new(&layer_dim, pool_size);
        let layer_size_2d = layer_dim[0] * layer_dim[1];
        let layer_size = layer_dim[0] * layer_dim[1] * layer_dim[2];

        let acc_length = ceil_div(layer_size_2d * n_batches, split_k);

        let layer_offset = vec![
            -(floor_div(layer_dim[0], 2) as i32),
            -(floor_div(layer_dim[1], 2) as i32),
            0,
        ];

        return ConvLayerInfo {
            layer_dim: layer_dim,
            layer_offset: layer_offset,

            layer_size_2d: layer_size_2d,
            layer_size: layer_size,
            acc_length: acc_length,

            n_kernals: n_kernals,
            kernal_info: kernal_info,
            pooling_info: pooling_info,
        };
    }
}

// POOLING INFO
#[derive(Clone)]
pub struct PoolingInfo {
    pub dim: Vec<usize>,
    pub k: usize,
}

impl PoolingInfo {
    pub fn new(pool_dim: &Vec<usize>, pool_size: usize) -> Self {
        return PoolingInfo {
            dim: pool_dim.clone(),
            k: pool_size,
        };
    }
}

// KERNAL INFO
#[derive(Clone)]
pub struct KernalInfo {
    pub dim: Vec<usize>,
    pub tens: TensorInfo,
    pub k_range: KernalRange,
    pub size: usize,
    pub size_2d: usize,
}

impl KernalInfo {
    pub fn new(kernal_dim: Vec<usize>) -> Self {
        let mut reversed = kernal_dim.clone();
        reversed.reverse();

        let k = kernal_dim[0];
        let c = kernal_dim[2];

        let k_range = KernalRange::new(-(floor_div(k, 2) as i32), ceil_div(k, 2) as i32);

        let tens = TensorInfo::new(&reversed);
        let size = tens.tens_length;

        return KernalInfo {
            dim: kernal_dim,
            tens: tens,
            k_range: k_range,
            size: size,
            size_2d: k * k,
        };
    }
}

// ACTIVITY INFO
/*
Swap Buffer
Storage Buffer (preactivation values)
Deriv Buffer
*/

#[derive(Clone)]
pub struct ConvActivityInfo {
    pub offset: usize,
    pub n_batches: usize,

    pub dim: Vec<TensorInfo>,
    pub strides: Vec<usize>,

    pub batch_swap_buffer_size: usize,
    pub swap_buffer_size: usize,
    pub deriv_buffer_size: usize,

    pub storage_buffer_size: usize,
}

impl ConvActivityInfo {
    pub fn new(layer_info: &Vec<ConvLayerInfo>, n_batches: usize) -> Self {
        let largest_layer = get_layer_max(layer_info);

        let mut layer_dim = Vec::new();

        let swap_buffer_size = largest_layer * n_batches;
        println!("asdkjfhsdakjfh {}", largest_layer);
        // storage buffer
        let mut stride_offset = 0;
        let mut stride_offset_batch = 0;

        let mut strides = Vec::new();

        let mut t_length = 0;
        for layer in layer_info {
            let mut reversed = layer.layer_dim.clone();
            reversed.reverse();
            let tens_info = TensorInfo::new(&reversed);

            let temp_stride_increment = tens_info.tens_length * n_batches;

            strides.push(stride_offset);
            layer_dim.push(tens_info);

            stride_offset += temp_stride_increment;
        }

        t_length += stride_offset;

        return ConvActivityInfo {
            offset: 0,
            n_batches,

            dim: layer_dim,
            strides: strides,

            batch_swap_buffer_size: largest_layer,
            swap_buffer_size: swap_buffer_size,
            deriv_buffer_size: swap_buffer_size,

            storage_buffer_size: t_length,
        };
    }

    // initial values in layer 1
    pub fn create_output_swap_buffer(&self) -> Vec<f32> {
        let mut empty = vec![0.0; self.swap_buffer_size * 2];

        for i in 0..self.swap_buffer_size {
            // empty[i] = (i % self.batch_swap_buffer_size + i / self.batch_swap_buffer_size) as f32;
            empty[i] = 1.0;
        }

        return empty;
    }

    // final deriv influence
    pub fn create_deriv_swap_buffer(&self) -> Vec<f32> {
        // return vec![1.0; self.swap_buffer_size * 2];
        let mut empty_vec = vec![1.0; self.swap_buffer_size * 2];

        for i in 0..self.swap_buffer_size * 2 {
            // empty_vec[i] = 1.0 + (i) as f32 * 0.01;
            empty_vec[i] = 1.0 + (i % self.batch_swap_buffer_size) as f32 * 0.01;
            // empty_vec[i] = 1.0;
        }

        return empty_vec;
    }

    pub fn create_output_storage_buffer(&self) -> Vec<f32> {
        let mut empty = vec![1.0; self.storage_buffer_size];

        // println!("asjkghkjdsafhksajdf {}", self.dim[0].tens_length);

        // for i in 0..self.storage_buffer_size {
        //     empty[i] = (i % 784) as f32;
        //     // empty[i] = 1 as f32;
        // }

        // for i in d_start..self.size{
        //     empty[i] = 1.0;
        // }

        // println!("{:?}", &empty[0..784]);

        return empty;

        // return vec![1.0; self.size];
    }

    pub fn create_pool_idx_storage_buffer(&self) -> Vec<u32> {
        let mut empty = vec![0; self.storage_buffer_size];

        return empty;
    }
}

// PARAMETERS INFO
/*
Storage Buffer (Parameters)
*/
#[derive(Clone)]
pub struct ConvParamInfo {
    pub dim: Vec<TensorInfo>,
    pub kernal_count: Vec<usize>,

    pub k_strides: Vec<usize>, // kernal strides
    pub b_strides: Vec<usize>, // bias strides

    pub b_offset: usize,
    pub split_k: usize,
    pub largest_kernal_layer: usize,
    pub acc_buffer_batch_length: usize,

    pub size: usize,
}

impl ConvParamInfo {
    pub fn new(layer_info: &Vec<ConvLayerInfo>, split_k: usize) -> Self {
        let mut layer_dim = Vec::new();
        let mut kernal_count = Vec::new();

        let largest_kernal_layer = get_kernal_max(layer_info);

        let mut t_size = 0;
        let mut b_offset = 0;

        let mut k_stride_offset = 0;
        let mut b_stride_offset = 0;

        let mut k_strides = Vec::new();
        let mut b_strides = Vec::new();

        for layer in layer_info {
            let tens_info = layer.kernal_info.tens.clone();

            k_strides.push(k_stride_offset);
            b_strides.push(b_stride_offset);
            kernal_count.push(layer.n_kernals);

            k_stride_offset += tens_info.tens_length * layer.n_kernals;
            b_stride_offset += layer.n_kernals;
            layer_dim.push(tens_info);
        }

        let acc_buffer_batch_length =
            largest_kernal_layer * ceil_div(largest_kernal_layer, split_k);

        t_size += k_stride_offset;
        b_offset = t_size;

        t_size += b_stride_offset;

        return ConvParamInfo {
            dim: layer_dim,
            kernal_count: kernal_count,
            k_strides: k_strides,
            b_strides: b_strides,

            b_offset: b_offset,
            split_k: split_k,
            largest_kernal_layer: largest_kernal_layer,
            acc_buffer_batch_length: acc_buffer_batch_length,

            size: t_size,
        };
    }

    // weights for the model
    pub fn create_buffer(&self) -> Vec<f32> {
        let mut empty_vec = vec![1.0; self.size];

        // for i in 0..self.k_strides[1] {
        //     empty_vec[i] = (i as f32 - 32.0) / 64.0;
        // }
        // for i in self.k_strides[1]..self.k_strides[2] {
        //     empty_vec[i] = ((i - self.k_strides[1]) as f32 - 32.0) / 64.0;
        // }

        // println!("{:?}", &empty_vec[self.k_strides[1]..self.k_strides[2]]);

        return empty_vec;

        // let mut rng = rand::thread_rng();

        // let random_floats: Vec<f32> = (0..self.size)
        //     .map(|_| rng.gen_range(-1.0..=1.0))
        //     .collect();

        // return random_floats;
    }

    pub fn create_accumulate_buffer_empty(&self, n_batches: usize) -> Vec<f32> {
        return vec![0.0; self.acc_buffer_batch_length * n_batches];
    }

    pub fn create_buffer_empty(&self, n_batches: usize) -> Vec<f32> {
        return vec![0.0; self.size * n_batches];
    }
}

fn get_vec_product(vec: &Vec<usize>) -> usize {
    let mut val = 1;

    for v in vec {
        val *= v;
    }

    return val;
}

pub fn get_layer_max(layer_info: &Vec<ConvLayerInfo>) -> usize {
    let mut greatest_size = 0;
    for layer in layer_info {
        let layer_size = get_vec_product(&layer.layer_dim) * layer.n_kernals;

        if layer_size > greatest_size {
            greatest_size = layer_size;
        }
    }
    return greatest_size;
}

pub fn get_kernal_max(layer_info: &Vec<ConvLayerInfo>) -> usize {
    let mut greatest_size = 0;
    for layer in layer_info {
        let layer_size = get_vec_product(&layer.kernal_info.dim) * layer.n_kernals;

        if layer_size > greatest_size {
            greatest_size = layer_size;
        }
    }
    return greatest_size;
}
