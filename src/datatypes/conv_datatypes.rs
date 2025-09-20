use crate::tensor::*;
use crate::range::*;
use crate::functions::*;

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

#[derive(Clone)]
pub struct ConvolutionConstructor{
    pub i_x: usize,
    pub i_y: usize,
    pub i_c: usize,
    pub n_layers: usize,
    pub n_batches: usize,

    pub pooling_dim: Vec<usize>,
    pub kernal_dim: Vec<usize>,
    pub layer_output: Vec<usize>,
}

impl ConvolutionConstructor{
    pub fn default() -> Self{
        return ConvolutionConstructor{
            i_x: 0,
            i_y: 0,
            i_c: 0,
            n_layers: 0,
            n_batches: 0,

            pooling_dim: Vec::new(),
            kernal_dim: Vec::new(),
            layer_output: Vec::new(),
        }
    }

    pub fn set_inputs(&mut self, input_dim: Vec<usize>, n_layers: usize){
        self.i_x = input_dim[0];
        self.i_y = input_dim[1];
        self.i_c = input_dim[2];

        self.n_layers = n_layers;
    }

    pub fn set_pool(&mut self, pool_dim: Vec<usize>){
        self.pooling_dim = pool_dim;
    }

    pub fn set_kernal(&mut self, kernal_dim: Vec<usize>){
        self.kernal_dim = kernal_dim;
    }

    pub fn set_outputs(&mut self, layer_output: Vec<usize>){
        self.layer_output = layer_output;
    }

    pub fn set_n_batches(&mut self, n_batches: usize){
        self.n_batches = n_batches;
    }
}

#[derive(Clone)]
pub struct ConvolutionInfo{
    pub conv_layers: Vec<ConvLayerInfo>,
    pub input_layer_dim: Vec<usize>,
    pub output_layer_dim: Vec<usize>,
    pub n_layers: usize,
    pub n_batches: usize,
    pub size: usize,
}

impl ConvolutionInfo{
    pub fn construct(constructor: ConvolutionConstructor) -> Self{
        let mut conv_layers = Vec::new();

        let mut prev_x = constructor.i_x;
        let mut prev_y = constructor.i_y;
        let mut prev_c = constructor.i_c;

        let mut output_layer_x = 0;
        let mut output_layer_y = 0;
        let mut output_layer_c = 0;

        let mut size = 0;

        for layer_i in 0..constructor.n_layers{
            let pool_k: usize;
            let kernal_k: usize;
            let output_n: usize;

            let mut temp_x = 0; 
            let mut temp_y = 0; 
            let mut temp_c = 0; 
            
            if layer_i == constructor.n_layers - 1{
                kernal_k = 0;
                pool_k = 0;
                output_n = 0;
                
                output_layer_x = prev_x;
                output_layer_y = prev_y;
                output_layer_c = prev_c;
            }
            else{
                kernal_k = constructor.kernal_dim[layer_i];
                pool_k = constructor.pooling_dim[layer_i];
                output_n = constructor.layer_output[layer_i];
                temp_x = ceil_div(prev_x, pool_k);
                temp_y = ceil_div(prev_y, pool_k);
                temp_c = output_n;
            }

            
            let layer_info = ConvLayerInfo::new(vec![prev_x, prev_y, prev_c], vec![kernal_k, kernal_k, prev_c], output_n, pool_k);
            size += layer_info.size;
            
            prev_x = temp_x;
            prev_y = temp_y;
            prev_c = temp_c;

            conv_layers.push(layer_info);
        }

        return ConvolutionInfo{
            conv_layers: conv_layers,
            input_layer_dim: vec![constructor.i_x, constructor.i_y, constructor.i_c],
            output_layer_dim: vec![output_layer_x, output_layer_y, output_layer_c],
            n_layers: constructor.n_layers,
            n_batches: constructor.n_batches,
            size: size,
        }
    }

    pub fn show_all_specs(&self){
        println!("\nCONVOLUTION LAYERS");

        println!("Input layer Dim");
        println!("{:?}", self.input_layer_dim);

        println!("Output layer Dim");
        println!("{:?} \n", self.output_layer_dim);
        
        println!("Convolutional layer Dim");
        for conv_layer_i in 0..self.n_layers{
            println!("- {:?}", self.conv_layers[conv_layer_i].layer_dim);
        }

        println!("Kernals");

        for kernal_layer_i in 0..self.n_layers - 1{
            
            println!("- {} {:?} {:?}", self.conv_layers[kernal_layer_i].n_kernals ,self.conv_layers[kernal_layer_i].kernal_info.dim);
        }


    }
}

#[derive (Clone)]
pub struct ConvLayerInfo{
    pub layer_dim: Vec<usize>,

    pub n_kernals: usize,
    pub kernal_info: KernalInfo,
    pub pooling: PoolingInfo,

    pub size: usize,
}

impl ConvLayerInfo{
    pub fn new(layer_dim: Vec<usize>, kernal_dim: Vec<usize>, n_kernals: usize, pool_size: usize) -> Self{

        let mut size = 0;

        let mut kernal_info = KernalInfo::new(kernal_dim);

        size += kernal_info.size * n_kernals;

        let pooling_info = PoolingInfo::new(&layer_dim, pool_size);

        return ConvLayerInfo{
            layer_dim: layer_dim,
            
            n_kernals: n_kernals,
            kernal_info: kernal_info,
            pooling: pooling_info,

            size: size,
        }
    }
}

#[derive (Clone)]
pub struct PoolingInfo{
    pub dim: Vec<usize>,
    pub k: usize,
}

impl PoolingInfo{
    pub fn new(pool_dim: &Vec<usize>, pool_size: usize) -> Self{
        return PoolingInfo{
            dim: pool_dim.clone(),
            k: pool_size,
        }
    }
}


#[derive (Clone)]
pub struct KernalInfo{
    pub dim: Vec<usize>,
    pub tens: TensorInfo,
    pub range: KernalRange,
    pub size: usize,
}

impl KernalInfo{
    pub fn new(kernal_dim: Vec<usize>) -> Self{
        let mut reversed = kernal_dim.clone();
        reversed.reverse();

        let k = kernal_dim[0];

        let range = KernalRange::new(-(floor_div(k, 2) as i32), ceil_div(k, 2) as i32);

        let tens = TensorInfo::new(&reversed);
        let size = tens.offset;

        return KernalInfo{
            dim: kernal_dim,
            tens: tens,
            range: range,
            size: size,
        }
    }
}

