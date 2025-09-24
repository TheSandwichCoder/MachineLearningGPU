use crate::conv_datatypes::ConvolutionInfo;
use bytemuck::{Pod, Zeroable};

/*
params:
1. read start 1
2. read start 2
3. kernal info
4. layer info

fields:
layer dim
kernal dim

*/
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Im2ColDir{
    kernal: [u32; 4],
    kernal_offset: [i32; 4],
    layer: [u32; 4],

    kernal_read_start: u32,
    layer_read_start: u32,
    write_start: u32,

    c_start: u32,

    transpose: u32,

    n: u32,
    m: u32,
    k: u32
}

impl Im2ColDir{
    pub fn new(conv_info: &ConvolutionInfo, dir_i: usize) -> Self{
        let conv_layer = &conv_info.conv_layers[dir_i]; 

        return Im2ColDir{
            kernal: [conv_layer.kernal_info.dim[0] as u32, conv_layer.kernal_info.dim[1] as u32, conv_layer.kernal_info.dim[2] as u32, 0],
            kernal_offset: [conv_layer.kernal_info.k_range.start_idx, conv_layer.kernal_info.k_range.start_idx, conv_layer.kernal_info.c_range.start_idx, 0],
            layer: [conv_layer.layer_dim[0] as u32, conv_layer.layer_dim[1] as u32, conv_layer.layer_dim[2] as u32, 0],
            
            kernal_read_start: conv_info.param_info.k_strides[dir_i] as u32,
            layer_read_start: 0,
            write_start: conv_info.activity_info.swap_buffer_size as u32,

            c_start: (conv_info.param_info.b_offset + conv_info.param_info.b_strides[dir_i]) as u32,
            transpose: 0,

            n: conv_layer.n_kernals as u32,
            m: conv_info.activity_info.dim[dir_i].tens_length as u32,
            k: conv_layer.kernal_info.size as u32,
            
        }
    }
}

