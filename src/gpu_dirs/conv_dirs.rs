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
pub struct Im2ColDir_F {
    kernal: [u32; 4],
    kernal_offset: [i32; 4],
    layer: [u32; 4],

    kernal_read_start: u32,
    layer_read_start: u32,
    write_start: u32,

    c_start: u32,

    n_outputs: u32,
    batch_swap_buffer_size: u32,

    n: u32,
    m: u32,
    k: u32,

    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

impl Im2ColDir_F {
    pub fn new(conv_info: &ConvolutionInfo, dir_i: usize) -> Self {
        let conv_layer = &conv_info.conv_layers[dir_i];

        return Im2ColDir_F {
            kernal: [
                conv_layer.kernal_info.dim[0] as u32,
                conv_layer.kernal_info.dim[1] as u32,
                conv_layer.kernal_info.dim[2] as u32,
                0,
            ],
            kernal_offset: [
                conv_layer.kernal_info.k_range.start_idx,
                conv_layer.kernal_info.k_range.start_idx,
                0,
                0,
            ],
            layer: [
                conv_layer.layer_dim[0] as u32,
                conv_layer.layer_dim[1] as u32,
                conv_layer.layer_dim[2] as u32,
                0,
            ],

            kernal_read_start: conv_info.param_info.k_strides[dir_i] as u32,
            layer_read_start: 0,
            write_start: conv_info.activity_info.swap_buffer_size as u32,

            c_start: (conv_info.param_info.b_offset + conv_info.param_info.b_strides[dir_i]) as u32,
            n_outputs: conv_layer.layer_size_2d as u32,

            batch_swap_buffer_size: conv_info.activity_info.batch_swap_buffer_size as u32,

            n: conv_layer.n_kernals as u32,
            m: (conv_layer.layer_size_2d * conv_info.n_batches) as u32,
            k: conv_layer.kernal_info.size as u32,

            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
        };
    }
}

// Backwards Gradients
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Im2ColDir_BG {
    kernal: [u32; 4],
    o_layer_offset: [i32; 4],
    o_layer_dim: [u32; 4],
    i_layer_dim: [u32; 4],

    deriv_read_start: u32,
    input_read_start: u32,
    write_start: u32,

    c_start: u32,

    n_outputs: u32,
    batch_swap_buffer_size: u32,
    acc_buffer_batch_length: u32,

    split_k: u32,
    n_k_splits: u32,
    batch_k_length: u32,

    n: u32,
    m: u32,
    k: u32,

    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

impl Im2ColDir_BG {
    pub fn new(conv_info: &ConvolutionInfo, dir_i: usize) -> Self {
        let curr_conv_layer = &conv_info.conv_layers[dir_i];
        let next_conv_layer = &conv_info.conv_layers[dir_i + 1];

        return Im2ColDir_BG {
            kernal: [
                curr_conv_layer.kernal_info.dim[0] as u32,
                curr_conv_layer.kernal_info.dim[1] as u32,
                curr_conv_layer.kernal_info.dim[2] as u32,
                0,
            ],
            o_layer_offset: [
                curr_conv_layer.kernal_info.k_range.start_idx,
                curr_conv_layer.kernal_info.k_range.start_idx,
                0,
                0,
            ],
            o_layer_dim: [
                curr_conv_layer.layer_dim[0] as u32,
                curr_conv_layer.layer_dim[1] as u32,
                curr_conv_layer.layer_dim[2] as u32,
                0,
            ],
            i_layer_dim: [
                curr_conv_layer.layer_dim[0] as u32,
                curr_conv_layer.layer_dim[1] as u32,
                curr_conv_layer.layer_dim[2] as u32,
                0,
            ],

            deriv_read_start: 0 as u32,
            input_read_start: conv_info.activity_info.strides[dir_i] as u32,
            write_start: 0,

            c_start: 0,

            n_outputs: curr_conv_layer.kernal_info.size as u32,
            batch_swap_buffer_size: conv_info.activity_info.batch_swap_buffer_size as u32,
            acc_buffer_batch_length: conv_info.param_info.acc_buffer_batch_length as u32,

            split_k: conv_info.split_k as u32,
            n_k_splits: curr_conv_layer.acc_length as u32,
            batch_k_length: curr_conv_layer.layer_size_2d as u32,

            n: curr_conv_layer.n_kernals as u32,
            m: curr_conv_layer.kernal_info.size as u32,
            k: (curr_conv_layer.layer_size_2d * conv_info.n_batches) as u32,

            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
        };
    }
}

// Backwards derivatives
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Im2ColDir_BD {
    kernal: [u32; 4],
    kernal_offset: [i32; 4],
    i_layer_dim: [u32; 4],
    o_layer_dim: [u32; 4],

    kernal_read_start: u32,
    layer_read_start: u32,
    write_start: u32,

    c_start: u32,

    n_outputs: u32,
    batch_swap_buffer_size: u32,
    kernal_layer_size: u32,
    kernal_size: u32,

    n: u32,
    m: u32,
    k: u32,

    _pad1: u32,
}

impl Im2ColDir_BD {
    pub fn new(conv_info: &ConvolutionInfo, dir_i: usize) -> Self {
        let curr_conv_layer = &conv_info.conv_layers[dir_i];
        let next_conv_layer = &conv_info.conv_layers[dir_i + 1];

        let curr_kernal = &curr_conv_layer.kernal_info;

        return Im2ColDir_BD {
            kernal: [
                curr_conv_layer.kernal_info.dim[0] as u32,
                curr_conv_layer.kernal_info.dim[1] as u32,
                curr_conv_layer.kernal_info.dim[2] as u32,
                0,
            ],
            kernal_offset: [
                -curr_conv_layer.kernal_info.k_range.end_idx + 1,
                -curr_conv_layer.kernal_info.k_range.end_idx + 1,
                0,
                0,
            ],
            i_layer_dim: [
                curr_conv_layer.layer_dim[0] as u32,
                curr_conv_layer.layer_dim[1] as u32,
                curr_conv_layer.layer_dim[2] as u32,
                0,
            ],
            o_layer_dim: [
                curr_conv_layer.layer_dim[0] as u32,
                curr_conv_layer.layer_dim[1] as u32,
                curr_conv_layer.n_kernals as u32,
                0,
            ],

            kernal_read_start: conv_info.param_info.k_strides[dir_i] as u32,
            layer_read_start: 0,
            write_start: conv_info.activity_info.swap_buffer_size as u32,

            c_start: 0,

            n_outputs: (curr_conv_layer.layer_size_2d) as u32,
            batch_swap_buffer_size: conv_info.activity_info.batch_swap_buffer_size as u32,

            kernal_layer_size: curr_kernal.size_2d as u32,
            kernal_size: curr_kernal.size as u32,

            n: (curr_conv_layer.layer_dim[2]) as u32,
            m: (curr_conv_layer.layer_size_2d * conv_info.n_batches) as u32,
            k: (curr_kernal.size_2d * curr_conv_layer.n_kernals) as u32,

            _pad1: 0,
        };
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct PoolDir {
    i_layer_dim: [u32; 4],
    o_layer_dim: [u32; 4],

    read_start: u32,
    write_start: u32,
    pool_k: u32,
    batch_swap_buffer_size: u32,

    storage_write_start: u32,
    storage_write_skip: u32,
    _pad1: u32,
    _pad2: u32,
}

impl PoolDir {
    pub fn forward_new(conv_info: &ConvolutionInfo, dir_i: usize) -> Self {
        let conv_layer = &conv_info.conv_layers[dir_i];
        let next_conv_layer = &conv_info.conv_layers[dir_i + 1];

        return PoolDir {
            i_layer_dim: [
                conv_layer.layer_dim[0] as u32,
                conv_layer.layer_dim[1] as u32,
                next_conv_layer.layer_dim[2] as u32,
                0,
            ],
            o_layer_dim: [
                next_conv_layer.layer_dim[0] as u32,
                next_conv_layer.layer_dim[1] as u32,
                next_conv_layer.layer_dim[2] as u32,
                0,
            ],

            read_start: conv_info.activity_info.swap_buffer_size as u32,
            write_start: 0,

            pool_k: conv_layer.pooling_info.k as u32,
            batch_swap_buffer_size: conv_info.activity_info.batch_swap_buffer_size as u32,

            storage_write_start: conv_info.activity_info.strides[dir_i + 1] as u32,
            storage_write_skip: conv_info.activity_info.dim[dir_i + 1].tens_length as u32,

            _pad1: 0,
            _pad2: 0,
        };
    }

    pub fn backward_new(conv_info: &ConvolutionInfo, dir_i: usize) -> Self {
        let conv_layer = &conv_info.conv_layers[dir_i];
        let next_conv_layer = &conv_info.conv_layers[dir_i + 1];

        return PoolDir {
            i_layer_dim: [
                conv_layer.layer_dim[0] as u32,
                conv_layer.layer_dim[1] as u32,
                next_conv_layer.layer_dim[2] as u32,
                0,
            ],
            o_layer_dim: [
                next_conv_layer.layer_dim[0] as u32,
                next_conv_layer.layer_dim[1] as u32,
                next_conv_layer.layer_dim[2] as u32,
                0,
            ],

            read_start: conv_info.activity_info.swap_buffer_size as u32,
            write_start: 0,

            pool_k: conv_layer.pooling_info.k as u32,
            batch_swap_buffer_size: conv_info.activity_info.batch_swap_buffer_size as u32,

            storage_write_start: 0,
            storage_write_skip: conv_info.activity_info.dim[dir_i + 1].tens_length as u32,

            _pad1: 0,
            _pad2: 0,
        };
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct AccDir {
    read_start: u32,
    write_start: u32,
    acc_length: u32,
    n_weights: u32,
}

impl AccDir {
    pub fn new(conv_info: &ConvolutionInfo, dir_i: usize) -> Self {
        let conv_layer = &conv_info.conv_layers[dir_i];

        return AccDir {
            read_start: 0,
            write_start: conv_info.activity_info.strides[dir_i] as u32,
            acc_length: conv_layer.acc_length as u32,
            n_weights: (conv_layer.n_kernals * conv_layer.kernal_info.size) as u32,
        };
    }
}
