use crate::data_reader::*;
use crate::datatypes::{conv_datatypes::*, nn_datatypes::*};
use bytemuck::{Pod, Zeroable};
use csv::Error;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct FlatApplyDir {
    batch_start_i: u32,
    batch_i: u32,
    gradient_start_i: u32,
    batch_contribution: f32,
    n_params: u32,
    lr: f32,
    mr: f32,
}

impl FlatApplyDir {
    pub fn new_nn(nn_info: &NeuralNetworkInfo, dir_i: usize) -> Self {
        return FlatApplyDir {
            batch_start_i: nn_info.activity_info.a_length as u32 * dir_i as u32,
            batch_i: dir_i as u32,
            gradient_start_i: 0,
            batch_contribution: 1.0 / nn_info.n_batches as f32,
            n_params: nn_info.p_length as u32,
            lr: nn_info.lr,
            mr: nn_info.mr,
        };
    }
    pub fn new_conv(conv_info: &ConvolutionInfo, dir_i: usize) -> Self {
        return FlatApplyDir {
            batch_start_i: 0 as u32,
            batch_i: 0 as u32,
            gradient_start_i: 0,
            batch_contribution: 1.0 / conv_info.n_batches as f32,
            n_params: conv_info.param_info.size as u32,
            lr: conv_info.lr,
            mr: conv_info.mr,
        };
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ErrorDir {
    act_size: u32,
    outputs_offset: u32,
    n_outputs: u32,
    ping_start: u32,
    data_size: u32,
}

impl ErrorDir {
    pub fn new(nn_info: &NeuralNetworkInfo) -> Self {
        let last_layer_i = nn_info.n_layers - 1;

        let ping_switch = (last_layer_i - 1) % 2;

        return ErrorDir {
            act_size: nn_info.activity_info.a_length as u32,
            outputs_offset: nn_info.activity_info.s_start as u32
                + nn_info.activity_info.a_strides[last_layer_i] as u32,
            n_outputs: nn_info.activity_info.a_dim[last_layer_i] as u32,
            ping_start: (nn_info.activity_info.d_start
                + ping_switch * nn_info.activity_info.a_deriv_buffer_size)
                as u32,
            data_size: nn_info.layer_dim[0] as u32, // temporary
        };
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ErrorPC {
    layer_idx: u32,
    n_batches: u32,
    _pad2: u32,
    _pad3: u32,
}

impl ErrorPC {
    pub fn new(dr: &DataReader, n_batches: usize) -> Self {
        return ErrorPC {
            layer_idx: (dr.load_batch_length * (dr.data_value_size + 1) * dr.load_batch_i
                + dr.sub_batch_length * (dr.data_value_size + 1) * dr.sub_batch_i)
                as u32,
            n_batches: n_batches as u32,
            _pad2: 0,
            _pad3: 0,
        };
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct TestMetrics {
    correct: u32,
    incorrect: u32,
    _pad1: u32,
    _pad2: u32,
}

impl TestMetrics {
    pub fn zero() -> Self {
        return TestMetrics {
            correct: 0,
            incorrect: 0,
            _pad1: 0,
            _pad2: 0,
        };
    }
}

fn make_transpose(n: bool, m: bool, o: bool) -> u32 {
    let mut v: u32 = 0;

    if n {
        v |= 1 << 0;
    }
    if m {
        v |= 1 << 1;
    }
    if o {
        v |= 1 << 2;
    }

    return v;
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct MatrixDir {
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
    k: u32,
}

impl MatrixDir {
    /*
    new forward:
    n: params/weights
    m: activity
    w: activity
    */
    pub fn null() -> Self {
        MatrixDir {
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

    pub fn new_forward(nn_info: &NeuralNetworkInfo, dir_i: usize) -> Self {
        let ping_switch = dir_i % 2;
        let pong_switch = (dir_i + 1) % 2;

        return MatrixDir {
            n_read_start: nn_info.layer_info[dir_i].offset as u32,
            m_read_start: nn_info.activity_info.a_swap_buffer_size as u32 * ping_switch as u32,

            n_stride_length: (nn_info.layer_dim[dir_i] + 1) as u32,
            m_stride_length: nn_info.activity_info.a_length as u32,

            w_start: nn_info.activity_info.a_swap_buffer_size as u32 * pong_switch as u32,
            w_stride_length: nn_info.activity_info.a_length as u32,

            add_const: 1,
            c_start: (nn_info.layer_info[dir_i].offset + nn_info.layer_dim[dir_i]) as u32,
            c_stride_length: (nn_info.layer_dim[dir_i] + 1) as u32,
            a_func_type: 1,

            extra: 1,
            e_start: nn_info.activity_info.s_start as u32
                + nn_info.activity_info.a_strides[dir_i + 1] as u32,
            e_stride_length: nn_info.activity_info.a_length as u32,

            transpose: make_transpose(false, false, false),

            n: nn_info.layer_dim[dir_i + 1] as u32,
            m: nn_info.n_batches as u32,
            k: nn_info.layer_dim[dir_i] as u32,
        };
    }

    pub fn new_backward_deriv(nn_info: &NeuralNetworkInfo, dir_i: usize) -> Self {
        let ping_switch = dir_i % 2;
        let pong_switch = (dir_i + 1) % 2;

        let ping_pong_default = nn_info.activity_info.d_start;

        if dir_i >= nn_info.n_layers - 1 {
            return MatrixDir::null();
        }

        return MatrixDir {
            n_read_start: nn_info.layer_info[dir_i].offset as u32,
            m_read_start: (ping_pong_default
                + nn_info.activity_info.a_deriv_buffer_size * ping_switch)
                as u32,

            n_stride_length: (nn_info.layer_dim[dir_i] + 1) as u32,
            m_stride_length: nn_info.activity_info.a_length as u32,

            w_start: (ping_pong_default + nn_info.activity_info.a_deriv_buffer_size * pong_switch)
                as u32,
            w_stride_length: nn_info.activity_info.a_length as u32,

            add_const: 0,
            c_start: 0,
            c_stride_length: 0,
            a_func_type: 2,

            extra: 2,
            e_start: nn_info.activity_info.s_start as u32
                + nn_info.activity_info.a_strides[dir_i] as u32,
            e_stride_length: nn_info.activity_info.a_length as u32,

            transpose: make_transpose(true, false, false),

            n: nn_info.layer_dim[dir_i] as u32,
            m: nn_info.n_batches as u32,
            k: nn_info.layer_dim[dir_i + 1] as u32,
        };
    }

    pub fn new_backward_gradients(nn_info: &NeuralNetworkInfo, dir_i: usize) -> Self {
        // have to add 1 to compensate
        let ping_switch = (dir_i + 1) % 2;
        let pong_switch = (dir_i + 1 + 1) % 2;

        let ping_pong_default = nn_info.activity_info.d_start;

        return MatrixDir {
            n_read_start: (ping_pong_default
                + nn_info.activity_info.a_deriv_buffer_size * pong_switch)
                as u32,
            m_read_start: nn_info.activity_info.s_start as u32
                + nn_info.activity_info.a_strides[dir_i] as u32,

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
        };
    }
}
