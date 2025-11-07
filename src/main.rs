use bytemuck::{Pod, Zeroable};
use std::num::NonZeroU64;
use std::time::Instant;
use wgpu::util::DeviceExt;

mod data_reader;
mod datatypes;
mod dispatch;
mod functions;
mod gpu_dirs;
mod model;

use crate::datatypes::*;
use crate::dispatch::{conv_dispatch::*, gpu_instance::GPUInstance};
use crate::model::*;

use crate::conv_datatypes::*;

// TODO
// fix the momentum calc for nn and conv
// integrate both of them together

fn main() {
    // pollster::block_on(run()););
    let mut model_construct = ModelConstructor::default();

    model_construct.set_conv_n_layers(3);
    model_construct.set_conv_input_layer_dim(vec![28, 28, 1]);

    model_construct.add_kernal_layer(4, 2, 16);
    model_construct.add_kernal_layer(2, 1, 8);

    model_construct.set_nn_dim(vec![0, 128, 64, 10]);

    model_construct.set_lr(0.00001);
    model_construct.set_mr(0.9);

    model_construct.set_batch(16);
    model_construct.set_epochs(10);

    model_construct.set_data_mnist();
    // model_construct.set_data_mnist_letters();

    let mut convnn_model = ConvNNModel::construct(model_construct);

    convnn_model.show_all_specs();

    convnn_model.debug();

    // convnn_model.train();
    // convnn_model.test();
    // convnn_model.save();
}
