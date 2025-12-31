use bytemuck::{Pod, Zeroable};
use std::num::NonZeroU64;
use std::time::Instant;
use wgpu::util::DeviceExt;

mod constants;
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
// check the parser (think it is wrong)

// Things that could go wrong
// might accumulate multiple times in the buffers?
// forward and backward process?
// might have something to do with gradients? checked it and looked repeated

// Things that I am pretty sure work
// conv forward
// conv backward (with deriv)
// nn forward and backward
// storage buffer (single pass)
//

fn main() {
    // temp thing

    // let mut model_construct = ModelConstructor::default();

    // model_construct.set_conv_n_layers(3);
    // model_construct.set_conv_input_layer_dim(vec![14, 14, 1]);

    // model_construct.add_kernal_layer(2, 2, 8);
    // model_construct.add_kernal_layer(3, 1, 16);
    // model_construct.add_kernal_layer(2, 2, 8);

    // model_construct.set_nn_dim(vec![0, 256, 128, 26]);

    let mut model_construct = ModelConstructor::default();

    // model_construct.set_conv_n_layers(3);
    // model_construct.set_conv_input_layer_dim(vec![28, 28, 1]);

    // model_construct.add_kernal_layer(2, 2, 32);
    // model_construct.add_kernal_layer(3, 1, 64);
    // model_construct.add_kernal_layer(2, 2, 32);

    // model_construct.set_nn_dim(vec![0, 512, 256, 26]);
    model_construct.set_nn_dim(vec![784, 512, 256, 26]);

    model_construct.set_lr(0.0001);
    model_construct.set_mr(0.9);
    model_construct.set_vr(0.999);
    model_construct.set_wd(0.00001);

    model_construct.set_batch(16);
    // model_construct.set_split_k_auto();
    model_construct.set_epochs(10);

    // model_construct.set_data_mnist_downsampled();

    // model_construct.set_data_mnist();
    model_construct.set_data_mnist_letters();

    // // NN STUFF
    let mut nn_model: NNModel = NNModel::construct(model_construct);

    nn_model.show_all_specs();
    nn_model.train();
    nn_model.test();

    // // CONV STUFF
    // let mut convnn_model = ConvNNModel::construct(model_construct);

    // convnn_model.show_all_specs();

    // convnn_model.train();
    // convnn_model.test();
    // convnn_model.save();
}
