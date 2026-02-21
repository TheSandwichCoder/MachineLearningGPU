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
// training + test splits
// crashing with large buffers
// strides
// better data storage

fn main() {
    // let mut model_construct = ModelConstructor::default();

    // model_construct.set_conv_n_layers(3);
    // model_construct.set_conv_input_layer_dim(vec![28, 28, 1]);

    // model_construct.add_kernal_layer(2, 2, 32);
    // model_construct.add_kernal_layer(3, 1, 64);
    // model_construct.add_kernal_layer(2, 2, 32);

    // model_construct.set_nn_dim(vec![0, 512, 256, 26]);

    // model_construct.set_lr(0.0001);
    // model_construct.set_mr(0.9);
    // model_construct.set_vr(0.999);
    // model_construct.set_wd(0.00001);

    let mut model_construct = ModelConstructor::default();

    model_construct.set_nn_dim(vec![784, 512, 256, 26]);

    model_construct.set_batch(16);
    model_construct.set_epochs(10);
    model_construct.set_train_test_splits(0.80);

    // model_construct.set_data_mnist();
    model_construct.set_data_mnist_letters();

    // CONV STUFF
    // let mut convnn_model = ConvNNModel::construct(model_construct);

    let mut convnn_model = NNModel::construct(model_construct);

    convnn_model.show_all_specs();

    convnn_model.train();
    convnn_model.test();
    convnn_model.save();
}
