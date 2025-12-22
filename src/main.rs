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
    // pollster::block_on(run()););
    let mut model_construct = ModelConstructor::default();

    model_construct.set_conv_n_layers(4);
    model_construct.set_conv_input_layer_dim(vec![28, 28, 1]);

    model_construct.add_kernal_layer(4, 2, 64);
    model_construct.add_kernal_layer(3, 1, 128);
    model_construct.add_kernal_layer(2, 2, 64);

    model_construct.set_nn_dim(vec![0, 512, 256, 26]);

    model_construct.set_lr(0.0001);
    model_construct.set_mr(0.9);
    model_construct.set_vr(0.999);
    model_construct.set_split_k(1024);

    model_construct.set_batch(16);
    model_construct.set_epochs(3);

    // model_construct.set_data_mnist();
    model_construct.set_data_mnist_letters();

    let mut convnn_model = ConvNNModel::construct(model_construct);

    convnn_model.show_all_specs();

    // convnn_model.debug();

    convnn_model.load();

    convnn_model.train();
    convnn_model.test();
    convnn_model.save();
}

/*
possible diagnoses

output logits are too large

rule out
 - not becuase of batching

brainstorm
 - running something multiple times
 - big input so big output

*/
