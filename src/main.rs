use std::num::NonZeroU64;
use wgpu::util::DeviceExt;
use bytemuck::{Pod, Zeroable};
use std::time::Instant;

mod dispatch;
mod data_reader;
mod model;
mod gpu_dirs;
mod functions;
mod datatypes;

use crate::datatypes::*;
use crate::dispatch::*;
use crate::model::*;

use crate::conv_datatypes::*;

fn main() {
    // pollster::block_on(run());
    // let mut constructor = ModelConstructor::default();

    let mut conv_construct = ConvolutionConstructor::default();

    conv_construct.set_inputs(vec![28, 28, 1], 3);
    conv_construct.set_pool(vec![2, 2]);
    conv_construct.set_kernal(vec![8, 4]);
    conv_construct.set_outputs(vec![10, 1]);
    conv_construct.set_n_batches(1);

    let conv_info = ConvolutionInfo::construct(conv_construct);
    
    conv_info.show_all_specs();

    // constructor.set_nn_dim(&vec![2, 17, 2]);
    // constructor.set_lr(1.0);
    // constructor.load_all_data(10000);

    // constructor.set_nn_dim(&vec![784, 128, 64, 10]);
    // constructor.set_datapath(String::from("./datasets/mnist_numbers.csv"));
    // constructor.set_epochs(10);
    // constructor.set_batch(64);
    // constructor.load_all_data(40000);
    // constructor.set_lr(0.0004);
    // constructor.set_mr(0.9);

    // let mut nn_model = BasicNNModel::construct(&constructor); 

    // nn_model.show_all_specs();

    
    // // nn_model.debug();
    // // nn_model.test();
    
    // // nn_model.show_params();
    // nn_model.train();
    // nn_model.test();

    // nn_model.save();


}