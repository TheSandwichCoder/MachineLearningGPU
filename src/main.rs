use std::num::NonZeroU64;
use wgpu::util::DeviceExt;
use bytemuck::{Pod, Zeroable};
use std::time::Instant;

mod datatypes;
mod dispatch;
mod data_reader;
mod model;

use crate::datatypes::*;
use crate::dispatch::*;
use crate::model::*;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Meta {
    len: u32,
    _pad: [u32; 6],
    scale: f32, // note: uniform layout rules are strict; this keeps it simple
    // Uniform buffers are read in 16B chunks; extra padding is fine.
}




fn main() {
    // pollster::block_on(run());
    let mut constructor = ModelConstructor::default();

    // constructor.set_nn_dim(&vec![2, 3, 2]);
    // constructor.set_lr(1.0);
    // constructor.load_all_data(10000);

    constructor.set_nn_dim(&vec![784, 128, 64, 10]);
    constructor.set_datapath(String::from("./datasets/mnist_numbers.csv"));
    constructor.set_epochs(10);
    constructor.set_batch(64);
    constructor.load_all_data(40000);
    constructor.set_lr(0.0004);
    constructor.set_mr(0.9);

    let mut nn_model = BasicNNModel::construct(&constructor); 

    nn_model.show_all_specs();

    
    // nn_model.debug();
    // nn_model.test();
    
    // nn_model.show_params();
    nn_model.train();
    nn_model.test();

    nn_model.save();


}