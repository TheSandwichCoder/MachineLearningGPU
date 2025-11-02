use crate::data_reader::DataConstructor;
use crate::datatypes::conv_datatypes::*;
use crate::datatypes::nn_datatypes::*;
use crate::dispatch::data_dispatch;
use crate::dispatch::data_dispatch::DataDispatch;
use crate::dispatch::{conv_dispatch::*, gpu_instance::*, nn_dispatch::*};
use std::time::{Duration, Instant};

#[derive(Clone)]
pub struct ModelConstructor {
    pub nn_dim: Vec<usize>,

    pub conv_n_layers: usize,
    pub conv_input_layer_dim: Vec<usize>,
    pub conv_pooling_dim: Vec<usize>,
    pub conv_kernal_dim: Vec<usize>,
    pub conv_layer_output: Vec<usize>,

    pub n_batches: usize,
    pub n_data_per_batch: usize, // n_batches * n_data_per_batch = total data loaded per batch
    pub n_epochs: usize,
    pub data_path: String,
    pub lr: f32,
    pub mr: f32,
}

impl ModelConstructor {
    pub fn default() -> Self {
        return ModelConstructor {
            nn_dim: Vec::new(),

            conv_n_layers: 0,
            conv_input_layer_dim: Vec::new(),
            conv_pooling_dim: Vec::new(),
            conv_kernal_dim: Vec::new(),
            conv_layer_output: Vec::new(),

            n_batches: 16,
            n_data_per_batch: 100,
            n_epochs: 1,
            data_path: String::from(""),
            lr: 0.1,
            mr: 0.9,
        };
    }

    pub fn set_conv_n_layers(&mut self, n_layers: usize) {
        self.conv_n_layers = n_layers;
    }

    pub fn set_conv_input_layer_dim(&mut self, input_dim: Vec<usize>) {
        self.conv_input_layer_dim = input_dim;
    }

    pub fn add_kernal_layer(&mut self, kernal_size: usize, pool_size: usize, n_kernals: usize) {
        self.conv_kernal_dim.push(kernal_size);
        self.conv_pooling_dim.push(pool_size);
        self.conv_layer_output.push(n_kernals);
    }

    pub fn set_nn_dim(&mut self, nn_dim: Vec<usize>) {
        self.nn_dim = nn_dim.clone();
    }

    pub fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }

    pub fn set_mr(&mut self, mr: f32) {
        self.mr = mr;
    }

    pub fn set_batch(&mut self, batch: usize) {
        self.n_batches = batch;
    }

    pub fn set_epochs(&mut self, epochs: usize) {
        self.n_epochs = epochs;
    }

    pub fn set_datapath(&mut self, path: String) {
        self.data_path = path;
    }

    pub fn load_all_data(&mut self, n_data: u32) {
        self.n_data_per_batch = n_data as usize / self.n_batches;
    }
}
//./datasets/testing.csv

pub struct BasicNNModel {
    pub nn_info: NeuralNetworkInfo,
    pub gpu_instance: GPUInstance,
    pub nn_dispatch: NNDispatch,
    pub data_dispatch: DataDispatch,
    pub model_info: ModelConstructor,
}

impl BasicNNModel {
    pub fn construct(constructor: &ModelConstructor) -> Self {
        let gpu_instance = pollster::block_on(GPUInstance::new());
        let nn_constructor = NNConstructor::from_model_constructor(&constructor);
        let data_constructor = DataConstructor::from_model_constructor(&constructor);

        let nn_dispatch = NNDispatch::new(&gpu_instance, &nn_constructor);
        let data_dispatch = DataDispatch::new_nn(&gpu_instance, &data_constructor, &nn_dispatch);

        let nn_info = NeuralNetworkInfo::construct(&nn_constructor);

        return BasicNNModel {
            nn_info: nn_info,
            gpu_instance: gpu_instance,
            nn_dispatch: nn_dispatch,
            data_dispatch: data_dispatch,

            model_info: constructor.clone(),
        };
    }

    pub fn show_all_specs(&self) {
        self.nn_info.show_all_specs();
    }

    pub fn show_params(&mut self) {
        self.nn_dispatch.read_back_params(&self.gpu_instance);
    }

    pub fn save(&self) {
        self.nn_dispatch.read_back_save(&self.gpu_instance);
    }

    pub fn debug(&mut self) {
        // self.data_dispatch.data_reader.load_batch_testing();
        // self.data_dispatch.data_reader.load_batch_mnist();
        self.data_dispatch.data_reader.load_batch_debug();

        self.data_dispatch
            .set_data_nn(&self.gpu_instance, &self.nn_dispatch);

        self.nn_dispatch.forward_mat(&self.gpu_instance);

        self.data_dispatch.apply_error(&self.gpu_instance);

        self.nn_dispatch.backward_mat(&self.gpu_instance);

        self.nn_dispatch.read_back_act_single(&self.gpu_instance);

        // println!("");
        // self.nn_dispatch.read_back_gradients(&self.gpu_instance);
        // println!("");
        // self.nn_dispatch.read_back_params(&self.gpu_instance);
    }

    pub fn train(&mut self) {
        let mut sub_batch_i = 0;

        for epoch_i in 0..self.model_info.n_epochs {
            self.data_dispatch.data_reader.reset_counters();
            println!("Epoch {}:", epoch_i);
            let t0 = Instant::now();

            for load_batch_i in 0..self.data_dispatch.data_reader.n_load_batches {
                // need to load new batch
                // self.dispatch.data_reader.load_batch_mnist();

                for sub_batch_i in 0..self.data_dispatch.data_reader.n_sub_batches {
                    self.data_dispatch
                        .set_data_nn(&self.gpu_instance, &self.nn_dispatch);

                    self.nn_dispatch.forward_mat(&self.gpu_instance);

                    self.data_dispatch.apply_error(&self.gpu_instance);

                    self.nn_dispatch.backward_mat(&self.gpu_instance);

                    self.nn_dispatch.update_momentum(&self.gpu_instance);

                    self.data_dispatch.data_reader.increment_sub_batch();
                }

                self.data_dispatch.data_reader.increment_load_batch();
            }

            self.gpu_instance.device.poll(wgpu::PollType::Wait).unwrap();
            println!("{:?}", t0.elapsed());
        }
    }

    // get this to work with testing data
    pub fn test(&mut self) {
        println!("Testing");
        self.data_dispatch.data_reader.reset_counters();
        self.data_dispatch.clear_metrics(&self.gpu_instance);
        let t0 = Instant::now();

        for load_batch_i in 0..self.data_dispatch.data_reader.n_load_batches {
            // need to load new batch
            // self.dispatch.data_reader.load_batch_testing();
            self.data_dispatch.data_reader.load_batch_mnist();

            for sub_batch_i in 0..self.data_dispatch.data_reader.n_sub_batches {
                self.data_dispatch
                    .set_data_nn(&self.gpu_instance, &self.nn_dispatch);

                self.nn_dispatch.forward_mat(&self.gpu_instance);

                self.data_dispatch.update_metrics(&self.gpu_instance);

                self.data_dispatch.data_reader.increment_sub_batch();
            }

            self.data_dispatch.data_reader.increment_load_batch();
        }
        self.gpu_instance.device.poll(wgpu::PollType::Wait).unwrap();
        println!("{:?}", t0.elapsed());

        self.data_dispatch.read_back_metrics(&self.gpu_instance);
    }
}

pub struct ConvNNModel {
    pub nn_info: NeuralNetworkInfo,
    pub conv_info: ConvolutionInfo,
    pub gpu_instance: GPUInstance,

    pub nn_dispatch: NNDispatch,
    pub conv_dispatch: ConvDispatch,
    pub data_dispatch: DataDispatch,

    pub model_info: ModelConstructor,
}

impl ConvNNModel {
    pub fn construct(constructor: ModelConstructor) -> Self {
        let gpu_instance = pollster::block_on(GPUInstance::new());

        let conv_constructor = ConvConstructor::from_model_constructor(&constructor);
        let mut nn_constructor = NNConstructor::from_model_constructor(&constructor);
        let data_constructor = DataConstructor::from_model_constructor(&constructor);

        let conv_info = ConvolutionInfo::construct(&conv_constructor);
        let conv_dispatch = ConvDispatch::new(&gpu_instance, &conv_constructor, &data_constructor);

        nn_constructor.update_input_n_dim(conv_info.conv_layers[conv_info.n_layers - 1].layer_size);

        let nn_dispatch = NNDispatch::new(&gpu_instance, &nn_constructor);
        let nn_info = NeuralNetworkInfo::construct(&nn_constructor);

        let data_dispatch = DataDispatch::new_convnn(
            &gpu_instance,
            &data_constructor,
            &conv_dispatch,
            &nn_dispatch,
        );

        return ConvNNModel {
            nn_info: nn_info,
            conv_info: conv_info,
            gpu_instance: gpu_instance,

            nn_dispatch: nn_dispatch,
            conv_dispatch: conv_dispatch,
            data_dispatch: data_dispatch,

            model_info: constructor,
        };
    }

    pub fn show_all_specs(&self) {
        self.nn_info.show_all_specs();
        self.conv_info.show_all_specs();
    }

    pub fn debug(&mut self) {
        self.data_dispatch
            .set_data_convnn(&self.gpu_instance, &self.conv_dispatch);

        self.conv_dispatch.forward_conv_mat(&self.gpu_instance);
        // self.conv_dispatch.read_back_act_single(&self.gpu_instance);

        self.nn_dispatch
            .transfer_conv_forward(&self.gpu_instance, &self.conv_dispatch);

        self.nn_dispatch.forward_mat(&self.gpu_instance);

        self.data_dispatch.apply_error(&self.gpu_instance);

        self.nn_dispatch.backward_mat_convnn(&self.gpu_instance);

        self.conv_dispatch
            .transfer_nn_deriv(&self.gpu_instance, &self.nn_dispatch);

        self.conv_dispatch.backward_conv_full(&self.gpu_instance);

        self.conv_dispatch.update_momentum(&self.gpu_instance);

        // self.conv_dispatch.read_back_act_single(&self.gpu_instance);

        // self.nn_dispatch.read_back_act_single(&self.gpu_instance);
    }

    pub fn train(&mut self) {
        let mut sub_batch_i = 0;

        for epoch_i in 0..self.model_info.n_epochs {
            self.data_dispatch.data_reader.reset_counters();
            println!("Epoch {}:", epoch_i);
            let t0 = Instant::now();

            for load_batch_i in 0..self.data_dispatch.data_reader.n_load_batches {
                // need to load new batch
                // self.dispatch.data_reader.load_batch_mnist();

                for sub_batch_i in 0..self.data_dispatch.data_reader.n_sub_batches {
                    self.data_dispatch
                        .set_data_convnn(&self.gpu_instance, &self.conv_dispatch);

                    self.conv_dispatch.forward_conv_mat(&self.gpu_instance);

                    self.nn_dispatch
                        .transfer_conv_forward(&self.gpu_instance, &self.conv_dispatch);

                    self.nn_dispatch.forward_mat(&self.gpu_instance);

                    self.data_dispatch.apply_error(&self.gpu_instance);

                    self.nn_dispatch.backward_mat_convnn(&self.gpu_instance);

                    self.conv_dispatch
                        .transfer_nn_deriv(&self.gpu_instance, &self.nn_dispatch);

                    self.conv_dispatch.backward_conv_full(&self.gpu_instance);

                    self.nn_dispatch.update_momentum(&self.gpu_instance);
                    self.conv_dispatch.update_momentum(&self.gpu_instance);

                    self.data_dispatch.data_reader.increment_sub_batch();
                }

                self.data_dispatch.data_reader.increment_load_batch();
            }

            self.gpu_instance.device.poll(wgpu::PollType::Wait).unwrap();
            println!("{:?}", t0.elapsed());
        }
    }

    pub fn test(&mut self) {
        println!("Testing");
        self.data_dispatch.data_reader.reset_counters();
        self.data_dispatch.clear_metrics(&self.gpu_instance);
        let t0 = Instant::now();

        for load_batch_i in 0..self.data_dispatch.data_reader.n_load_batches {
            // need to load new batch
            // self.dispatch.data_reader.load_batch_testing();
            self.data_dispatch.data_reader.load_batch_mnist();

            for sub_batch_i in 0..self.data_dispatch.data_reader.n_sub_batches {
                self.data_dispatch
                    .set_data_convnn(&self.gpu_instance, &self.conv_dispatch);

                self.conv_dispatch.forward_conv_mat(&self.gpu_instance);

                self.nn_dispatch
                    .transfer_conv_forward(&self.gpu_instance, &self.conv_dispatch);

                self.nn_dispatch.forward_mat(&self.gpu_instance);

                self.data_dispatch.update_metrics(&self.gpu_instance);

                self.data_dispatch.data_reader.increment_sub_batch();
            }

            self.data_dispatch.data_reader.increment_load_batch();
        }
        self.gpu_instance.device.poll(wgpu::PollType::Wait).unwrap();
        println!("{:?}", t0.elapsed());

        self.data_dispatch.read_back_metrics(&self.gpu_instance);
    }
}
