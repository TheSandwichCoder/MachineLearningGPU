use crate::constants::*;
use crate::data_reader::DataConstructor;
use crate::datatypes::conv_datatypes::*;
use crate::datatypes::nn_datatypes::*;
use crate::dispatch::data_dispatch;
use crate::dispatch::data_dispatch::DataDispatch;
use crate::dispatch::{conv_dispatch::*, gpu_instance::*, nn_dispatch::*};
use crate::functions::*;
use std::time::{Duration, Instant};

use std::fs;

#[derive(Clone)]
pub struct ModelConstructor {
    pub nn_dim: Vec<usize>,

    pub conv_n_layers: usize,
    pub conv_input_layer_dim: Vec<usize>,
    pub conv_pooling_dim: Vec<usize>,
    pub conv_kernal_dim: Vec<usize>,
    pub conv_layer_output: Vec<usize>,
    pub split_k: usize,

    pub n_batches: usize,

    pub data_batches_per_load: usize,
    pub n_epochs: usize,
    pub data_path: String,
    pub dataset_length: usize,
    pub data_value_size: usize,

    pub lr: f32,
    pub mr: f32,
    pub vr: f32,
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
            split_k: 256,

            n_batches: 16,
            data_batches_per_load: 100,
            n_epochs: 1,
            data_path: String::from(""),
            dataset_length: 0,
            data_value_size: 0,

            lr: 0.1,
            mr: 0.9,
            vr: 0.999,
        };
    }

    pub fn load_specs() -> Self {
        let file_string = fs::read_to_string("saves/saved_convnn.txt").unwrap();

        let file_lines: Vec<&str> = file_string.trim().split("\n").collect();

        let nn_dim_str: Vec<&str> = file_lines[0].trim().split(" ").collect();

        let mut nn_dim = Vec::new();
        let mut conv_line_skip = 1;
        for layer_i in 0..nn_dim_str.len() {
            let layer_usize: usize = nn_dim_str[layer_i].parse().unwrap();

            if layer_i != 0 {
                conv_line_skip += layer_usize;
            }
            nn_dim.push(layer_usize);
        }

        let conv_n_layers: usize = file_lines[conv_line_skip].parse().unwrap();

        let input_layer_info: Vec<&str> =
            file_lines[conv_line_skip + 1].trim().split(" ").collect();

        let conv_input_layer_dim: Vec<usize> = vec![
            input_layer_info[0].parse().unwrap(),
            input_layer_info[1].parse().unwrap(),
            input_layer_info[2].parse().unwrap(),
        ];

        let mut conv_pooling_dim: Vec<usize> = Vec::new();
        let mut conv_kernal_dim: Vec<usize> = Vec::new();
        let mut conv_layer_output: Vec<usize> = Vec::new();

        for layer_i in 0..conv_n_layers {
            let layer_info: Vec<&str> = file_lines[conv_line_skip + 2 + layer_i]
                .trim()
                .split(" ")
                .collect();

            conv_kernal_dim.push(layer_info[0].parse().unwrap());
            conv_pooling_dim.push(layer_info[4].parse().unwrap());
            conv_layer_output.push(layer_info[3].parse().unwrap());
        }

        return ModelConstructor {
            nn_dim: nn_dim,
            conv_n_layers: conv_n_layers + 1,
            conv_input_layer_dim: conv_input_layer_dim,
            conv_pooling_dim: conv_pooling_dim,
            conv_kernal_dim: conv_kernal_dim,
            conv_layer_output: conv_layer_output,
            split_k: 256,
            n_batches: 16,
            data_batches_per_load: 100,
            n_epochs: 1,
            data_path: String::from(""),
            dataset_length: 0,
            data_value_size: 0,
            lr: 0.1,
            mr: 0.9,
            vr: 0.999,
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

    pub fn set_vr(&mut self, vr: f32) {
        self.vr = vr;
    }

    pub fn set_batch(&mut self, batch: usize) {
        self.n_batches = batch;
    }

    pub fn set_epochs(&mut self, epochs: usize) {
        self.n_epochs = epochs;
    }

    pub fn set_split_k(&mut self, split_k: usize) {
        self.split_k = split_k;
    }

    pub fn set_split_k_auto(&mut self) {
        let mut greatest_size = 0;
        for layer_i in 0..self.conv_n_layers - 1 {
            let prev_layer_size: usize;
            if layer_i == 0 {
                prev_layer_size = self.conv_input_layer_dim[2];
            } else {
                prev_layer_size = self.conv_layer_output[layer_i - 1];
            }

            let layer_size = self.conv_kernal_dim[layer_i]
                * self.conv_kernal_dim[layer_i]
                * self.conv_layer_output[layer_i]
                * prev_layer_size;

            if layer_size > greatest_size {
                greatest_size = layer_size;
            }
        }
        self.split_k = ceil_div_inv_y(
            LARGEST_BUFFER_SIZE as usize / (self.n_batches * greatest_size),
            greatest_size,
        );
    }

    pub fn set_data_mnist(&mut self) {
        self.set_data_info(String::from("datasets/mnist_numbers.csv"), 42001, 1, 784);
        self.load_all_data();
    }

    pub fn set_data_mnist_letters(&mut self) {
        self.set_data_info(
            String::from("datasets/mnist_letters.csv"),
            124801,
            1500,
            784,
        );
        // self.set_data_info(String::from("datasets/mnist_letters.csv"), 256, 1, 784);
        // self.load_all_data();
    }

    pub fn set_data_info(
        &mut self,
        path: String,
        dataset_length: usize,
        data_batches_per_load: usize,
        data_value_size: usize,
    ) {
        self.data_path = path;
        self.dataset_length = dataset_length;
        self.data_batches_per_load = data_batches_per_load;
        self.data_value_size = data_value_size;
    }

    pub fn load_all_data(&mut self) {
        self.data_batches_per_load = self.dataset_length / self.n_batches;
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
    pub fn construct(constructor: ModelConstructor) -> Self {
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

        let mut t_i = 1;

        for epoch_i in 0..self.model_info.n_epochs {
            self.data_dispatch.data_reader.reset_counters();
            println!("Epoch {}:", epoch_i);
            let t0 = Instant::now();

            for load_batch_i in 0..self.data_dispatch.data_reader.n_load_batches {
                // need to load new batch
                self.data_dispatch.load_data(&self.gpu_instance);

                for sub_batch_i in 0..self.data_dispatch.data_reader.n_sub_batches {
                    self.data_dispatch
                        .set_data_nn(&self.gpu_instance, &self.nn_dispatch);

                    self.nn_dispatch.forward_mat(&self.gpu_instance);

                    self.data_dispatch.apply_error(&self.gpu_instance);

                    self.nn_dispatch.backward_mat(&self.gpu_instance);

                    self.nn_dispatch.update_momentum(&self.gpu_instance, t_i);

                    self.data_dispatch.data_reader.increment_sub_batch();
                    t_i += 1;
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
            self.data_dispatch.load_data(&self.gpu_instance);

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
        self.data_dispatch.data_reader.show_all_specs();
    }

    pub fn show_epoch_specs(&self, time_taken: Duration) {
        let (_forward_nn_flop, training_nn_flop) = self.nn_info.get_n_flops();
        let (_forward_conv_flop, training_conv_flop) = self.conv_info.get_n_flops();

        let total_flops = (get_gflops(training_nn_flop) + get_gflops(training_conv_flop))
            * (self.data_dispatch.data_reader.n_sub_batches
                * self.data_dispatch.data_reader.n_load_batches) as f32;

        let training_size = self.data_dispatch.data_reader.n_load_batches
            * self.data_dispatch.data_reader.n_sub_batches
            * self.data_dispatch.data_reader.n_batches;

        let time_taken_seconds = time_taken.as_secs_f64();

        println!(
            "\n{:?} {} GFLOPS {} samples/s",
            time_taken,
            (total_flops as f64 / time_taken_seconds).round(),
            (training_size as f64 / time_taken_seconds).round(),
        );
    }

    pub fn debug(&mut self) {
        self.data_dispatch.data_reader.reset_counters();
        self.data_dispatch.load_data(&self.gpu_instance);

        self.data_dispatch
            .set_data_convnn(&self.gpu_instance, &self.conv_dispatch);

        self.conv_dispatch.forward_conv_mat(&self.gpu_instance);

        self.nn_dispatch
            .transfer_conv_forward(&self.gpu_instance, &self.conv_dispatch);

        self.nn_dispatch.forward_mat(&self.gpu_instance);

        self.data_dispatch.apply_error(&self.gpu_instance);

        self.nn_dispatch.backward_mat_convnn(&self.gpu_instance);

        // self.conv_dispatch
        //     .transfer_nn_deriv(&self.gpu_instance, &self.nn_dispatch);

        // self.conv_dispatch.backward_conv_full(&self.gpu_instance);

        // self.nn_dispatch.read_back_gradients(&self.gpu_instance);
        // self.nn_dispatch.read_back_act_single(&self.gpu_instance);
        self.conv_dispatch.read_back_act_single(&self.gpu_instance);
    }

    pub fn load(&mut self) {
        let file_string = fs::read_to_string("saves/saved_convnn.txt").unwrap();

        self.nn_dispatch
            .load_params(&self.gpu_instance, &file_string);
        self.conv_dispatch
            .load_params(&self.gpu_instance, &file_string);
    }

    pub fn train(&mut self) {
        let mut t_i = 1;

        for epoch_i in 0..self.model_info.n_epochs {
            self.data_dispatch.data_reader.reset_counters();
            println!("Epoch {}:", epoch_i);
            let t0 = Instant::now();

            for load_batch_i in 0..self.data_dispatch.data_reader.n_load_batches {
                // need to load new batch
                self.data_dispatch.load_data(&self.gpu_instance);

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

                    self.nn_dispatch.update_momentum(&self.gpu_instance, t_i);
                    self.conv_dispatch.update_momentum(&self.gpu_instance, t_i);

                    self.data_dispatch.data_reader.increment_sub_batch();

                    show_progress(
                        self.data_dispatch.data_reader.n_sub_batches
                            * load_batch_i
                            * self.model_info.n_batches
                            + sub_batch_i * self.model_info.n_batches,
                        load_batch_i,
                        self.data_dispatch.data_reader.n_load_batches
                            * self.data_dispatch.data_reader.n_sub_batches
                            * self.data_dispatch.data_reader.n_batches,
                    );

                    t_i += 1;
                }

                self.data_dispatch.data_reader.increment_load_batch();
            }

            self.gpu_instance.device.poll(wgpu::PollType::Wait).unwrap();
            self.show_epoch_specs(t0.elapsed());
            // println!("{:?}", t0.elapsed());
        }
        self.data_dispatch.data_reader.reset_counters();
    }

    pub fn test(&mut self) {
        println!("Testing");
        self.data_dispatch.data_reader.reset_counters();
        self.data_dispatch.clear_metrics(&self.gpu_instance);
        let t0 = Instant::now();

        for load_batch_i in 0..self.data_dispatch.data_reader.n_load_batches {
            // need to load new batch
            self.data_dispatch.load_data(&self.gpu_instance);

            for sub_batch_i in 0..self.data_dispatch.data_reader.n_sub_batches {
                self.data_dispatch
                    .set_data_convnn(&self.gpu_instance, &self.conv_dispatch);

                self.conv_dispatch.forward_conv_mat(&self.gpu_instance);

                self.nn_dispatch
                    .transfer_conv_forward(&self.gpu_instance, &self.conv_dispatch);

                self.nn_dispatch.forward_mat(&self.gpu_instance);

                self.data_dispatch.update_metrics(&self.gpu_instance);

                self.data_dispatch.data_reader.increment_sub_batch();

                show_progress(
                    self.data_dispatch.data_reader.n_sub_batches
                        * load_batch_i
                        * self.model_info.n_batches
                        + sub_batch_i * self.model_info.n_batches,
                    load_batch_i,
                    self.data_dispatch.data_reader.n_load_batches
                        * self.data_dispatch.data_reader.n_sub_batches
                        * self.data_dispatch.data_reader.n_batches,
                );
            }

            self.data_dispatch.data_reader.increment_load_batch();
        }
        self.gpu_instance.device.poll(wgpu::PollType::Wait).unwrap();
        println!("\n{:?}", t0.elapsed());

        self.data_dispatch.read_back_metrics(&self.gpu_instance);
    }

    pub fn save(&self) {
        let nn_params = self.nn_dispatch.get_param_slice(&self.gpu_instance);
        let conv_params = self.conv_dispatch.get_param_slice(&self.gpu_instance);

        let nn_save_string = self.nn_dispatch.nn_info.get_save_string(&nn_params);
        let conv_save_string = self.conv_dispatch.conv_info.get_save_string(&conv_params);

        // println!("{} {}", nn_save_string, conv_save_string);

        let save_string = nn_save_string + &conv_save_string;

        fs::write("./saves/saved_convnn.txt", save_string);
    }
}
