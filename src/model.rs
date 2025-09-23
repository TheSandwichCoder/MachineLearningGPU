use crate::dispatch::{nn_dispatch::*, gpu_instance::*};
use crate::datatypes::nn_datatypes::*;
use std::time::{Duration, Instant};

#[derive(Clone)]
pub struct ModelConstructor{
    pub nn_dim: Vec<usize>,
    pub n_batches: u32,
    pub n_data_per_batch: u32, // n_batches * n_data_per_batch = total data loaded per batch 
    pub n_epochs: u32,
    pub data_path: String,
    pub lr: f32,
    pub mr: f32,
}

impl ModelConstructor{
    pub fn default() -> Self{
        return ModelConstructor{
            nn_dim: Vec::new(),
            n_batches: 16,
            n_data_per_batch: 100,
            n_epochs: 1,
            data_path: String::from(""),
            lr: 0.1,
            mr: 0.9
        }
    }

    pub fn set_nn_dim(&mut self, nn_dim: &Vec<usize>){
        self.nn_dim = nn_dim.clone();
    }

    pub fn set_lr(&mut self, lr: f32){
        self.lr = lr;
    }
    
    pub fn set_mr(&mut self, mr: f32){
        self.mr = mr;
    }

    pub fn set_batch(&mut self, batch: u32){
        self.n_batches = batch;
    }

    pub fn set_epochs(&mut self, epochs: u32){
        self.n_epochs = epochs;
    }

    pub fn set_datapath(&mut self, path: String){
        self.data_path = path;
    }

    pub fn load_all_data(&mut self, n_data: u32){
        self.n_data_per_batch = n_data / self.n_batches;
    }
}
//./datasets/testing.csv

pub struct BasicNNModel{
    pub nn_info: NeuralNetworkInfo,
    pub gpu_instance: GPUInstance, 
    pub nn_dispatch: NNDispatch,
    pub model_info: ModelConstructor,
}

impl BasicNNModel{
    pub fn construct(constructor: &ModelConstructor) -> Self{
        let gpu_instance = pollster::block_on(GPUInstance::new());
        let nn_dispatch = NNDispatch::new(&gpu_instance, &constructor.nn_dim, constructor.n_batches, constructor.data_path.clone(), constructor.n_data_per_batch, constructor.lr, constructor.mr);

        return BasicNNModel{
            nn_info: NeuralNetworkInfo::new(&constructor.nn_dim, constructor.n_batches as usize, constructor.lr, constructor.mr),
            gpu_instance: gpu_instance,
            nn_dispatch: nn_dispatch,
            model_info: constructor.clone(),
        }
    }

    pub fn show_all_specs(&self){
        self.nn_info.show_all_specs();
    }

    pub fn show_params(&mut self){
        self.nn_dispatch.read_back_params(&self.gpu_instance);
    }

    pub fn save(&self){
        self.nn_dispatch.read_back_save(&self.gpu_instance);
    }

    pub fn debug(&mut self){
        // self.dispatch.data_reader.load_batch_testing();
        // self.dispatch.data_reader.load_batch_mnist();
        self.nn_dispatch.data_reader.load_batch_debug();

        self.nn_dispatch.set_data(&self.gpu_instance);
        
        self.nn_dispatch.forward_mat(&self.gpu_instance);

        self.nn_dispatch.apply_error(&self.gpu_instance);

        self.nn_dispatch.backward_mat(&self.gpu_instance);

        self.nn_dispatch.read_back_act_single(&self.gpu_instance);

        println!("");
        self.nn_dispatch.read_back_gradients(&self.gpu_instance);
        println!("");
        self.nn_dispatch.read_back_params(&self.gpu_instance);

    }

    pub fn train(&mut self){

        let mut sub_batch_i = 0;

        for epoch_i in 0..self.model_info.n_epochs{
            self.nn_dispatch.data_reader.reset_counters();
            println!("Epoch {}:", epoch_i);
            let t0 = Instant::now();

            for load_batch_i in 0..self.nn_dispatch.data_reader.n_load_batches{
                
                // need to load new batch
                // self.dispatch.data_reader.load_batch_mnist();
                
                for sub_batch_i in 0..self.nn_dispatch.data_reader.n_sub_batches{
                    self.nn_dispatch.set_data(&self.gpu_instance);
                                        
                    self.nn_dispatch.forward_mat(&self.gpu_instance);
                    
                    self.nn_dispatch.apply_error(&self.gpu_instance);
                    
                    self.nn_dispatch.backward_mat(&self.gpu_instance);
                    
                    self.nn_dispatch.update_momentum(&self.gpu_instance);
                    
                    self.nn_dispatch.data_reader.increment_sub_batch();
                }
                
                self.nn_dispatch.data_reader.increment_load_batch();
            }
            
            self.gpu_instance.device.poll(wgpu::PollType::Wait).unwrap();
            println!("{:?}", t0.elapsed());
        }
    }

    // get this to work with testing data 
    pub fn test(&mut self){
        println!("Testing");
        self.nn_dispatch.data_reader.reset_counters();
        self.nn_dispatch.clear_metrics(&self.gpu_instance);
        let t0 = Instant::now();

        for load_batch_i in 0..self.nn_dispatch.data_reader.n_load_batches{
            // need to load new batch
            // self.dispatch.data_reader.load_batch_testing();
            self.nn_dispatch.data_reader.load_batch_mnist();

            for sub_batch_i in 0..self.nn_dispatch.data_reader.n_sub_batches{
                self.nn_dispatch.set_data(&self.gpu_instance);
    
                self.nn_dispatch.forward_mat(&self.gpu_instance);
    
                self.nn_dispatch.update_metrics(&self.gpu_instance);
    
                self.nn_dispatch.data_reader.increment_sub_batch();
            }

            self.nn_dispatch.data_reader.increment_load_batch();

        }
        self.gpu_instance.device.poll(wgpu::PollType::Wait).unwrap();
        println!("{:?}", t0.elapsed());

        self.nn_dispatch.read_back_metrics(&self.gpu_instance);

    }
}
