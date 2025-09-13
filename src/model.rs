use crate::dispatch::NNDispatch;
use crate::datatypes::NeuralNetworkInfo;
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
            n_data_per_batch: 2624,
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
    pub dispatch: NNDispatch,
    pub model_info: ModelConstructor,
}

impl BasicNNModel{
    pub fn construct(constructor: &ModelConstructor) -> Self{
        return BasicNNModel{
            nn_info: NeuralNetworkInfo::new(&constructor.nn_dim, constructor.n_batches as usize, constructor.lr, constructor.mr),
            dispatch: pollster::block_on(NNDispatch::new(&constructor.nn_dim, constructor.n_batches, constructor.data_path.clone(), constructor.n_data_per_batch, constructor.lr, constructor.mr)),
            model_info: constructor.clone(),
        }
    }

    pub fn show_all_specs(&self){
        self.nn_info.show_all_specs();
    }

    pub fn show_params(&mut self){
        self.dispatch.read_back_params();
    }

    pub fn save(&self){
        self.dispatch.read_back_save();
    }

    pub fn debug(&mut self){
        // self.dispatch.data_reader.load_batch_testing();
        // self.dispatch.data_reader.load_batch_mnist();
        self.dispatch.data_reader.load_batch_debug();

        self.dispatch.set_data();
        
        self.dispatch.forward_mat();
        // self.dispatch.forward();

        // self.dispatch.apply_error();

        // self.dispatch.backward();

        self.dispatch.read_back_act_single();

        self.dispatch.read_back_params();

    }

    pub fn train(&mut self){

        let mut sub_batch_i = 0;

        for epoch_i in 0..self.model_info.n_epochs{
            self.dispatch.data_reader.reset_counters();
            println!("Epoch {}:", epoch_i);
            let t0 = Instant::now();

            for load_batch_i in 0..self.dispatch.data_reader.n_load_batches{
                
                // need to load new batch
                // self.dispatch.data_reader.load_batch_mnist();
                
                for sub_batch_i in 0..self.dispatch.data_reader.n_sub_batches{
                    self.dispatch.set_data();
                    
                    
                    // self.dispatch.forward();
                    // let t0 = Instant::now();
                    self.dispatch.forward_mat();
                    // self.dispatch.device.poll(wgpu::PollType::Wait).unwrap();
                    // println!("{:?}", t0.elapsed());
                    
                    self.dispatch.apply_error();
                    
                    self.dispatch.backward();
                    
                    
                    self.dispatch.update_gradients();
                    self.dispatch.update_momentum();
                    
                    self.dispatch.data_reader.increment_sub_batch();
                }
                
                self.dispatch.data_reader.increment_load_batch();
            }
            
            self.dispatch.device.poll(wgpu::PollType::Wait).unwrap();
            println!("{:?}", t0.elapsed());
        }
    }

    // get this to work with testing data 
    pub fn test(&mut self){
        println!("Testing");
        self.dispatch.data_reader.reset_counters();
        self.dispatch.clear_metrics();

        for load_batch_i in 0..self.dispatch.data_reader.n_load_batches{
            // need to load new batch
            // self.dispatch.data_reader.load_batch_testing();
            self.dispatch.data_reader.load_batch_mnist();

            for sub_batch_i in 0..self.dispatch.data_reader.n_sub_batches{
                self.dispatch.set_data();
    
                self.dispatch.forward_mat();
                // self.dispatch.forward();
    
                self.dispatch.update_metrics();
    
                self.dispatch.data_reader.increment_sub_batch();
            }

            self.dispatch.data_reader.increment_load_batch();

        }

        self.dispatch.read_back_metrics();

    }
}
