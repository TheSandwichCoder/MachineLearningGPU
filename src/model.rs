use crate::dispatch::NNDispatch;
use crate::datatypes::NeuralNetworkInfo;

#[derive(Clone)]
pub struct ModelConstructor{
    pub nn_dim: Vec<usize>,
    pub n_batches: u32,
    pub n_data_per_batch: u32, // n_batches * n_data_per_batch = total data loaded per batch 
    pub n_epochs: u32,
    pub data_path: String,
    pub lr: f32,
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
        }
    }

    pub fn set_nn_dim(&mut self, nn_dim: &Vec<usize>){
        self.nn_dim = nn_dim.clone();
    }

    pub fn set_lr(&mut self, lr: f32){
        self.lr = lr;
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
            nn_info: NeuralNetworkInfo::new(&constructor.nn_dim, constructor.n_batches as usize, constructor.lr),
            dispatch: pollster::block_on(NNDispatch::new(&constructor.nn_dim, constructor.n_batches, constructor.data_path.clone(), constructor.n_data_per_batch, constructor.lr)),
            model_info: constructor.clone(),
        }
    }

    pub fn show_all_specs(&self){
        self.nn_info.show_all_specs();
    }

    pub fn show_params(&mut self){
        self.dispatch.read_back_params();
    }

    pub fn debug(&mut self){
        // self.dispatch.data_reader.load_batch_testing();
        self.dispatch.data_reader.load_batch_mnist();

        self.dispatch.set_data();
        
        self.dispatch.forward();

        self.dispatch.apply_error();

        self.dispatch.backward();

        self.dispatch.apply_gradients();

        self.dispatch.read_back_act_single();

        self.dispatch.read_back_params();

    }

    pub fn train(&mut self){

        let mut sub_batch_i = 0;

        for epoch_i in 0..self.model_info.n_epochs{
            self.dispatch.data_reader.reset_counters();
            println!("Epoch {}:", epoch_i);

            for load_batch_i in 0..self.dispatch.data_reader.n_load_batches{

                // need to load new batch
                // self.dispatch.data_reader.load_batch_testing();
                // self.dispatch.data_reader.load_batch_mnist();
                
                for sub_batch_i in 0..self.dispatch.data_reader.n_sub_batches{
                    // println!("load {}/{}", sub_batch_i, self.dispatch.data_reader.n_sub_batches);
                    self.dispatch.set_data();
        
                    self.dispatch.forward();
        
                    self.dispatch.apply_error();
        
                    self.dispatch.backward();
        
                    self.dispatch.apply_gradients();
        
                    self.dispatch.data_reader.increment_sub_batch();

                }
                
                self.dispatch.data_reader.increment_load_batch();
            }

            self.dispatch.device.poll(wgpu::PollType::Wait).unwrap();
        }
    }

    // get this to work with testing data 
    pub fn test(&mut self){
        self.dispatch.data_reader.reset_counters();
        self.dispatch.clear_metrics();
        println!("asdfasdf {} {}", self.dispatch.data_reader.n_load_batches, self.dispatch.data_reader.n_sub_batches);

        for load_batch_i in 0..self.dispatch.data_reader.n_load_batches{
            // need to load new batch
            // self.dispatch.data_reader.load_batch_testing();
            self.dispatch.data_reader.load_batch_mnist();

            for sub_batch_i in 0..self.dispatch.data_reader.n_sub_batches{
                self.dispatch.set_data();
    
                self.dispatch.forward();
    
                self.dispatch.update_metrics();
    
                self.dispatch.data_reader.increment_sub_batch();
            }

            self.dispatch.data_reader.increment_load_batch();

        }

        self.dispatch.read_back_metrics();

    }
}
