use std::fs::File;

use crate::functions::*;
use crate::model::ModelConstructor;
use csv::StringRecord;

// data_batches_per_load - n sub batches
pub struct DataConstructor {
    pub data_path: String,
    pub dataset_length: usize,
    pub data_batches_per_load: usize,
    pub data_value_size: usize,
    pub n_batches: usize,
}

impl DataConstructor {
    pub fn new(
        data_path: String,
        dataset_length: usize,
        data_batches_per_load: usize,
        data_value_size: usize,
        n_batches: usize,
    ) -> Self {
        return DataConstructor {
            data_path: data_path,
            dataset_length: dataset_length,
            data_batches_per_load: data_batches_per_load,
            data_value_size: data_value_size,
            n_batches: n_batches,
        };
    }

    pub fn from_model_constructor(model_constructor: &ModelConstructor) -> Self {
        return DataConstructor {
            data_path: model_constructor.data_path.clone(),
            dataset_length: model_constructor.dataset_length,
            data_batches_per_load: model_constructor.data_batches_per_load,
            data_value_size: model_constructor.data_value_size,
            n_batches: model_constructor.n_batches,
        };
    }
}

pub struct DataValue {
    label: f32,
    info: Vec<f32>,
    data_size: usize,
}

impl DataValue {
    pub fn from_mnist(srecord: &csv::StringRecord) -> DataValue {
        let label = srecord[0].parse::<f32>().unwrap();

        let mut info_vec = Vec::new();

        let mut value_i = 1;
        while value_i < 785 {
            let value = srecord[value_i].parse::<f32>().unwrap();
            info_vec.push(value / 255.0);
            value_i += 1;
        }

        return DataValue {
            label: label,
            info: info_vec.clone(),
            data_size: info_vec.len(),
        };
    }

    pub fn from_mnist_letters(srecord: &csv::StringRecord) -> DataValue {
        let label = srecord[784].parse::<f32>().unwrap() - 1.0;

        let mut info_vec = Vec::new();

        let mut value_i = 0;
        while value_i < 784 {
            let value = srecord[value_i].parse::<f32>().unwrap();
            info_vec.push(value);
            value_i += 1;
        }

        return DataValue {
            label: label,
            info: info_vec,
            data_size: 784,
        };
    }

    pub fn from_testing(srecord: &csv::StringRecord) -> DataValue {
        let label = srecord[0].parse::<f32>().unwrap();

        let mut info_vec = Vec::new();

        let mut value_i = 1;
        while value_i < 3 {
            let value = srecord[value_i].parse::<f32>().unwrap();
            info_vec.push(value);
            value_i += 1;
        }

        return DataValue {
            label: label,
            info: info_vec.clone(),
            data_size: info_vec.len(),
        };
    }

    pub fn from_debug() -> DataValue {
        return DataValue {
            label: 0.0,
            info: vec![1.0; 784],
            data_size: 784,
        };
    }
}

// load batch - number of times the model has to reload new data
// sub batch - number of forward steps the model can use the existing data
// n batches - number of data that is loaded in 1 forward step (n batches of the model)

pub struct DataReader {
    pub data_path: String,

    pub loaded_data: Vec<DataValue>,
    pub dataset_length: usize,

    pub load_batch_i: usize, // current load batch
    pub sub_batch_i: usize,

    pub n_load_batches: usize, // number of sub batches in a load batch
    pub n_sub_batches: usize,  // number of data batches in a sub batch
    pub n_batches: usize,      // literally number of batches run in the model

    pub data_value_size: usize,
}

impl DataReader {
    pub fn construct(data_constructor: &DataConstructor) -> Self {
        return DataReader::new(
            data_constructor.data_path.clone(),
            data_constructor.dataset_length,
            data_constructor.data_batches_per_load,
            data_constructor.data_value_size,
            data_constructor.n_batches,
        );
    }

    pub fn new(
        data_path: String,
        dataset_length: usize,
        n_sub_batches: usize,
        data_value_size: usize,
        n_batches: usize,
    ) -> Self {
        return DataReader {
            data_path,

            loaded_data: Vec::new(),
            dataset_length: dataset_length,

            load_batch_i: 0,
            sub_batch_i: 0,

            n_load_batches: dataset_length / (n_sub_batches * n_batches),
            n_sub_batches: n_sub_batches,
            n_batches: n_batches,

            data_value_size: data_value_size,
        };
    }

    // called dump buffer because we wipe values
    pub fn get_load_batch_buffer(&self) -> Vec<f32> {
        let mut buffer_vec: Vec<f32> = Vec::new();

        let start_idx = self.load_batch_i * self.n_sub_batches * self.n_batches;
        let stop_idx = (self.load_batch_i + 1) * self.n_sub_batches * self.n_batches;

        for data_i in start_idx..stop_idx {
            let data = &self.loaded_data[data_i];

            buffer_vec.push(data.label);
            buffer_vec.extend(data.info.iter());
        }

        return buffer_vec;
    }

    pub fn reset_counters(&mut self) {
        self.load_batch_i = 0;
        self.sub_batch_i = 0;
    }

    pub fn increment_sub_batch(&mut self) {
        self.sub_batch_i += 1;
    }

    pub fn increment_load_batch(&mut self) {
        self.sub_batch_i = 0;

        if self.load_batch_i < self.n_load_batches - 1 {
            self.load_batch_i += 1;
        } else {
            self.load_batch_i = 0;
        }
    }

    pub fn load_data_single_type(&mut self, string_record: &StringRecord) -> DataValue {
        // return DataValue::from_mnist(string_record);
        return DataValue::from_mnist_letters(string_record);
    }

    pub fn load_data(&mut self) {
        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_path(self.data_path.clone())
            .unwrap();

        self.loaded_data.clear();

        let mut counter: usize = 0;
        for result in rdr.records() {
            let record = result.unwrap();

            let data_value = self.load_data_single_type(&record);

            self.loaded_data.push(data_value);
            counter += 1;
        }
    }

    pub fn load_batch_debug(&mut self) {
        for i in 0..10000 {
            self.loaded_data.push(DataValue::from_debug());
        }
    }

    pub fn load_batch_testing(&mut self) {
        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(false)
            .from_path(self.data_path.clone())
            .unwrap();

        self.loaded_data.clear();

        for result in rdr.records() {
            let record = result.unwrap();

            self.loaded_data.push(DataValue::from_testing(&record));
        }
    }

    pub fn show_all_specs(&self) {
        println!("DATA READER INFO");

        println!("\nCAPACITY INFO");
        println!("volume: {}", self.dataset_length);

        let dataset_mem = self.dataset_length * self.data_value_size * 4;

        println!("size: {} floats ({}MB)", dataset_mem, get_mb(dataset_mem));
        println!(
            "reloads per epoch (n load batches): {}",
            self.n_load_batches
        );

        println!("\nUSAGE INFO");
        println!("n sub batches: {}", self.n_sub_batches);
        println!("n batches: {}", self.n_batches);

        println!("Efficiency");
        let n_data_points_used = self.n_load_batches * self.n_sub_batches * self.n_batches;
        println!(
            "{} / {} ({}%)",
            n_data_points_used,
            self.dataset_length,
            n_data_points_used as f32 / self.dataset_length as f32
        );
        println!("\n");
    }
}
