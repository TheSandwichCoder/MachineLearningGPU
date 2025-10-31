use std::fs::File;

use crate::model::ModelConstructor;

pub struct DataConstructor {
    pub data_path: String,
    pub data_per_batch: usize,
}

impl DataConstructor {
    pub fn new(data_path: String, data_per_batch: usize) -> Self {
        return DataConstructor {
            data_path: data_path,
            data_per_batch: data_per_batch,
        };
    }

    pub fn from_model_constructor(model_constructor: &ModelConstructor) -> Self {
        return DataConstructor {
            data_path: model_constructor.data_path.clone(),
            data_per_batch: model_constructor.n_data_per_batch,
        };
    }
}

struct DataValue {
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
            info: vec![1.0, 1.0],
            data_size: 2,
        };
    }
}

pub struct DataReader {
    pub data_path: String,

    pub loaded_data: Vec<DataValue>,
    pub dataset_length: usize,

    pub load_batch_i: usize, // current load batch
    pub sub_batch_i: usize,

    pub n_load_batches: usize, // number of load batches
    pub n_sub_batches: usize,

    pub load_batch_length: usize, // number of data values in a load batch
    pub sub_batch_length: usize,  // number of data values in a sub batch

    pub data_value_size: usize,
}

impl DataReader {
    pub fn new(data_path: String, load_batch_length: usize, sub_batch_length: usize) -> Self {
        return DataReader {
            data_path,

            loaded_data: Vec::new(),
            dataset_length: 0,

            load_batch_i: 0,
            sub_batch_i: 0,

            n_load_batches: 0,
            n_sub_batches: load_batch_length / sub_batch_length,

            load_batch_length: load_batch_length,
            sub_batch_length: sub_batch_length,

            data_value_size: 0,
        };
    }

    // called dump buffer because we wipe values
    pub fn get_buffer(&self) -> Vec<f32> {
        let mut buffer_vec: Vec<f32> = Vec::new();

        for data in &self.loaded_data {
            buffer_vec.push(data.label);
            buffer_vec.extend(data.info.iter());
        }

        return buffer_vec;
    }

    pub fn initialise_params_testing(&mut self) {
        self.dataset_length = 1590;
        self.data_value_size = 2;

        self.n_load_batches = self.dataset_length / self.load_batch_length;
    }

    pub fn initialise_params_mnist(&mut self) {
        self.dataset_length = 42002;

        self.data_value_size = 784;

        self.n_load_batches = self.dataset_length / self.load_batch_length;
    }

    pub fn initialise_params_debug(&mut self) {
        self.dataset_length = 10000;
        self.data_value_size = 2;

        self.n_load_batches = self.dataset_length / self.load_batch_length;
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

    pub fn load_batch_mnist(&mut self) {
        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_path(self.data_path.clone())
            .unwrap();

        self.loaded_data.clear();

        for result in rdr
            .records()
            .skip(self.load_batch_length * self.load_batch_i)
            .take(self.load_batch_length)
        {
            let record = result.unwrap();

            self.loaded_data.push(DataValue::from_mnist(&record));
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
}
