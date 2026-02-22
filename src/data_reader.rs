use std::fs::File;
use std::string;

use crate::functions::*;
use crate::model::ModelConstructor;
use csv::StringRecord;

// data_batches_per_load - n sub batches
pub struct DataConstructor {
    pub dataset_info: DatasetInfo,
    pub data_batches_per_load: usize,
    pub train_split: f32,
    pub test_split: f32,

    pub n_batches: usize,
}

impl DataConstructor {
    pub fn new(
        dataset_info: DatasetInfo,
        train_split: f32,
        test_split: f32,
        data_batches_per_load: usize,
        n_batches: usize,
    ) -> Self {
        return DataConstructor {
            dataset_info: dataset_info,
            train_split: train_split,
            test_split: test_split,
            data_batches_per_load: data_batches_per_load,
            n_batches: n_batches,
        };
    }

    pub fn from_model_constructor(model_constructor: &ModelConstructor) -> Self {
        return DataConstructor {
            dataset_info: get_dataset_info(model_constructor.dataset.clone()),
            train_split: model_constructor.train_test_split,
            test_split: 1.0 - model_constructor.train_test_split,
            data_batches_per_load: model_constructor.data_batches_per_load,
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
    pub fn null() -> DataValue {
        return DataValue {
            label: -1.0,
            info: Vec::new(),
            data_size: 0,
        };
    }

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

    pub fn from_mnist_downsampled(srecord: &csv::StringRecord) -> DataValue {
        let label = srecord[0].parse::<f32>().unwrap();

        let mut info_vec = Vec::new();

        let mut value_i = 1;
        while value_i < 197 {
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
            info: vec![1.0; 784],
            data_size: 784,
        };
    }
}

// load batch - number of times the model has to reload new data
// sub batch - number of forward steps the model can use the existing data
// n batches - number of data that is loaded in 1 forward step (n batches of the model)

pub enum DataReaderType {
    Train,
    Test,
}

#[derive(Clone)]
pub struct DatasetInfo {
    pub dataset_type: Dataset,
    pub length: usize,
    pub data_value_size: usize,
    pub path: String,
}

#[derive(Clone)]
pub enum Dataset {
    Empty,
    MNIST,
    EMNIST,
    CIFAR,
}

pub fn get_dataset_info(dataset: Dataset) -> DatasetInfo {
    match dataset {
        Dataset::Empty => {
            return DatasetInfo {
                dataset_type: dataset,
                length: 0,
                data_value_size: 0,
                path: String::new(),
            };
        }
        Dataset::MNIST => {
            return DatasetInfo {
                dataset_type: dataset,
                length: 42000,
                data_value_size: 784,
                path: String::from("datasets/mnist_numbers.csv"),
            };
        }
        Dataset::EMNIST => {
            return DatasetInfo {
                dataset_type: dataset,
                length: 124801,
                data_value_size: 784,
                path: String::from("datasets/mnist_letters.csv"),
            };
        }
        Dataset::CIFAR => {
            return DatasetInfo {
                dataset_type: dataset,
                length: 60000,
                data_value_size: 3072,
                path: String::from("datasets/cifar"),
            };
        }
    }
}

pub struct DataReader {
    pub dataset_info: DatasetInfo,
    pub loaded_data: Vec<DataValue>,

    pub set_type: DataReaderType,
    pub train_data_reader: SubDataReader,
    pub test_data_reader: SubDataReader,

    pub n_batches: usize,
}

impl DataReader {
    pub fn construct(data_constructor: &DataConstructor) -> Self {
        let train_length =
            (data_constructor.dataset_info.length as f32 * data_constructor.train_split) as usize;
        let test_length =
            (data_constructor.dataset_info.length as f32 * data_constructor.test_split) as usize;

        let train_start = 0 as usize;
        let test_start = train_length;

        return DataReader::new(
            data_constructor.dataset_info.clone(),
            SubDataReader::construct(data_constructor, train_start, train_length),
            SubDataReader::construct(data_constructor, test_start, test_length),
            data_constructor.n_batches,
        );
    }

    pub fn new(
        dataset_info: DatasetInfo,
        train_data_reader: SubDataReader,
        test_data_reader: SubDataReader,
        n_batches: usize,
    ) -> Self {
        return DataReader {
            dataset_info,

            loaded_data: Vec::new(),

            set_type: DataReaderType::Train,
            train_data_reader: train_data_reader,
            test_data_reader: test_data_reader,

            n_batches: n_batches,
        };
    }

    pub fn set_set_type(&mut self, new_type: DataReaderType) {
        self.set_type = new_type;
    }

    pub fn load_data(&mut self) {
        self.loaded_data.clear();

        match self.dataset_info.dataset_type {
            Dataset::MNIST | Dataset::EMNIST => {
                let mut rdr = csv::ReaderBuilder::new()
                    .has_headers(true)
                    .from_path(self.dataset_info.path.clone())
                    .unwrap();
                let mut counter: usize = 0;
                for result in rdr.records() {
                    let record = result.unwrap();

                    let data_value = self.load_data_single_type(&record);

                    self.loaded_data.push(data_value);
                    counter += 1;
                }
            }
            Dataset::CIFAR => {
                // todo
            }

            _ => {
                // todo
            }
        }
    }

    pub fn load_data_single_type(&mut self, string_record: &StringRecord) -> DataValue {
        // return DataValue::from_mnist_downsampled(string_record);
        // return DataValue::from_mnist(string_record);
        match self.dataset_info.dataset_type {
            Dataset::EMNIST => {
                return DataValue::from_mnist_letters(string_record);
            }
            Dataset::MNIST => {
                return DataValue::from_mnist(string_record);
            }
            _ => {
                return DataValue::null();
            }
        }
    }

    pub fn get_load_batch_buffer(&self) -> Vec<f32> {
        match self.set_type {
            DataReaderType::Train => {
                return self
                    .train_data_reader
                    .get_load_batch_buffer(&self.loaded_data);
            }

            DataReaderType::Test => {
                return self
                    .test_data_reader
                    .get_load_batch_buffer(&self.loaded_data);
            }
        }
    }

    pub fn get_sub_batch_i(&self) -> usize {
        match self.set_type {
            DataReaderType::Train => {
                return self.train_data_reader.sub_batch_i;
            }

            DataReaderType::Test => {
                return self.test_data_reader.sub_batch_i;
            }
        }
    }

    pub fn increment_sub_batch(&mut self) {
        match self.set_type {
            DataReaderType::Train => {
                self.train_data_reader.increment_sub_batch();
            }

            DataReaderType::Test => {
                self.test_data_reader.increment_sub_batch();
            }
        }
    }

    pub fn increment_load_batch(&mut self) {
        match self.set_type {
            DataReaderType::Train => {
                self.train_data_reader.increment_load_batch();
            }

            DataReaderType::Test => {
                self.test_data_reader.increment_load_batch();
            }
        }
    }

    pub fn get_n_sub_batches(&self) -> usize {
        match self.set_type {
            DataReaderType::Train => {
                return self.train_data_reader.n_sub_batches;
            }

            DataReaderType::Test => {
                return self.test_data_reader.n_sub_batches;
            }
        }
    }

    pub fn get_n_load_batches(&self) -> usize {
        match self.set_type {
            DataReaderType::Train => {
                return self.train_data_reader.n_load_batches;
            }

            DataReaderType::Test => {
                return self.test_data_reader.n_load_batches;
            }
        }
    }

    pub fn reset_counters(&mut self) {
        self.train_data_reader.reset_counters();
        self.test_data_reader.reset_counters();
    }

    pub fn show_all_specs(&self) {
        println!("DATA READER INFO");

        println!("\nCAPACITY INFO");
        println!("volume: {}", self.dataset_info.length);

        let dataset_mem = self.dataset_info.length * self.dataset_info.data_value_size * 4;

        println!("size: {} floats ({}MB)", dataset_mem, get_mb(dataset_mem));

        println!("\nTRAIN DATASET:");
        self.train_data_reader.show_all_specs();
        println!("TEST DATASET:");
        self.test_data_reader.show_all_specs();
    }
}

pub struct SubDataReader {
    pub dataset_length: usize,
    pub data_read_start: usize,

    pub load_batch_i: usize, // current load batch
    pub sub_batch_i: usize,

    pub n_load_batches: usize, // number of sub batches in a load batch
    pub n_sub_batches: usize,  // number of data batches in a sub batch
    pub n_batches: usize,      // literally number of batches run in the model

    pub data_value_size: usize,
}

impl SubDataReader {
    pub fn construct(
        data_constructor: &DataConstructor,
        data_read_start: usize,
        dataset_length: usize,
    ) -> Self {
        return SubDataReader::new(
            dataset_length,
            data_read_start,
            data_constructor.data_batches_per_load,
            data_constructor.dataset_info.data_value_size,
            data_constructor.n_batches,
        );
    }

    pub fn new(
        dataset_length: usize,
        data_read_start: usize,
        n_sub_batches: usize,
        data_value_size: usize,
        n_batches: usize,
    ) -> Self {
        return SubDataReader {
            dataset_length: dataset_length,
            data_read_start: data_read_start,

            load_batch_i: 0,
            sub_batch_i: 0,

            n_load_batches: dataset_length / (n_sub_batches * n_batches),
            n_sub_batches: n_sub_batches,
            n_batches: n_batches,

            data_value_size: data_value_size,
        };
    }

    // called dump buffer because we wipe values
    pub fn get_load_batch_buffer(&self, loaded_data: &Vec<DataValue>) -> Vec<f32> {
        let mut buffer_vec: Vec<f32> = Vec::new();

        let start_idx =
            self.data_read_start + self.load_batch_i * self.n_sub_batches * self.n_batches;
        let stop_idx =
            self.data_read_start + (self.load_batch_i + 1) * self.n_sub_batches * self.n_batches;

        for data_i in start_idx..stop_idx {
            let data = &loaded_data[data_i];

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

    pub fn show_all_specs(&self) {
        println!("CAPACITY INFO");
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
