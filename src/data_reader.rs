use std::fs::File;

struct DataValue{
    label: f32,
    info: Vec<f32>,
    data_size: usize,
}

impl DataValue{
    pub fn from_mnist(srecord: &csv::StringRecord) -> DataValue{
        let label = srecord[0].parse::<f32>().unwrap();

        let mut info_vec = Vec::new();

        let mut value_i = 1;
        while value_i < 785{
            let value = srecord[value_i].parse::<f32>().unwrap();
            info_vec.push(value);
            value_i += 1;
        }

        return DataValue{
            label: label,
            info: info_vec.clone(),
            data_size: info_vec.len(),
        }
    }
}

pub struct DataReader{
    pub data_path: String,

    pub loaded_data: Vec<DataValue>,
    pub dataset_length: usize,

    pub load_batch_i: usize, // current load batch
    pub n_load_batches: usize, // number of load batches

    pub load_batch_length: usize, // number of data values in a load batch

    pub data_value_size: usize,
}

impl DataReader{
    pub fn new(data_path: String, load_batch_length: usize) -> Self{
        return DataReader{
            data_path,

            loaded_data: Vec::new(),
            dataset_length: 0,

            load_batch_i: 0,
            n_load_batches: 0,

            load_batch_length: load_batch_length,
            data_value_size: 0,
        }
    }

    // called dump buffer because we wipe values
    pub fn get_buffer(&self) -> Vec<f32>{
        let mut buffer_vec : Vec<f32> = Vec::new();

        for data in &self.loaded_data{
            buffer_vec.push(data.label);
            buffer_vec.extend(data.info.iter());
        }

        return buffer_vec;
    }

    pub fn initialise_mnist_params(&mut self){
        self.dataset_length = 42002;

        self.data_value_size = 784;

        self.n_load_batches = self.dataset_length / self.load_batch_length;
    }

    pub fn increment_batch_i(&mut self){
        if self.load_batch_i < self.n_load_batches - 1{
            self.load_batch_i += 1;
        }
        else{
            self.load_batch_i = 0;
        }
    }

    pub fn load_batch_mnist(&mut self){

        let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path(self.data_path.clone()).unwrap();

        self.loaded_data.clear();

        for result in rdr.records().skip(self.load_batch_length * self.load_batch_i).take(self.load_batch_length) {
            // Notice that we need to provide a type hint for automatic
            // deserialization.
            let record = result.unwrap();

            self.loaded_data.push(DataValue::from_mnist(&record));
        }
    }
}

struct DataDir{

}