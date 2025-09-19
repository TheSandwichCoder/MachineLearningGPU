
#[derive(Clone)]
pub struct TensorInfo{
    pub offset: usize,
    pub tens_dim: Vec<usize>,
    pub tens_n_dim: usize,
    pub tens_strides: Vec<usize>,
    pub tens_length: usize,
}

impl TensorInfo{
    pub fn null() -> Self{
        return TensorInfo{
            offset: 0,
            tens_dim: Vec::new(),
            tens_n_dim: 0,
            tens_strides: Vec::new(),
            tens_length: 0,
        };
    }
    
    pub fn new(tens_dim: &Vec<usize>) -> Self{
        return TensorInfo{
            offset: 0,
            tens_dim: tens_dim.clone(),
            tens_n_dim: tens_dim.len(),
            tens_strides: get_tensor_strides(tens_dim),
            tens_length: get_tensor_size(tens_dim),
        }
    }

    // start idx + slice length -> (mem_start, mem_end)
    pub fn get_slice(&self, idx_coord: &Vec<usize>, slice_length: usize) -> (usize, usize){
        let start_i = self.get_index(idx_coord);
        let end_i = start_i + slice_length;

        return (start_i, end_i);
    }

    pub fn get_index(&self, idx_coord: &Vec<usize>) -> usize{
        return vec_dot(&self.tens_strides, idx_coord) + self.offset;
    }
}


fn get_tensor_size(tens_dim: &[usize]) -> usize{
    let mut size: usize = 1;

    for n in tens_dim{
        size *= n;
    } 

    return size;
}

fn get_tensor_strides(tens_dim: &Vec<usize>) -> Vec<usize>{
    let n_dim = tens_dim.len();
    
    let mut strides: Vec<usize> = vec![0; n_dim];

    for i in 0..(n_dim - 1){
        strides[i] = get_tensor_size(&tens_dim[(i + 1)..n_dim]);
    }

    strides[n_dim - 1] = 1;

    return strides;
}


fn vec_dot(vec1: &Vec<usize>, vec2: &Vec<usize>) -> usize{
    let l1 = vec1.len();
    let l2 = vec2.len();

    let mut n : usize = 0;

    if l1 != l2{
        panic!("Vectors are not same length");
    }

    else{
        for i in 0..l1{
            n += vec1[i] * vec2[i];
        }
    }

    return n;
}

fn vec_dot_f(vec1: &[f32], vec2: &[f32]) -> f32{
    let l1 = vec1.len();
    let l2 = vec2.len();

    let mut n : f32 = 0.0;

    if l1 != l2{
        panic!("Vectors are not same length");
    }

    else{
        for i in 0..l1{
            n += vec1[i] * vec2[i];
        }
    }

    return n;
}