
#[derive(Clone)]
pub struct KernalRange{
    start_idx: i32,
    end_idx: i32,
}

impl KernalRange{
    pub fn new(start: i32, end: i32) -> Self{
        return KernalRange{
            start_idx: start,
            end_idx: end,
        }
    }
}