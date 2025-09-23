
#[derive(Clone)]
pub struct KernalRange{
    pub start_idx: i32,
    pub end_idx: i32,
}

impl KernalRange{
    pub fn new(start: i32, end: i32) -> Self{
        return KernalRange{
            start_idx: start,
            end_idx: end,
        }
    }
}