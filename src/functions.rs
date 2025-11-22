pub fn ceil_div(x: usize, y: usize) -> usize {
    return (x + y - 1) / y;
}

pub fn floor_div(x: usize, y: usize) -> usize {
    return x / y;
}

pub fn get_mb(n_floats: usize) -> f32 {
    let n_bytes = (n_floats * 4) as f32;

    let n_mb = n_bytes / 1000000.0;

    return (n_mb * 10.0).round() / 10.0;
}
