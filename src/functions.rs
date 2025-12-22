use std::io::Write;

pub fn ceil_div(x: usize, y: usize) -> usize {
    return (x + y - 1) / y;
}

pub fn ceil_div_inv_y(c: usize, x: usize) -> usize {
    return (x - 1) / (c - 1);
}

pub fn floor_div(x: usize, y: usize) -> usize {
    return x / y;
}

pub fn get_mb(n_floats: usize) -> f32 {
    let n_bytes = (n_floats * 4) as f32;

    let n_mb = n_bytes / 1000000.0;

    return (n_mb * 10.0).round() / 10.0;
}

pub fn get_gflops(n_flops: usize) -> f32 {
    let n_gflops: f32 = n_flops as f32 / 1000000000.0;

    return (n_gflops * 10.0).round() / 10.0;
}

pub fn show_progress(done: usize, sub_batch_i: usize, total: usize) {
    let bar_width = 40;
    let filled = done * bar_width / total;
    let empty = bar_width - filled;

    print!(
        "\r[{}{}] {:>3}% ({}/{}) (load-batch i: {})",
        "=".repeat(filled),
        " ".repeat(empty),
        done * 100 / total,
        done,
        total,
        sub_batch_i
    );
    std::io::stdout().flush().unwrap();
}
