use std::num::NonZeroU64;
use wgpu::util::DeviceExt;
use bytemuck::{Pod, Zeroable};

mod datatypes;
mod dispatch;

use crate::datatypes::*;
use crate::dispatch::*;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Meta {
    len: u32,
    _pad: [u32; 6],
    scale: f32, // note: uniform layout rules are strict; this keeps it simple
    // Uniform buffers are read in 16B chunks; extra padding is fine.
}




fn main() {
    // pollster::block_on(run());
    let mut gpu = pollster::block_on(NNDispatch::new(&vec![2, 3, 2]));
    gpu.nn_info.show_all_specs();
    gpu.forward();
    
    gpu.backward();
    gpu.apply_gradients();
    
    gpu.forward();
    
    gpu.read_back_raw();
    // gpu.read_back_params();

}

// async fn run() {
    
    
//     // ---------- 1) Instance/Adapter/Device/Queue ----------
//     let instance = wgpu::Instance::default();
//     let adapter = instance.request_adapter(
//         &wgpu::RequestAdapterOptions::default(),
//     )
//     .await.expect("No GPU adapter found");
    
//     let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor {
//         required_features: wgpu::Features::empty(),
//         required_limits: wgpu::Limits::default(),
//         memory_hints: wgpu::MemoryHints::Performance, // or Default::default()
//         trace: wgpu::Trace::Off,                      // <-- not Option
//     })
//     .await
//     .expect("request_device failed");

//     // ---------- 2) Prepare data ----------
//     let input: Vec<f32> = (0..1024).map(|i| i as f32).collect();
//     let len = input.len() as u32;
//     let scale = 3.5_f32;

//     // ---------- 3) Create GPU buffers ----------
//     let in_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
//         label: Some("in_buf"),
//         contents: bytemuck::cast_slice(&input),
//         usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
//     });

//     let out_buf = device.create_buffer(&wgpu::BufferDescriptor{
//         label: Some("out_buf"),
//         size: (input.len() * std::mem::size_of::<f32>()) as u64,
//         usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
//         mapped_at_creation: false,
//     });

//     // Uniform (len + scale); we’ll keep alignment simple with padding.
//     let meta = Meta { len, _pad:[0;6], scale};
//     let meta_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
//         label: Some("meta_buf"),
//         contents: bytemuck::bytes_of(&meta),
//         usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
//     });

//     // ---------- 4) Shader & pipeline ----------
//     let wgsl_src = include_str!("shaders/scale.wgsl");
//     let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor{
//         label: Some("scale_shader"),
//         source: wgpu::ShaderSource::Wgsl(wgsl_src.into()),
//     });

//     let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
//         label: Some("bind_layout"),
//         entries: &[
//             wgpu::BindGroupLayoutEntry {
//                 binding: 0,
//                 visibility: wgpu::ShaderStages::COMPUTE,
//                 ty: wgpu::BindingType::Buffer {
//                     ty: wgpu::BufferBindingType::Storage { read_only: true },
//                     has_dynamic_offset: false,
//                     min_binding_size: None,
//                 },
//                 count: None,
//             },
//             wgpu::BindGroupLayoutEntry {
//                 binding: 1,
//                 visibility: wgpu::ShaderStages::COMPUTE,
//                 ty: wgpu::BindingType::Buffer {
//                     ty: wgpu::BufferBindingType::Storage { read_only: false },
//                     has_dynamic_offset: false,
//                     min_binding_size: None,
//                 },
//                 count: None,
//             },
//             wgpu::BindGroupLayoutEntry {
//                 binding: 2,
//                 visibility: wgpu::ShaderStages::COMPUTE,
//                 ty: wgpu::BindingType::Buffer {
//                     ty: wgpu::BufferBindingType::Uniform,
//                     has_dynamic_offset: false,
//                     min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<Meta>() as u64),
//                 },
//                 count: None,
//             },
//         ],
//     });

//     let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor{
//         label: Some("pipeline_layout"),
//         bind_group_layouts: &[&bind_layout],
//         push_constant_ranges: &[],
//     });

//     let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
//         label: Some("scale_pipeline"),
//         layout: Some(&pipeline_layout),
//         module: &shader,
//         entry_point: Some("main"),
//         cache: None,
//         compilation_options: wgpu::PipelineCompilationOptions::default(),
//     });

//     let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor{
//         label: Some("bind_group"),
//         layout: &bind_layout,
//         entries: &[
//             wgpu::BindGroupEntry { binding: 0, resource: in_buf.as_entire_binding() },
//             wgpu::BindGroupEntry { binding: 1, resource: out_buf.as_entire_binding() },
//             wgpu::BindGroupEntry { binding: 2, resource: meta_buf.as_entire_binding() },
//         ],
//     });

//     // ---------- 5) Dispatch ----------
//     let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor{ label: Some("encoder") });
//     {
//         let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
//             label: Some("cpass"),
//             timestamp_writes: None,
//         });
//         cpass.set_pipeline(&pipeline);
//         cpass.set_bind_group(0, &bind_group, &[]);
//         let workgroup_size = 256u32;
//         let groups = (len + workgroup_size - 1) / workgroup_size;
//         cpass.dispatch_workgroups(groups, 1, 1);
//     }

//     // We’ll copy results to a mappable buffer to read them on CPU.
//     let readback = device.create_buffer(&wgpu::BufferDescriptor{
//         label: Some("readback"),
//         size: (input.len() * std::mem::size_of::<f32>()) as u64,
//         usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
//         mapped_at_creation: false,
//     });

//     encoder.copy_buffer_to_buffer(&out_buf, 0, &readback, 0, (input.len() * 4) as u64);
//     queue.submit(Some(encoder.finish()));

//     // ---------- 6) Read results (for learning/testing only) ----------
//     {
//         let slice = readback.slice(..);
//         slice.map_async(wgpu::MapMode::Read, |_| ());
//         device.poll(wgpu::PollType::Wait).unwrap();

//         // Now it's safe to read.
//         let data = slice.get_mapped_range();
//         let out: &[f32] = bytemuck::cast_slice(&data);

//         // (Optional) print a few values
//         for i in 0..5 {
//             println!("{i}: {}", out[i]);
//         }
//     }
// }
