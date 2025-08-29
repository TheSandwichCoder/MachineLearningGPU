use std::num::NonZeroU64;
use wgpu::util::DeviceExt;
use bytemuck::{Pod, Zeroable};

use crate::datatypes::{NeuralNetworkInfo, ForwardDir, BackwardDir};

pub struct NNPassInfo{
    pub dir_buffer: wgpu::Buffer, // for metas

    pub shader: wgpu::ShaderModule,

    pub bind_group: wgpu::BindGroup,

    pub pipeline: wgpu::ComputePipeline,

    pub dir_slot_size: u64,
}

impl NNPassInfo{
    pub fn new(b: wgpu::Buffer, s: wgpu::ShaderModule, bg: wgpu::BindGroup, p: wgpu::ComputePipeline, ss: u64) -> Self{
        return NNPassInfo{
            dir_buffer: b, // for metas
            shader: s,
            bind_group: bg,
            pipeline: p,
            dir_slot_size: ss,
        }
    }
}

pub struct NNDispatch{
    // gpu stuff
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,

    param_buffer: wgpu::Buffer, // holds the params for the model
    act_buffer: wgpu::Buffer, // holds intermediate layer outputs and gradients
    out_buffer: wgpu::Buffer, // this is only for info that we want (eg. accuracy)


    forward_pass_info: NNPassInfo,

    backward_pass_info: NNPassInfo,

    // nn info
    pub nn_info: NeuralNetworkInfo,

}

impl NNDispatch{
    pub async fn new(nn_dim: &Vec<usize>) -> Self{
        // GPU stuff
        let instance = wgpu::Instance::default();
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions::default())
        .await.expect("No GPU adapter found");

        let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::Performance, // or Default::default()
            trace: wgpu::Trace::Off,                      // <-- not Option
        })
        .await.expect("request_device failed");

        let align = device.limits().min_uniform_buffer_offset_alignment as u64;

        // ---------------------Neural Network Info---------------------
        let nn_info = NeuralNetworkInfo::new(nn_dim, 16);

        let (p_dir, a_dir) = nn_info.create_dirs();
        
        // ---------------------Buffer Stuff---------------------
        let param_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
            label: Some("params buffer"),
            contents: bytemuck::cast_slice(&p_dir.create_buffer()),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        });

        let act_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
            label: Some("activities buffer"),
            contents: bytemuck::cast_slice(&a_dir.create_buffer()),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        });

        let out_buffer = device.create_buffer(&wgpu::BufferDescriptor{
            label: Some("out_buf"),
            size: (1024 * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // +--------------------------------------------------------+
        // |                                                        |
        // |                    Forward Pass                        |
        // |                                                        |
        // +--------------------------------------------------------+


        let forward_slot   = ((std::mem::size_of::<ForwardDir>() as u64 + align - 1) / align) * align;
        let forward_dir_buffer_size = forward_slot * (nn_info.get_n_layers() - 1) as u64;

        let forward_dir_buffer = device.create_buffer(&wgpu::BufferDescriptor{
            label: Some("nn_dir_buf"),
            size: forward_dir_buffer_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ---------------------Special Binding---------------------
        let forward_dir_binding = wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &forward_dir_buffer,
            offset: 0,
            size: Some(wgpu::BufferSize::new(forward_slot).unwrap()),
        });


        // ---------------------Shader Sources---------------------
        let wgsl_src = include_str!("shaders/forward_nn.wgsl");
        let forward_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor{
            label: Some("forward_shader"),
            source: wgpu::ShaderSource::Wgsl(wgsl_src.into()),
        });

        // ---------------------Bind Layout---------------------
        let forward_bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("forward_bind_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<ForwardDir>() as u64),
                    },
                    count: None,
                },
            ],
        });

        

        let forward_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor{
            label: Some("bg"),
            layout: &forward_bind_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: param_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: act_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: forward_dir_binding },
            ],
        });

            // ---------------------Update Meta---------------------

        for layer_i in 0..nn_info.get_n_layers() - 1{
            let forward_dir = ForwardDir::new(&nn_info, layer_i); 

            println!("{}", forward_dir.read_layer_length);

            queue.write_buffer(&forward_dir_buffer, layer_i as u64 * forward_slot, bytemuck::bytes_of(&forward_dir));
        }

        // ---------------------Pipeline Layout---------------------

        let forward_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor{
            label: Some("forward_pipeline_layout"),
            bind_group_layouts: &[&forward_bind_layout],
            push_constant_ranges: &[],
        });

        let forward_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("forward_pipeline"),
            layout: Some(&forward_pipeline_layout),
            module: &forward_shader,
            entry_point: Some("main"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });


        
        // +--------------------------------------------------------+
        // |                                                        |
        // |                    Backward Pass                       |
        // |                                                        |
        // +--------------------------------------------------------+
        
        let backward_slot   = ((std::mem::size_of::<BackwardDir>() as u64 + align - 1) / align) * align;
        let backward_dir_buffer_size = backward_slot * (nn_info.get_n_layers() - 1) as u64;

        let backward_dir_buffer = device.create_buffer(&wgpu::BufferDescriptor{
            label: Some("backward_dir_buf"),
            size: backward_dir_buffer_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ---------------------Special Binding---------------------
        let backward_dir_binding = wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &backward_dir_buffer,
            offset: 0,
            size: Some(wgpu::BufferSize::new(backward_slot).unwrap()),
        });


        // ---------------------Shader Sources---------------------
        let wgsl_src = include_str!("shaders/backward_nn.wgsl");
        let backward_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor{
            label: Some("backward_shader"),
            source: wgpu::ShaderSource::Wgsl(wgsl_src.into()),
        });

        // ---------------------Bind Layout---------------------
        let backward_bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("backward_bind_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<BackwardDir>() as u64),
                    },
                    count: None,
                },
            ],
        });

        let backward_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor{
            label: Some("bg"),
            layout: &backward_bind_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: param_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: act_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: backward_dir_binding },
            ],
        });

            // ---------------------Update Meta---------------------

        for layer_i in 0..nn_info.get_n_layers() - 1{
            let backward_dir = BackwardDir::new(&nn_info, layer_i); 

            queue.write_buffer(&backward_dir_buffer, layer_i as u64 * backward_slot, bytemuck::bytes_of(&backward_dir));
        }

        // ---------------------Pipeline Layout---------------------

        let backward_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor{
            label: Some("backward_pipeline_layout"),
            bind_group_layouts: &[&backward_bind_layout],
            push_constant_ranges: &[],
        });

        let backward_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("backward_pipeline"),
            layout: Some(&backward_pipeline_layout),
            module: &backward_shader,
            entry_point: Some("main"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        let forward_pass_info = NNPassInfo::new(forward_dir_buffer, forward_shader, forward_bind_group, forward_pipeline, forward_slot);
        let backward_pass_info = NNPassInfo::new(backward_dir_buffer, backward_shader, backward_bind_group, backward_pipeline, backward_slot);

        println!("SUCCESSFULLY INITIALISED GPU");

        return NNDispatch{
            instance,
            adapter,
            device,
            queue,

            param_buffer,
            act_buffer,
            out_buffer,

            forward_pass_info,
            backward_pass_info,

            nn_info,
        }
    }

    pub fn forward(&self){
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor{ label: Some("encoder") });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cpass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&self.forward_pass_info.pipeline);

            for layer_i in 0..(self.nn_info.get_n_layers() - 1){
                let dyn_off = layer_i as u32 * self.forward_pass_info.dir_slot_size as u32;
                pass.set_bind_group(0, &self.forward_pass_info.bind_group, &[dyn_off]);

                pass.dispatch_workgroups(self.nn_info.get_dim_n(layer_i + 1) as u32, self.nn_info.get_n_batches() as u32, 1);
            }
        }

        let forward_commands = encoder.finish();
        self.queue.submit([forward_commands]);  
    }

    pub fn backward(&self){
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor{ label: Some("encoder")});

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cpass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&self.backward_pass_info.pipeline);

            for layer_i in (0..(self.nn_info.get_n_layers() - 1)).rev(){
                let dyn_off = layer_i as u32 * self.backward_pass_info.dir_slot_size as u32;
                pass.set_bind_group(0, &self.backward_pass_info.bind_group, &[dyn_off]);

                pass.dispatch_workgroups(self.nn_info.get_dim_n(layer_i + 1) as u32, self.nn_info.get_n_batches() as u32, 1);
            }
        }

        let backward_commands = encoder.finish();
        self.queue.submit([backward_commands]); 
    }

    pub fn read_back_params(&self){
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor{ label: Some("encoder") });
        
        encoder.copy_buffer_to_buffer(&self.param_buffer, 0, &self.out_buffer, 0, self.nn_info.p_length as u64 *4);

        self.queue.submit(Some(encoder.finish()));

        let slice = self.out_buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| ());
        self.device.poll(wgpu::PollType::Wait).unwrap();

        // Now it's safe to read.
        let data = slice.get_mapped_range();
        let out: &[f32] = bytemuck::cast_slice(&data);

        self.nn_info.read_readback(out);
    }

    pub fn read_back_raw(&self){
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor{ label: Some("encoder") });
        
        encoder.copy_buffer_to_buffer(&self.act_buffer, 0, &self.out_buffer, 0, 1024);

        self.queue.submit(Some(encoder.finish()));

        let slice = self.out_buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| ());
        self.device.poll(wgpu::PollType::Wait).unwrap();

        // Now it's safe to read.
        let data = slice.get_mapped_range();
        let out: &[f32] = bytemuck::cast_slice(&data);

        println!("{:?}", out);
    }
}
