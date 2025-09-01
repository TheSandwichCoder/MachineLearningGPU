use std::num::NonZeroU64;
use wgpu::util::DeviceExt;
use bytemuck::{Pod, Zeroable};

use crate::datatypes::*;
use crate::data_reader::DataReader;

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
    out_buffer: wgpu::Buffer, // retrieves parameters and debug stuff
    data_buffer: wgpu::Buffer, // holds current batch of data
    metric_buffer: wgpu::Buffer, // gets accuracy and metrics


    forward_pass_info: NNPassInfo,
    backward_pass_info: NNPassInfo,
    gradient_pass_info: NNPassInfo,
    error_pass_info: NNPassInfo,
    test_pass_info: NNPassInfo,

    // nn info
    pub nn_info: NeuralNetworkInfo,
    pub data_reader: DataReader,

}

impl NNDispatch{
    pub async fn new(nn_dim: &Vec<usize>, n_batches: u32, data_path: String, data_per_batch: u32, learning_rate: f32) -> Self{
        // GPU stuff
        let instance = wgpu::Instance::default();
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions::default())
        .await.expect("No GPU adapter found");

        let feats = wgpu::Features::PUSH_CONSTANTS;
        let limits = wgpu::Limits {
            max_push_constant_size: 32, // or adapter.limits().max_push_constant_size.min(128)
            ..wgpu::Limits::default()
        };

        let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("device"),
            required_features: feats,
            required_limits: limits,
            memory_hints: wgpu::MemoryHints::Performance, // or Default::default()
            trace: wgpu::Trace::Off,                      // <-- not Option
        })
        .await.expect("request_device failed");

        let align = device.limits().min_uniform_buffer_offset_alignment as u64;

        // ---------------------Neural Network Info---------------------
        let nn_info = NeuralNetworkInfo::new(nn_dim, n_batches as usize, learning_rate as f32);

        let (p_dir, a_dir) = nn_info.create_dirs();

        let test_metrics = TestMetrics::zero();

        // ---------------------Data Reader---------------------
        let mut data_reader = DataReader::new(data_path, (n_batches * data_per_batch) as usize, n_batches as usize);
        data_reader.initialise_params_mnist();
        data_reader.load_batch_mnist();
        
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
            size: (32768 * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let data_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
            label: Some("data buffer"),
            contents: bytemuck::cast_slice(&data_reader.get_buffer()),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        });

        let metric_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
            label: Some("metric buffer"),
            contents: bytemuck::bytes_of(&test_metrics),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
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

        // +--------------------------------------------------------+
        // |                                                        |
        // |                    Gradient Pass                       |
        // |                                                        |
        // +--------------------------------------------------------+

        let gradient_slot   = ((std::mem::size_of::<GradientDir>() as u64 + align - 1) / align) * align;
        let gradient_dir_buffer_size = backward_slot as u64 * nn_info.n_batches as u64;

        let gradient_dir_buffer = device.create_buffer(&wgpu::BufferDescriptor{
            label: Some("gradient_dir_buf"),
            size: gradient_dir_buffer_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ---------------------Special Binding---------------------
        let gradient_dir_binding = wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &gradient_dir_buffer,
            offset: 0,
            size: Some(wgpu::BufferSize::new(backward_slot).unwrap()),
        });

        // ---------------------Shader Sources---------------------
        let wgsl_src = include_str!("shaders/apply_gradients.wgsl");
        let gradient_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor{
            label: Some("gradient_shader"),
            source: wgpu::ShaderSource::Wgsl(wgsl_src.into()),
        });

        // ---------------------Bind Layout---------------------
        let gradient_bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gradient_bind_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
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
                        min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<GradientDir>() as u64),
                    },
                    count: None,
                },
            ],
        });

        let gradient_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor{
            label: Some("bg"),
            layout: &gradient_bind_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: param_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: act_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: gradient_dir_binding },
            ],
        });

            // ---------------------Update Meta---------------------

        for batch_i in 0..nn_info.n_batches{
            let gradient_dir = GradientDir::new(&nn_info, batch_i); 

            println!("{}", gradient_dir.batch_start_i);

            queue.write_buffer(&gradient_dir_buffer, batch_i as u64 * gradient_slot, bytemuck::bytes_of(&gradient_dir));
        }

        // ---------------------Pipeline Layout---------------------

        let gradient_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor{
            label: Some("gradient_pipeline_layout"),
            bind_group_layouts: &[&gradient_bind_layout],
            push_constant_ranges: &[],
        });

        let gradient_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gradient_pipeline"),
            layout: Some(&gradient_pipeline_layout),
            module: &gradient_shader,
            entry_point: Some("main"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        // +--------------------------------------------------------+
        // |                                                        |
        // |                      Error Pass                        |
        // |                                                        |
        // +--------------------------------------------------------+

        let error_slot   = ((std::mem::size_of::<ErrorDir>() as u64 + align - 1) / align) * align;
        let error_dir_buffer_size = backward_slot as u64 * 1;

        let error_dir = ErrorDir::new(&nn_info);

        let error_dir_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
            label: Some("error_dir_buf"),
            contents: bytemuck::bytes_of(&error_dir),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });


        // ---------------------Shader Sources---------------------
        let wgsl_src = include_str!("shaders/apply_error.wgsl");
        let error_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor{
            label: Some("error_shader"),
            source: wgpu::ShaderSource::Wgsl(wgsl_src.into()),
        });

        // ---------------------Bind Layout---------------------
        let error_bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("error_bind_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
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
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<ErrorDir>() as u64),
                    },
                    count: None,
                },
            ],
        });

        let error_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor{
            label: Some("bg"),
            layout: &error_bind_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: act_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: data_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: error_dir_buffer.as_entire_binding() },
            ],
        });

        // ---------------------Pipeline Layout---------------------

        let error_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor{
            label: Some("error_pipeline_layout"),
            bind_group_layouts: &[&error_bind_layout],
            push_constant_ranges: &[wgpu::PushConstantRange{
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..std::mem::size_of::<ErrorPC>() as u32, // bytes
            }],
        });

        let error_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("error_pipeline"),
            layout: Some(&error_pipeline_layout),
            module: &error_shader,
            entry_point: Some("main"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        // +--------------------------------------------------------+
        // |                                                        |
        // |                    Testing Pass                        |
        // |                                                        |
        // +--------------------------------------------------------+

        // ---------------------Shader Sources---------------------
        let wgsl_src = include_str!("shaders/test_model.wgsl");
        let test_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor{
            label: Some("test_shader"),
            source: wgpu::ShaderSource::Wgsl(wgsl_src.into()),
        });

        // ---------------------Bind Layout---------------------
        let test_bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("test_bind_layout"),
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
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage {read_only: false},
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<TestMetrics>() as u64),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<ErrorDir>() as u64),
                    },
                    count: None,
                },
            ],
        });

        let test_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor{
            label: Some("bg"),
            layout: &test_bind_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: act_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: data_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: metric_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: error_dir_buffer.as_entire_binding() },
            ],
        });

        // ---------------------Pipeline Layout---------------------

        let test_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor{
            label: Some("test_pipeline_layout"),
            bind_group_layouts: &[&test_bind_layout],
            push_constant_ranges: &[wgpu::PushConstantRange{
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..std::mem::size_of::<ErrorPC>() as u32, // bytes
            }],
        });

        let test_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("test_pipeline"),
            layout: Some(&test_pipeline_layout),
            module: &test_shader,
            entry_point: Some("main"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });
        


        let forward_pass_info = NNPassInfo::new(forward_dir_buffer, forward_shader, forward_bind_group, forward_pipeline, forward_slot);
        let backward_pass_info = NNPassInfo::new(backward_dir_buffer, backward_shader, backward_bind_group, backward_pipeline, backward_slot);
        let gradient_pass_info = NNPassInfo::new(gradient_dir_buffer, gradient_shader, gradient_bind_group, gradient_pipeline, gradient_slot);
        
        let error_pass_info = NNPassInfo::new(error_dir_buffer.clone(), error_shader, error_bind_group, error_pipeline, error_slot);
        let test_pass_info = NNPassInfo::new(error_dir_buffer.clone(), test_shader, test_bind_group, test_pipeline, error_slot);


        println!("SUCCESSFULLY INITIALISED GPU");

        return NNDispatch{
            instance,
            adapter,
            device,
            queue,

            param_buffer,
            act_buffer,
            out_buffer,
            data_buffer,
            metric_buffer,

            forward_pass_info,
            backward_pass_info,
            gradient_pass_info,
            error_pass_info,
            test_pass_info,

            nn_info,
            data_reader,
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

                pass.dispatch_workgroups(self.nn_info.get_dim_n(layer_i + 1) as u32, self.nn_info.get_dim_n(layer_i) as u32 + 1, self.nn_info.get_n_batches() as u32);
            }
        }

        let backward_commands = encoder.finish();
        self.queue.submit([backward_commands]); 
    }

    pub fn set_data(&self){
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor{ label: Some("encoder")});

        let data_slot = (self.data_reader.data_value_size + 1);
        // let curr_batch_start = self.data_reader.load_batch_i * self.data_reader.load_batch_length * data_slot  + self.data_reader.sub_batch_i * self.data_reader.sub_batch_length * data_slot;
        let curr_batch_start = self.data_reader.sub_batch_i * self.data_reader.sub_batch_length * data_slot;

        for batch_i in 0..self.nn_info.n_batches{
            let write_i = batch_i * self.nn_info.activity_info.a_length;

            let read_i = batch_i * (self.data_reader.data_value_size + 1) + curr_batch_start + 1; // plus one to skip the label

            encoder.copy_buffer_to_buffer(
                &self.data_buffer, 
                (read_i * 4) as u64, 
                &self.act_buffer, 
                (write_i * 4) as u64, 
                (self.data_reader.data_value_size * 4) as u64
            );
        }

        self.queue.submit([encoder.finish()]);
    }

    pub fn apply_gradients(&self){
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor{ label: Some("encoder")});

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cpass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&self.gradient_pass_info.pipeline);

            for batch_i in 0..self.nn_info.n_batches{
                let dyn_off = batch_i as u32 * self.gradient_pass_info.dir_slot_size as u32;
                pass.set_bind_group(0, &self.gradient_pass_info.bind_group, &[dyn_off]);

                pass.dispatch_workgroups(self.nn_info.p_length as u32 / 256 + 1, 256, 1);
                // pass.dispatch_workgroups(self.nn_info.p_length as u32, 1, 1);
            }
        }

        let gradient_commands = encoder.finish();
        self.queue.submit([gradient_commands]); 
    }

    pub fn apply_error(&self){
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor{ label: Some("encoder")});

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cpass"),
                timestamp_writes: None,
            });

            let params = ErrorPC::new(&self.data_reader);

            // println!("{}", params.batch_start_idx);

            pass.set_pipeline(&self.error_pass_info.pipeline);

            pass.set_bind_group(0, &self.error_pass_info.bind_group, &[]);

            pass.set_push_constants(0, bytemuck::bytes_of(&params));

            pass.dispatch_workgroups(self.nn_info.n_batches as u32, 1, 1);
        }

        let error_commands = encoder.finish();
        self.queue.submit([error_commands]); 
    }

    pub fn clear_metrics(&self){
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor{ label: Some("encoder")});
        
        encoder.clear_buffer(&self.metric_buffer, 0, Some(16));

        let error_commands = encoder.finish();
        self.queue.submit([error_commands]); 
    }

    pub fn update_metrics(&self){
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor{ label: Some("encoder")});
        
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cpass"),
                timestamp_writes: None,
            });

            let params = ErrorPC::new(&self.data_reader);

            pass.set_pipeline(&self.test_pass_info.pipeline);

            pass.set_bind_group(0, &self.test_pass_info.bind_group, &[]);

            pass.set_push_constants(0, bytemuck::bytes_of(&params));

            pass.dispatch_workgroups(self.nn_info.n_batches as u32, 1, 1);
        }

        let error_commands = encoder.finish();
        self.queue.submit([error_commands]); 
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

        drop(data);
        self.out_buffer.unmap();
    }

    pub fn read_back_metrics(&self){
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor{ label: Some("encoder") });
        
        encoder.copy_buffer_to_buffer(&self.metric_buffer, 0, &self.out_buffer, 0, 2 as u64 *4);

        self.queue.submit(Some(encoder.finish()));

        let slice = self.out_buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| ());
        self.device.poll(wgpu::PollType::Wait).unwrap();

        // Now it's safe to read.
        let data = slice.get_mapped_range();
        let out: &[u32] = bytemuck::cast_slice(&data);

        println!("{:?}", &out[..2]);

        drop(data);
        self.out_buffer.unmap();
    }

    pub fn read_back_data(&self){
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor{ label: Some("encoder") });
        
        encoder.copy_buffer_to_buffer(&self.data_buffer, 0, &self.out_buffer, 0, 1024);

        self.queue.submit(Some(encoder.finish()));

        let slice = self.out_buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| ());
        self.device.poll(wgpu::PollType::Wait).unwrap();

        // Now it's safe to read.
        let data = slice.get_mapped_range();
        let out: &[f32] = bytemuck::cast_slice(&data);

        println!("{:?}", out);
        drop(data);
        self.out_buffer.unmap();
    }

    pub fn read_back_act_single(&self){
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor{ label: Some("encoder") });
        
        encoder.copy_buffer_to_buffer(&self.act_buffer, 0, &self.out_buffer, 0, self.nn_info.activity_info.a_length as u64 *4);

        self.queue.submit(Some(encoder.finish()));

        let slice = self.out_buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| ());
        self.device.poll(wgpu::PollType::Wait).unwrap();

        // Now it's safe to read.
        let data = slice.get_mapped_range();
        let out: &[f32] = bytemuck::cast_slice(&data);

        println!("v: {:?}", &out[..self.nn_info.activity_info.g_start  as usize]);
        println!("g: {:?}", &out[self.nn_info.activity_info.g_start as usize..self.nn_info.activity_info.d_start  as usize]);
        println!("p1: {:?}", &out[self.nn_info.activity_info.d_start as usize..(self.nn_info.activity_info.d_start + self.nn_info.activity_info.a_deriv_buffer_size ) as usize]);
        println!("p2: {:?}", &out[(self.nn_info.activity_info.d_start + self.nn_info.activity_info.a_deriv_buffer_size) as usize..(self.nn_info.activity_info.d_start + self.nn_info.activity_info.a_deriv_buffer_size * 2) as usize]);

        

        drop(data);
        self.out_buffer.unmap();
    }

    pub fn read_back_raw(&self, n_floats: u64){
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor{ label: Some("encoder") });
        
        encoder.copy_buffer_to_buffer(&self.act_buffer, 0, &self.out_buffer, 0, n_floats*4);

        self.queue.submit(Some(encoder.finish()));

        let slice = self.out_buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| ());
        self.device.poll(wgpu::PollType::Wait).unwrap();

        // Now it's safe to read.
        let data = slice.get_mapped_range();
        let out: &[f32] = bytemuck::cast_slice(&data);

        println!("{:?}", &out[..n_floats as usize]);

        drop(data);
        self.out_buffer.unmap();
    }
}
