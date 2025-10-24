use bytemuck::{Pod, Zeroable};
use std::fs;
use std::num::NonZeroU64;
use wgpu::util::DeviceExt;

use crate::data_reader::DataReader;
use crate::datatypes::*;
use crate::datatypes::{nn_datatypes::*, workgroup::*};
use crate::dispatch::gpu_instance::*;
use crate::functions::*;
use crate::gpu_dirs::nn_dirs::*;

pub struct NNPassInfo {
    pub dir_buffer: wgpu::Buffer, // for metas

    pub shader: wgpu::ShaderModule,

    pub bind_group: wgpu::BindGroup,

    pub pipeline: wgpu::ComputePipeline,

    pub dir_slot_size: u64,

    pub workgroup_dim: WorkgroupDim,
}

impl NNPassInfo {
    pub fn new(
        b: wgpu::Buffer,
        s: wgpu::ShaderModule,
        bg: wgpu::BindGroup,
        p: wgpu::ComputePipeline,
        ss: u64,
        workgroup_dim: Vec<usize>,
    ) -> Self {
        return NNPassInfo {
            dir_buffer: b, // for metas
            shader: s,
            bind_group: bg,
            pipeline: p,
            dir_slot_size: ss,
            workgroup_dim: WorkgroupDim {
                x: workgroup_dim[0],
                y: workgroup_dim[1],
                z: workgroup_dim[2],
            },
        };
    }
}

pub struct NNDispatch {
    param_buffer: wgpu::Buffer,    // holds the params for the model
    gradient_buffer: wgpu::Buffer, // holds the param gradients
    momentum_buffer: wgpu::Buffer, // holds the param momentum
    act_buffer: wgpu::Buffer,      // holds intermediate layer outputs and gradients
    out_buffer: wgpu::Buffer,      // retrieves parameters and debug stuff
    data_buffer: wgpu::Buffer,     // holds current batch of data
    metric_buffer: wgpu::Buffer,   // gets accuracy and metrics

    forward_mat_pass_info: NNPassInfo,
    backward_deriv_mat_pass_info: NNPassInfo,
    backward_gradient_mat_pass_info: NNPassInfo,
    momentum_pass_info: NNPassInfo,
    error_pass_info: NNPassInfo,
    test_pass_info: NNPassInfo,

    // nn info
    pub nn_info: NeuralNetworkInfo,
    pub data_reader: DataReader,
}

impl NNDispatch {
    pub fn new(
        gpu_instance: &GPUInstance,
        nn_dim: &Vec<usize>,
        n_batches: u32,
        data_path: String,
        data_per_batch: u32,
        learning_rate: f32,
        momentum_rate: f32,
    ) -> Self {
        // ---------------------Neural Network Info---------------------
        let nn_info =
            NeuralNetworkInfo::new(nn_dim, n_batches as usize, learning_rate, momentum_rate);

        let (p_dir, a_dir) = nn_info.create_dirs();

        let test_metrics = TestMetrics::zero();

        // ---------------------Data Reader---------------------
        let mut data_reader = DataReader::new(
            data_path,
            (n_batches * data_per_batch) as usize,
            n_batches as usize,
        );
        // data_reader.initialise_params_debug();
        // data_reader.load_batch_debug();
        data_reader.initialise_params_mnist();
        data_reader.load_batch_mnist();

        // ---------------------Buffer Stuff---------------------
        let param_buffer =
            gpu_instance
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("params buffer"),
                    contents: bytemuck::cast_slice(&p_dir.create_buffer()),
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                });
        let gradient_buffer =
            gpu_instance
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("gradient buffer"),
                    contents: bytemuck::cast_slice(&p_dir.create_buffer_empty()),
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                });

        let momentum_buffer =
            gpu_instance
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("momentum buffer"),
                    contents: bytemuck::cast_slice(&p_dir.create_buffer_empty()),
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                });

        let act_buffer =
            gpu_instance
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("activities buffer"),
                    contents: bytemuck::cast_slice(&a_dir.create_buffer()),
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                });

        let out_buffer = gpu_instance.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("out_buf"),
            size: (p_dir.buffer_size * std::mem::size_of::<f32>()) as u64,
            // size: (2000 * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let data_buffer =
            gpu_instance
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("data buffer"),
                    contents: bytemuck::cast_slice(&data_reader.get_buffer()),
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                });

        let metric_buffer =
            gpu_instance
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("metric buffer"),
                    contents: bytemuck::bytes_of(&test_metrics),
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                });

        // +--------------------------------------------------------+
        // |                                                        |
        // |                     GEMM Configs                       |
        // |                                                        |
        // +--------------------------------------------------------+

        // ---------------------Bind Layout---------------------
        let gemm_bind_layout =
            gpu_instance
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("gemm_bind_layout"),
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
                                min_binding_size: wgpu::BufferSize::new(
                                    std::mem::size_of::<MatrixDir>() as u64,
                                ),
                            },
                            count: None,
                        },
                    ],
                });

        let gemm_mat_slot = ((std::mem::size_of::<MatrixDir>() as u64 + gpu_instance.align - 1)
            / gpu_instance.align)
            * gpu_instance.align;

        // +--------------------------------------------------------+
        // |                                                        |
        // |                   Forward Mat Pass                     |
        // |                                                        |
        // +--------------------------------------------------------+

        let forward_mat_dir_buffer_size = gemm_mat_slot * (nn_info.get_n_layers() - 1) as u64;

        let forward_mat_dir_buffer = gpu_instance.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("nn_dir_buf"),
            size: forward_mat_dir_buffer_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ---------------------Special Binding---------------------
        let forward_mat_dir_binding = wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &forward_mat_dir_buffer,
            offset: 0,
            size: Some(wgpu::BufferSize::new(gemm_mat_slot).unwrap()),
        });

        // ---------------------Shader Sources---------------------
        let wgsl_src = include_str!("../shaders/gemm_op/gemm_i_io.wgsl");
        let forward_mat_shader =
            gpu_instance
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("forward_gemm_shader"),
                    source: wgpu::ShaderSource::Wgsl(wgsl_src.into()),
                });

        let forward_mat_bind_group =
            gpu_instance
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("bg"),
                    layout: &gemm_bind_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: param_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: act_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: forward_mat_dir_binding,
                        },
                    ],
                });

        // ---------------------Update Meta---------------------

        for layer_i in 0..nn_info.get_n_layers() - 1 {
            let mat_dir = MatrixDir::new_forward(&nn_info, layer_i);

            gpu_instance.queue.write_buffer(
                &forward_mat_dir_buffer,
                layer_i as u64 * gemm_mat_slot,
                bytemuck::bytes_of(&mat_dir),
            );
        }

        // ---------------------Pipeline Layout---------------------

        let forward_mat_pipeline_layout =
            gpu_instance
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("forward_gemm_pipeline_layout"),
                    bind_group_layouts: &[&gemm_bind_layout],
                    push_constant_ranges: &[],
                });

        let forward_mat_pipeline =
            gpu_instance
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("forward_gemm_pipeline"),
                    layout: Some(&forward_mat_pipeline_layout),
                    module: &forward_mat_shader,
                    entry_point: Some("main"),
                    cache: None,
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                });

        // +--------------------------------------------------------+
        // |                                                        |
        // |                 Backward Deriv Pass                    |
        // |                                                        |
        // +--------------------------------------------------------+

        let backward_deriv_mat_dir_buffer_size =
            gemm_mat_slot * (nn_info.get_n_layers() - 1) as u64;

        let backward_deriv_mat_dir_buffer =
            gpu_instance.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("nn_dir_buf"),
                size: backward_deriv_mat_dir_buffer_size,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

        // ---------------------Special Binding---------------------
        let backward_deriv_mat_dir_binding = wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &backward_deriv_mat_dir_buffer,
            offset: 0,
            size: Some(wgpu::BufferSize::new(gemm_mat_slot).unwrap()),
        });

        // ---------------------Shader Sources---------------------
        let wgsl_src = include_str!("../shaders/gemm_op/gemm_i_io.wgsl");
        let backward_deriv_mat_shader =
            gpu_instance
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("backward_deriv_gemm_shader"),
                    source: wgpu::ShaderSource::Wgsl(wgsl_src.into()),
                });

        let backward_deriv_mat_bind_group =
            gpu_instance
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("bg"),
                    layout: &gemm_bind_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: param_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: act_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: backward_deriv_mat_dir_binding,
                        },
                    ],
                });

        // ---------------------Update Meta---------------------

        for layer_i in 1..nn_info.get_n_layers() {
            let mat_dir = MatrixDir::new_backward_deriv(&nn_info, layer_i);

            gpu_instance.queue.write_buffer(
                &backward_deriv_mat_dir_buffer,
                (layer_i - 1) as u64 * gemm_mat_slot,
                bytemuck::bytes_of(&mat_dir),
            );
        }

        // ---------------------Pipeline Layout---------------------

        let backward_deriv_mat_pipeline_layout =
            gpu_instance
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("backward_deriv_gemm_pipeline_layout"),
                    bind_group_layouts: &[&gemm_bind_layout],
                    push_constant_ranges: &[],
                });

        let backward_deriv_mat_pipeline =
            gpu_instance
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("backward_deriv_gemm_pipeline"),
                    layout: Some(&backward_deriv_mat_pipeline_layout),
                    module: &backward_deriv_mat_shader,
                    entry_point: Some("main"),
                    cache: None,
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                });

        // +--------------------------------------------------------+
        // |                                                        |
        // |               Backward Gradient Pass                   |
        // |                                                        |
        // +--------------------------------------------------------+

        let backward_gradient_mat_dir_buffer_size =
            gemm_mat_slot * (nn_info.get_n_layers() - 1) as u64;

        let backward_gradient_mat_dir_buffer =
            gpu_instance.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("nn_dir_buf"),
                size: backward_gradient_mat_dir_buffer_size,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

        // ---------------------Special Binding---------------------
        let backward_gradient_mat_dir_binding =
            wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &backward_gradient_mat_dir_buffer,
                offset: 0,
                size: Some(wgpu::BufferSize::new(gemm_mat_slot).unwrap()),
            });

        // ---------------------Shader Sources---------------------
        let wgsl_src = include_str!("../shaders/gemm_op/gemm_i_o.wgsl");
        let backward_gradient_mat_shader =
            gpu_instance
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("backward_gradient_mat_gemm_shader"),
                    source: wgpu::ShaderSource::Wgsl(wgsl_src.into()),
                });

        let backward_gradient_mat_bind_group =
            gpu_instance
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("bg"),
                    layout: &gemm_bind_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: act_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: gradient_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: backward_gradient_mat_dir_binding,
                        },
                    ],
                });

        // ---------------------Update Meta---------------------

        for layer_i in 0..nn_info.get_n_layers() - 1 {
            let mat_dir = MatrixDir::new_backward_gradients(&nn_info, layer_i);

            gpu_instance.queue.write_buffer(
                &backward_gradient_mat_dir_buffer,
                layer_i as u64 * gemm_mat_slot,
                bytemuck::bytes_of(&mat_dir),
            );
        }

        // ---------------------Pipeline Layout---------------------

        let backward_gradient_mat_pipeline_layout =
            gpu_instance
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("backward_gradient_gemm_pipeline_layout"),
                    bind_group_layouts: &[&gemm_bind_layout],
                    push_constant_ranges: &[],
                });

        let backward_gradient_mat_pipeline =
            gpu_instance
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("backward_gradient_gemm_pipeline"),
                    layout: Some(&backward_gradient_mat_pipeline_layout),
                    module: &backward_gradient_mat_shader,
                    entry_point: Some("main"),
                    cache: None,
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                });

        // +--------------------------------------------------------+
        // |                                                        |
        // |                    Momentum Pass                       |
        // |                                                        |
        // +--------------------------------------------------------+

        let momentum_slot = ((std::mem::size_of::<FlatApplyDir>() as u64 + gpu_instance.align - 1)
            / gpu_instance.align)
            * gpu_instance.align;
        let momentum_dir_buffer_size = momentum_slot as u64 * nn_info.n_batches as u64;

        let momentum_dir_buffer = gpu_instance.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("momentum_dir_buf"),
            size: momentum_dir_buffer_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ---------------------Special Binding---------------------
        let momentum_dir_binding = wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &momentum_dir_buffer,
            offset: 0,
            size: Some(wgpu::BufferSize::new(momentum_slot).unwrap()),
        });

        // ---------------------Shader Sources---------------------
        let wgsl_src = include_str!("../shaders/apply_op/apply_momentum.wgsl");
        let momentum_shader =
            gpu_instance
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("momentum_shader"),
                    source: wgpu::ShaderSource::Wgsl(wgsl_src.into()),
                });

        // ---------------------Bind Layout---------------------
        let momentum_bind_layout =
            gpu_instance
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("momentum_bind_layout"),
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
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: true,
                                min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<
                                    FlatApplyDir,
                                >(
                                )
                                    as u64),
                            },
                            count: None,
                        },
                    ],
                });

        let momentum_bind_group =
            gpu_instance
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("bg"),
                    layout: &momentum_bind_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: param_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: momentum_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: gradient_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: momentum_dir_binding,
                        },
                    ],
                });

        // ---------------------Update Meta---------------------

        for batch_i in 0..nn_info.n_batches {
            let momentum_dir = FlatApplyDir::new_nn(&nn_info, batch_i);

            gpu_instance.queue.write_buffer(
                &momentum_dir_buffer,
                batch_i as u64 * momentum_slot,
                bytemuck::bytes_of(&momentum_dir),
            );
        }

        // ---------------------Pipeline Layout---------------------

        let momentum_pipeline_layout =
            gpu_instance
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("momentum_pipeline_layout"),
                    bind_group_layouts: &[&momentum_bind_layout],
                    push_constant_ranges: &[],
                });

        let momentum_pipeline =
            gpu_instance
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("momentum_pipeline"),
                    layout: Some(&momentum_pipeline_layout),
                    module: &momentum_shader,
                    entry_point: Some("main"),
                    cache: None,
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                });

        // +--------------------------------------------------------+
        // |                                                        |
        // |                      Error Pass                        |
        // |                                                        |
        // +--------------------------------------------------------+

        let error_slot = ((std::mem::size_of::<ErrorDir>() as u64 + gpu_instance.align - 1)
            / gpu_instance.align)
            * gpu_instance.align;
        let error_dir_buffer_size = error_slot as u64 * 1;

        let error_dir = ErrorDir::new(&nn_info);

        let error_dir_buffer =
            gpu_instance
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("error_dir_buf"),
                    contents: bytemuck::bytes_of(&error_dir),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });

        // ---------------------Shader Sources---------------------
        let wgsl_src = include_str!("../shaders/apply_op/apply_error.wgsl");
        let error_shader = gpu_instance
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("error_shader"),
                source: wgpu::ShaderSource::Wgsl(wgsl_src.into()),
            });

        // ---------------------Bind Layout---------------------
        let error_bind_layout =
            gpu_instance
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
                                min_binding_size: wgpu::BufferSize::new(
                                    std::mem::size_of::<ErrorDir>() as u64,
                                ),
                            },
                            count: None,
                        },
                    ],
                });

        let error_bind_group = gpu_instance
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bg"),
                layout: &error_bind_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: act_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: data_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: error_dir_buffer.as_entire_binding(),
                    },
                ],
            });

        // ---------------------Pipeline Layout---------------------

        let error_pipeline_layout =
            gpu_instance
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("error_pipeline_layout"),
                    bind_group_layouts: &[&error_bind_layout],
                    push_constant_ranges: &[wgpu::PushConstantRange {
                        stages: wgpu::ShaderStages::COMPUTE,
                        range: 0..std::mem::size_of::<ErrorPC>() as u32, // bytes
                    }],
                });

        let error_pipeline =
            gpu_instance
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
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
        let wgsl_src = include_str!("../shaders/test_model.wgsl");
        let test_shader = gpu_instance
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("test_shader"),
                source: wgpu::ShaderSource::Wgsl(wgsl_src.into()),
            });

        // ---------------------Bind Layout---------------------
        let test_bind_layout =
            gpu_instance
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<
                                    TestMetrics,
                                >(
                                )
                                    as u64),
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: wgpu::BufferSize::new(
                                    std::mem::size_of::<ErrorDir>() as u64,
                                ),
                            },
                            count: None,
                        },
                    ],
                });

        let test_bind_group = gpu_instance
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bg"),
                layout: &test_bind_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: act_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: data_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: metric_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: error_dir_buffer.as_entire_binding(),
                    },
                ],
            });

        // ---------------------Pipeline Layout---------------------

        let test_pipeline_layout =
            gpu_instance
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("test_pipeline_layout"),
                    bind_group_layouts: &[&test_bind_layout],
                    push_constant_ranges: &[wgpu::PushConstantRange {
                        stages: wgpu::ShaderStages::COMPUTE,
                        range: 0..std::mem::size_of::<ErrorPC>() as u32, // bytes
                    }],
                });

        let test_pipeline =
            gpu_instance
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("test_pipeline"),
                    layout: Some(&test_pipeline_layout),
                    module: &test_shader,
                    entry_point: Some("main"),
                    cache: None,
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                });

        let gem_wg = vec![16, 16, 0];

        let forward_mat_pass_info = NNPassInfo::new(
            forward_mat_dir_buffer,
            forward_mat_shader,
            forward_mat_bind_group,
            forward_mat_pipeline,
            gemm_mat_slot,
            gem_wg.clone(),
        );

        let backward_deriv_mat_pass_info = NNPassInfo::new(
            backward_deriv_mat_dir_buffer,
            backward_deriv_mat_shader,
            backward_deriv_mat_bind_group,
            backward_deriv_mat_pipeline,
            gemm_mat_slot,
            gem_wg.clone(),
        );
        let backward_gradient_mat_pass_info = NNPassInfo::new(
            backward_gradient_mat_dir_buffer,
            backward_gradient_mat_shader,
            backward_gradient_mat_bind_group,
            backward_gradient_mat_pipeline,
            gemm_mat_slot,
            gem_wg.clone(),
        );

        let momentum_pass_info = NNPassInfo::new(
            momentum_dir_buffer,
            momentum_shader,
            momentum_bind_group,
            momentum_pipeline,
            momentum_slot,
            vec![256, 0, 0],
        );

        let error_pass_info = NNPassInfo::new(
            error_dir_buffer.clone(),
            error_shader,
            error_bind_group,
            error_pipeline,
            error_slot,
            vec![64, 0, 0],
        );
        let test_pass_info = NNPassInfo::new(
            error_dir_buffer.clone(),
            test_shader,
            test_bind_group,
            test_pipeline,
            error_slot,
            vec![64, 0, 0],
        );

        println!("SUCCESSFULLY INITIALISED GPU");

        return NNDispatch {
            param_buffer,
            gradient_buffer,
            momentum_buffer,
            act_buffer,
            out_buffer,
            data_buffer,
            metric_buffer,

            forward_mat_pass_info,
            backward_deriv_mat_pass_info,
            backward_gradient_mat_pass_info,
            momentum_pass_info,
            error_pass_info,
            test_pass_info,

            nn_info,
            data_reader,
        };
    }

    pub fn forward_mat(&self, gpu_instance: &GPUInstance) {
        let mut encoder =
            gpu_instance
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("encoder"),
                });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cpass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&self.forward_mat_pass_info.pipeline);

            for layer_i in 0..(self.nn_info.get_n_layers() - 1) {
                let dyn_off = layer_i as u32 * self.forward_mat_pass_info.dir_slot_size as u32;
                pass.set_bind_group(0, &self.forward_mat_pass_info.bind_group, &[dyn_off]);

                let gx = ceil_div(
                    self.nn_info.get_dim_n(layer_i + 1),
                    self.forward_mat_pass_info.workgroup_dim.x,
                );
                let gy = ceil_div(
                    self.nn_info.n_batches,
                    self.forward_mat_pass_info.workgroup_dim.y,
                );

                pass.dispatch_workgroups(gx as u32, gy as u32, 1);
            }
        }

        let forward_commands = encoder.finish();
        gpu_instance.queue.submit([forward_commands]);
        // self.gpu_instance.device.poll(wgpu::PollType::Wait).unwrap();
    }

    pub fn backward_mat(&self, gpu_instance: &GPUInstance) {
        let mut encoder =
            gpu_instance
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("encoder"),
                });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cpass"),
                timestamp_writes: None,
            });

            for layer_i in (1..self.nn_info.get_n_layers()).rev() {
                let dyn_off =
                    (layer_i - 1) as u32 * self.backward_deriv_mat_pass_info.dir_slot_size as u32;

                pass.set_pipeline(&self.backward_deriv_mat_pass_info.pipeline);
                if layer_i != self.nn_info.get_n_layers() - 1 {
                    pass.set_bind_group(
                        0,
                        &self.backward_deriv_mat_pass_info.bind_group,
                        &[dyn_off],
                    );

                    let gx = ceil_div(
                        self.nn_info.layer_dim[layer_i],
                        self.backward_deriv_mat_pass_info.workgroup_dim.x,
                    );
                    let gy = ceil_div(
                        self.nn_info.n_batches,
                        self.backward_deriv_mat_pass_info.workgroup_dim.y,
                    );

                    pass.dispatch_workgroups(gx as u32, gy as u32, 1);
                }

                pass.set_pipeline(&self.backward_gradient_mat_pass_info.pipeline);

                pass.set_bind_group(
                    0,
                    &self.backward_gradient_mat_pass_info.bind_group,
                    &[dyn_off],
                );

                let gx = ceil_div(
                    self.nn_info.layer_dim[layer_i],
                    self.backward_gradient_mat_pass_info.workgroup_dim.x,
                );
                let gy = ceil_div(
                    self.nn_info.layer_dim[layer_i - 1],
                    self.backward_gradient_mat_pass_info.workgroup_dim.y,
                );

                pass.dispatch_workgroups(gx as u32, gy as u32, 1);
            }
        }

        let backward_commands = encoder.finish();
        gpu_instance.queue.submit([backward_commands]);
    }

    pub fn set_data(&self, gpu_instance: &GPUInstance) {
        let mut encoder =
            gpu_instance
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("encoder"),
                });

        let data_slot = (self.data_reader.data_value_size + 1);
        // let curr_batch_start = self.data_reader.load_batch_i * self.data_reader.load_batch_length * data_slot  + self.data_reader.sub_batch_i * self.data_reader.sub_batch_length * data_slot;
        let curr_batch_start =
            self.data_reader.sub_batch_i * self.data_reader.sub_batch_length * data_slot;

        for batch_i in 0..self.nn_info.n_batches {
            let write_i_swap = batch_i * self.nn_info.activity_info.a_length;
            let write_i_storage =
                batch_i * self.nn_info.activity_info.a_length + self.nn_info.activity_info.s_start;

            let read_i = batch_i * (self.data_reader.data_value_size + 1) + curr_batch_start + 1; // plus one to skip the label

            encoder.copy_buffer_to_buffer(
                &self.data_buffer,
                (read_i * 4) as u64,
                &self.act_buffer,
                (write_i_swap * 4) as u64,
                (self.data_reader.data_value_size * 4) as u64,
            );

            encoder.copy_buffer_to_buffer(
                &self.data_buffer,
                (read_i * 4) as u64,
                &self.act_buffer,
                (write_i_storage * 4) as u64,
                (self.data_reader.data_value_size * 4) as u64,
            );
        }

        gpu_instance.queue.submit([encoder.finish()]);
    }

    pub fn update_momentum(&self, gpu_instance: &GPUInstance) {
        let mut encoder =
            gpu_instance
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("encoder"),
                });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cpass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&self.momentum_pass_info.pipeline);

            for batch_i in 0..self.nn_info.n_batches {
                let dyn_off = batch_i as u32 * self.momentum_pass_info.dir_slot_size as u32;
                pass.set_bind_group(0, &self.momentum_pass_info.bind_group, &[dyn_off]);

                let gx = ceil_div(
                    self.nn_info.p_length,
                    self.momentum_pass_info.workgroup_dim.x,
                );

                pass.dispatch_workgroups(gx as u32, 1, 1);
            }
        }

        let momentum_commands = encoder.finish();
        gpu_instance.queue.submit([momentum_commands]);
    }

    pub fn apply_error(&self, gpu_instance: &GPUInstance) {
        let mut encoder =
            gpu_instance
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("encoder"),
                });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cpass"),
                timestamp_writes: None,
            });

            let params = ErrorPC::new(&self.data_reader, self.nn_info.n_batches);

            // println!("{}", params.batch_start_idx);

            pass.set_pipeline(&self.error_pass_info.pipeline);

            pass.set_bind_group(0, &self.error_pass_info.bind_group, &[]);

            pass.set_push_constants(0, bytemuck::bytes_of(&params));

            let gx = ceil_div(self.nn_info.n_batches, self.error_pass_info.workgroup_dim.x);

            pass.dispatch_workgroups(gx as u32, 1, 1);
        }

        let error_commands = encoder.finish();
        gpu_instance.queue.submit([error_commands]);
    }

    pub fn clear_metrics(&self, gpu_instance: &GPUInstance) {
        let mut encoder =
            gpu_instance
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("encoder"),
                });

        encoder.clear_buffer(&self.metric_buffer, 0, Some(16));

        let error_commands = encoder.finish();
        gpu_instance.queue.submit([error_commands]);
    }

    pub fn update_metrics(&self, gpu_instance: &GPUInstance) {
        let mut encoder =
            gpu_instance
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("encoder"),
                });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cpass"),
                timestamp_writes: None,
            });

            let params = ErrorPC::new(&self.data_reader, self.nn_info.n_batches);

            pass.set_pipeline(&self.test_pass_info.pipeline);

            pass.set_bind_group(0, &self.test_pass_info.bind_group, &[]);

            pass.set_push_constants(0, bytemuck::bytes_of(&params));

            let gx = ceil_div(self.nn_info.n_batches, self.test_pass_info.workgroup_dim.x);

            pass.dispatch_workgroups(gx as u32, 1, 1);
        }

        let error_commands = encoder.finish();
        gpu_instance.queue.submit([error_commands]);
    }

    pub fn read_back_params(&self, gpu_instance: &GPUInstance) {
        let mut encoder =
            gpu_instance
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("encoder"),
                });

        encoder.copy_buffer_to_buffer(
            &self.param_buffer,
            0,
            &self.out_buffer,
            0,
            self.nn_info.p_length as u64 * 4,
        );

        gpu_instance.queue.submit(Some(encoder.finish()));

        let slice = self.out_buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| ());
        gpu_instance.device.poll(wgpu::PollType::Wait).unwrap();

        // Now it's safe to read.
        let data = slice.get_mapped_range();
        let out: &[f32] = bytemuck::cast_slice(&data);

        self.nn_info.read_readback(out);

        drop(data);
        self.out_buffer.unmap();
    }

    pub fn read_back_save(&self, gpu_instance: &GPUInstance) {
        let mut encoder =
            gpu_instance
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("encoder"),
                });

        encoder.copy_buffer_to_buffer(
            &self.param_buffer,
            0,
            &self.out_buffer,
            0,
            self.nn_info.p_length as u64 * 4,
        );

        gpu_instance.queue.submit(Some(encoder.finish()));

        let slice = self.out_buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| ());
        gpu_instance.device.poll(wgpu::PollType::Wait).unwrap();

        // Now it's safe to read.
        let data = slice.get_mapped_range();
        let out: &[f32] = bytemuck::cast_slice(&data);

        let save_string = self.nn_info.get_save_string(out);

        fs::write("./saves/saved_nn.txt", save_string);

        drop(data);
        self.out_buffer.unmap();
    }

    pub fn read_back_metrics(&self, gpu_instance: &GPUInstance) {
        let mut encoder =
            gpu_instance
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("encoder"),
                });

        encoder.copy_buffer_to_buffer(&self.metric_buffer, 0, &self.out_buffer, 0, 2 as u64 * 4);

        gpu_instance.queue.submit(Some(encoder.finish()));

        let slice = self.out_buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| ());
        gpu_instance.device.poll(wgpu::PollType::Wait).unwrap();

        // Now it's safe to read.
        let data = slice.get_mapped_range();
        let out: &[u32] = bytemuck::cast_slice(&data);

        println!("{:?}", &out[..2]);

        drop(data);
        self.out_buffer.unmap();
    }

    pub fn read_back_data(&self, gpu_instance: &GPUInstance) {
        let mut encoder =
            gpu_instance
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("encoder"),
                });

        encoder.copy_buffer_to_buffer(&self.data_buffer, 0, &self.out_buffer, 0, 1024);

        gpu_instance.queue.submit(Some(encoder.finish()));

        let slice = self.out_buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| ());
        gpu_instance.device.poll(wgpu::PollType::Wait).unwrap();

        // Now it's safe to read.
        let data = slice.get_mapped_range();
        let out: &[f32] = bytemuck::cast_slice(&data);

        println!("{:?}", out);
        drop(data);
        self.out_buffer.unmap();
    }

    pub fn read_back_act_single(&self, gpu_instance: &GPUInstance) {
        let mut encoder =
            gpu_instance
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("encoder"),
                });

        encoder.copy_buffer_to_buffer(
            &self.act_buffer,
            0,
            &self.out_buffer,
            0,
            self.nn_info.activity_info.a_length as u64 * 4,
        );

        gpu_instance.queue.submit(Some(encoder.finish()));

        let slice = self.out_buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| ());
        gpu_instance.device.poll(wgpu::PollType::Wait).unwrap();

        // Now it's safe to read.
        let data = slice.get_mapped_range();
        let out: &[f32] = bytemuck::cast_slice(&data);

        println!(
            "v: {:?}",
            &out[self.nn_info.activity_info.s_start as usize
                ..self.nn_info.activity_info.d_start as usize]
        );
        println!(
            "p1: {:?}",
            &out[self.nn_info.activity_info.d_start as usize
                ..(self.nn_info.activity_info.d_start
                    + self.nn_info.activity_info.a_deriv_buffer_size) as usize]
        );
        println!(
            "p2: {:?}",
            &out[(self.nn_info.activity_info.d_start
                + self.nn_info.activity_info.a_deriv_buffer_size) as usize
                ..(self.nn_info.activity_info.d_start
                    + self.nn_info.activity_info.a_deriv_buffer_size * 2)
                    as usize]
        );

        drop(data);
        self.out_buffer.unmap();
    }

    pub fn read_back_gradients(&self, gpu_instance: &GPUInstance) {
        let mut encoder =
            gpu_instance
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("encoder"),
                });

        encoder.copy_buffer_to_buffer(
            &self.gradient_buffer,
            0,
            &self.out_buffer,
            0,
            self.nn_info.p_length as u64 * 4,
        );

        gpu_instance.queue.submit(Some(encoder.finish()));

        let slice = self.out_buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| ());
        gpu_instance.device.poll(wgpu::PollType::Wait).unwrap();

        // Now it's safe to read.
        let data = slice.get_mapped_range();
        let out: &[f32] = bytemuck::cast_slice(&data);

        self.nn_info.read_readback(out);

        drop(data);
        self.out_buffer.unmap()
    }

    pub fn read_back_raw(&self, gpu_instance: &GPUInstance, n_floats: u64) {
        let mut encoder =
            gpu_instance
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("encoder"),
                });

        encoder.copy_buffer_to_buffer(&self.act_buffer, 0, &self.out_buffer, 0, n_floats * 4);

        gpu_instance.queue.submit(Some(encoder.finish()));

        let slice = self.out_buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| ());
        gpu_instance.device.poll(wgpu::PollType::Wait).unwrap();

        // Now it's safe to read.
        let data = slice.get_mapped_range();
        let out: &[f32] = bytemuck::cast_slice(&data);

        println!("{:?}", &out[..n_floats as usize]);

        drop(data);
        self.out_buffer.unmap();
    }
}
