use crate::datatypes::{conv_datatypes::*, workgroup::*};
use crate::functions::*;
use crate::gpu_dirs::conv_dirs::*;
use bytemuck::{Pod, Zeroable};
use std::num::NonZeroU64;
use wgpu::util::DeviceExt;

use std::fs;

use crate::dispatch::gpu_instance::GPUInstance;

pub struct ConvPassInfo {
    pub dir_buffer: wgpu::Buffer, // for metas

    pub shader: wgpu::ShaderModule,

    pub bind_group: wgpu::BindGroup,

    pub pipeline: wgpu::ComputePipeline,

    pub dir_slot_size: u64,

    pub workgroup_dim: WorkgroupDim,
}

impl ConvPassInfo {
    pub fn new(
        b: wgpu::Buffer,
        s: wgpu::ShaderModule,
        bg: wgpu::BindGroup,
        p: wgpu::ComputePipeline,
        ss: u64,
        workgroup_dim: Vec<usize>,
    ) -> Self {
        return ConvPassInfo {
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

pub struct ConvDispatch {
    param_buffer: wgpu::Buffer,            // holds the params for the model
    gradient_buffer: wgpu::Buffer,         // holds the param gradients
    momentum_buffer: wgpu::Buffer,         // holds the param momentum
    o_storage_buffer: wgpu::Buffer,        // holds intermediate layer outputs and gradients
    pool_idx_storage_buffer: wgpu::Buffer, // holds the idx of val that was pooled
    conv_output_swap_buffer: wgpu::Buffer,
    conv_deriv_swap_buffer: wgpu::Buffer,
    accumulate_buffer: wgpu::Buffer,
    out_buffer: wgpu::Buffer, // retrieves parameters and debug stuff

    forward_mat_pass_info: ConvPassInfo,
    forward_pool_pass_info: ConvPassInfo,

    backward_gradient_pass_info: ConvPassInfo,
    backward_deriv_pass_info: ConvPassInfo,
    backward_pool_pass_info: ConvPassInfo,

    gradient_acc_pass_info: ConvPassInfo,

    pub conv_info: ConvolutionInfo,
}

impl ConvDispatch {
    pub fn new(gpu_instance: &GPUInstance, conv_constructor: ConvolutionConstructor) -> Self {
        // -------------------Convolution Info-------------------
        let conv_info = ConvolutionInfo::construct(conv_constructor);

        // ---------------------Buffer Stuff---------------------
        let param_buffer =
            gpu_instance
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("cparam buffer"),
                    contents: bytemuck::cast_slice(&conv_info.param_info.create_buffer()),
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                });

        let gradient_buffer =
            gpu_instance
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("cgradient buffer"),
                    contents: bytemuck::cast_slice(
                        &conv_info
                            .param_info
                            .create_buffer_empty(conv_info.n_batches),
                    ),
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                });

        let conv_output_swap_buffer =
            gpu_instance
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("swap output buffer"),
                    contents: bytemuck::cast_slice(
                        &conv_info.activity_info.create_output_swap_buffer(),
                    ),
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                });

        let conv_deriv_swap_buffer =
            gpu_instance
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("swap output buffer"),
                    contents: bytemuck::cast_slice(
                        &conv_info.activity_info.create_deriv_swap_buffer(),
                    ),
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                });

        let accumulate_buffer =
            gpu_instance
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("accumulate buffer"),
                    // do this
                    contents: bytemuck::cast_slice(
                        &conv_info
                            .param_info
                            .create_accumulate_buffer_empty(conv_info.n_batches),
                    ),
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                });

        let momentum_buffer =
            gpu_instance
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("cmomentum buffer"),
                    contents: bytemuck::cast_slice(&conv_info.param_info.create_buffer_empty(1)),
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                });

        let o_storage_buffer =
            gpu_instance
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("o_storage_buffer buffer"),
                    contents: bytemuck::cast_slice(
                        &conv_info.activity_info.create_output_storage_buffer(),
                    ),
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                });

        let pool_idx_storage_buffer =
            gpu_instance
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("pool_idx_storage_buffer buffer"),
                    contents: bytemuck::cast_slice(
                        &conv_info.activity_info.create_pool_idx_storage_buffer(),
                    ),
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                });

        let out_buffer = gpu_instance.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("out_buf"),
            // size: (&conv_info.activity_info.storage_buffer_size * std::mem::size_of::<f32>()) as u64,
            size: (conv_info.activity_info.deriv_buffer_size * 2 * std::mem::size_of::<f32>())
                as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ---------------------Bind Layout---------------------
        let forward_mat_bind_layout =
            gpu_instance
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("forward_mat_bind_layout"),
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
                                min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<
                                    Im2ColDir_F,
                                >(
                                )
                                    as u64),
                            },
                            count: None,
                        },
                    ],
                });

        let imcol_mat_slot = ((std::mem::size_of::<Im2ColDir_F>() as u64 + gpu_instance.align - 1)
            / gpu_instance.align)
            * gpu_instance.align;

        // +--------------------------------------------------------+
        // |                                                        |
        // |                   Forward Mat Pass                     |
        // |                                                        |
        // +--------------------------------------------------------+

        let forward_mat_dir_buffer_size = imcol_mat_slot * (conv_info.n_layers - 1) as u64;

        let forward_mat_dir_buffer = gpu_instance.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cforward_mat_dir_buf"),
            size: forward_mat_dir_buffer_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ---------------------Special Binding---------------------
        let forward_mat_dir_binding = wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &forward_mat_dir_buffer,
            offset: 0,
            size: Some(wgpu::BufferSize::new(imcol_mat_slot).unwrap()),
        });

        // ---------------------Shader Sources---------------------
        let wgsl_src = include_str!("../shaders/gemm_op/gemm_imcol.wgsl");
        let forward_mat_shader =
            gpu_instance
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("cforward_gemm_shader"),
                    source: wgpu::ShaderSource::Wgsl(wgsl_src.into()),
                });

        let forward_mat_bind_group =
            gpu_instance
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("forward mat bg"),
                    layout: &forward_mat_bind_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: param_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: conv_output_swap_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: forward_mat_dir_binding,
                        },
                    ],
                });

        // ---------------------Update Meta---------------------

        for layer_i in 0..conv_info.n_layers - 1 {
            let mat_dir = Im2ColDir_F::new(&conv_info, layer_i);

            // println!("{}", mat_dir.);

            gpu_instance.queue.write_buffer(
                &forward_mat_dir_buffer,
                layer_i as u64 * imcol_mat_slot,
                bytemuck::bytes_of(&mat_dir),
            );
        }

        // ---------------------Pipeline Layout---------------------

        let forward_mat_pipeline_layout =
            gpu_instance
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("cforward_gemm_pipeline_layout"),
                    bind_group_layouts: &[&forward_mat_bind_layout],
                    push_constant_ranges: &[],
                });

        let forward_mat_pipeline =
            gpu_instance
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("cforward_gemm_pipeline"),
                    layout: Some(&forward_mat_pipeline_layout),
                    module: &forward_mat_shader,
                    entry_point: Some("main"),
                    cache: None,
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                });

        // +--------------------------------------------------------+
        // |                                                        |
        // |                  Forward Pool Pass                     |
        // |                                                        |
        // +--------------------------------------------------------+

        let pool_bind_layout =
            gpu_instance
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("c_gemm_bind_layout"),
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
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
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
                                min_binding_size: wgpu::BufferSize::new(
                                    std::mem::size_of::<PoolDir>() as u64,
                                ),
                            },
                            count: None,
                        },
                    ],
                });

        let pool_slot = ((std::mem::size_of::<PoolDir>() as u64 + gpu_instance.align - 1)
            / gpu_instance.align)
            * gpu_instance.align;

        let forward_pool_dir_buffer_size = pool_slot * (conv_info.n_layers - 1) as u64;

        let forward_pool_dir_buffer = gpu_instance.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cforward_pool_dir_buf"),
            size: forward_pool_dir_buffer_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ---------------------Special Binding---------------------
        let forward_pool_dir_binding = wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &forward_pool_dir_buffer,
            offset: 0,
            size: Some(wgpu::BufferSize::new(pool_slot).unwrap()),
        });

        // ---------------------Shader Sources---------------------
        let wgsl_src = include_str!("../shaders/conv_op/pool.wgsl");
        let forward_pool_shader =
            gpu_instance
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("cforward_pool_shader"),
                    source: wgpu::ShaderSource::Wgsl(wgsl_src.into()),
                });

        let forward_pool_bind_group =
            gpu_instance
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("pool bg"),
                    layout: &pool_bind_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: conv_output_swap_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: o_storage_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: pool_idx_storage_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: forward_pool_dir_binding,
                        },
                    ],
                });

        // ---------------------Update Meta---------------------

        for layer_i in 0..conv_info.n_layers - 1 {
            let pool_dir = PoolDir::forward_new(&conv_info, layer_i);

            gpu_instance.queue.write_buffer(
                &forward_pool_dir_buffer,
                layer_i as u64 * pool_slot,
                bytemuck::bytes_of(&pool_dir),
            );
        }

        // ---------------------Pipeline Layout---------------------

        let forward_pool_pipeline_layout =
            gpu_instance
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("cforward_pool_pipeline_layout"),
                    bind_group_layouts: &[&pool_bind_layout],
                    push_constant_ranges: &[],
                });

        let forward_pool_pipeline =
            gpu_instance
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("cforward_pool_pipeline"),
                    layout: Some(&forward_pool_pipeline_layout),
                    module: &forward_pool_shader,
                    entry_point: Some("main"),
                    cache: None,
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                });

        // +--------------------------------------------------------+
        // |                                                        |
        // |               Gradient Accumulate Pass                 |
        // |                                                        |
        // +--------------------------------------------------------+

        let gradient_acc_bind_layout =
            gpu_instance
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("gradient_acc_bind_layout"),
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
                                    std::mem::size_of::<AccDir>() as u64,
                                ),
                            },
                            count: None,
                        },
                    ],
                });

        let gradient_acc_slot = ((std::mem::size_of::<AccDir>() as u64 + gpu_instance.align - 1)
            / gpu_instance.align)
            * gpu_instance.align;

        let gradient_acc_dir_buffer_size = gradient_acc_slot * (conv_info.n_layers - 1) as u64;

        let gradient_acc_dir_buffer = gpu_instance.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cgradient_acc_dir_buf"),
            size: gradient_acc_dir_buffer_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ---------------------Special Binding---------------------
        let gradient_acc_dir_binding = wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &gradient_acc_dir_buffer,
            offset: 0,
            size: Some(wgpu::BufferSize::new(gradient_acc_slot).unwrap()),
        });

        // ---------------------Shader Sources---------------------
        let wgsl_src = include_str!("../shaders/conv_op/acc_gradients.wgsl");
        let gradient_acc_shader =
            gpu_instance
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("acc_gradient_shader"),
                    source: wgpu::ShaderSource::Wgsl(wgsl_src.into()),
                });

        let gradient_acc_bind_group =
            gpu_instance
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("gradient acc bg"),
                    layout: &gradient_acc_bind_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: accumulate_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: gradient_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: gradient_acc_dir_binding,
                        },
                    ],
                });

        // ---------------------Update Meta---------------------

        for layer_i in 0..conv_info.n_layers - 1 {
            let gradient_acc_dir = AccDir::new(&conv_info, layer_i);

            gpu_instance.queue.write_buffer(
                &gradient_acc_dir_buffer,
                layer_i as u64 * gradient_acc_slot,
                bytemuck::bytes_of(&gradient_acc_dir),
            );
        }

        // ---------------------Pipeline Layout---------------------

        let gradient_acc_pipeline_layout =
            gpu_instance
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("cgradient_acc_pipeline_layout"),
                    bind_group_layouts: &[&gradient_acc_bind_layout],
                    push_constant_ranges: &[],
                });

        let gradient_acc_pipeline =
            gpu_instance
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("gradient_acc_pipeline"),
                    layout: Some(&gradient_acc_pipeline_layout),
                    module: &gradient_acc_shader,
                    entry_point: Some("main"),
                    cache: None,
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                });

        // +--------------------------------------------------------+
        // |                                                        |
        // |                Backward Gradient Pass                  |
        // |                                                        |
        // +--------------------------------------------------------+

        let backward_gradient_bind_layout =
            gpu_instance
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("c_b_g_bind_layout"),
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
                                    Im2ColDir_BG,
                                >(
                                )
                                    as u64),
                            },
                            count: None,
                        },
                    ],
                });

        let imcol_bg_slot = ((std::mem::size_of::<Im2ColDir_BG>() as u64 + gpu_instance.align - 1)
            / gpu_instance.align)
            * gpu_instance.align;

        let backward_gradient_dir_buffer_size = imcol_bg_slot * (conv_info.n_layers - 1) as u64;

        let backward_gradient_dir_buffer =
            gpu_instance.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("cbackwards_gradient_dir_buf"),
                size: backward_gradient_dir_buffer_size,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

        // ---------------------Special Binding---------------------
        let backward_gradient_dir_binding = wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &backward_gradient_dir_buffer,
            offset: 0,
            size: Some(wgpu::BufferSize::new(imcol_bg_slot).unwrap()),
        });

        // ---------------------Shader Sources---------------------
        let wgsl_src = include_str!("../shaders/gemm_op/gemm_dldw.wgsl");
        let backward_gradient_shader =
            gpu_instance
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("cbackward_gradient_shader"),
                    source: wgpu::ShaderSource::Wgsl(wgsl_src.into()),
                });

        let backward_gradient_bind_group =
            gpu_instance
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("backward gradient bg"),
                    layout: &backward_gradient_bind_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: o_storage_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: conv_deriv_swap_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: accumulate_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: backward_gradient_dir_binding,
                        },
                    ],
                });

        // ---------------------Update Meta---------------------

        for layer_i in 0..conv_info.n_layers - 1 {
            let backward_gradient_dir = Im2ColDir_BG::new(&conv_info, layer_i);

            gpu_instance.queue.write_buffer(
                &backward_gradient_dir_buffer,
                layer_i as u64 * imcol_bg_slot,
                bytemuck::bytes_of(&backward_gradient_dir),
            );
        }

        // ---------------------Pipeline Layout---------------------

        let backward_gradient_pipeline_layout =
            gpu_instance
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("cbackward_gradient_pipeline_layout"),
                    bind_group_layouts: &[&backward_gradient_bind_layout],
                    push_constant_ranges: &[],
                });

        let backward_gradient_pipeline =
            gpu_instance
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("cbackward_gradient_pipeline"),
                    layout: Some(&backward_gradient_pipeline_layout),
                    module: &backward_gradient_shader,
                    entry_point: Some("main"),
                    cache: None,
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                });

        // +--------------------------------------------------------+
        // |                                                        |
        // |              Backward Derivative Pass                  |
        // |                                                        |
        // +--------------------------------------------------------+

        let backward_deriv_bind_layout =
            gpu_instance
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("c_b_d_bind_layout"),
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
                                min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<
                                    Im2ColDir_BD,
                                >(
                                )
                                    as u64),
                            },
                            count: None,
                        },
                    ],
                });

        let imcol_bd_slot = ((std::mem::size_of::<Im2ColDir_BD>() as u64 + gpu_instance.align - 1)
            / gpu_instance.align)
            * gpu_instance.align;

        let backward_deriv_dir_buffer_size = imcol_bd_slot * (conv_info.n_layers - 1) as u64;

        let backward_deriv_dir_buffer =
            gpu_instance.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("cbackwards_deriv_dir_buf"),
                size: backward_deriv_dir_buffer_size,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

        // ---------------------Special Binding---------------------
        let backward_deriv_dir_binding = wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &backward_deriv_dir_buffer,
            offset: 0,
            size: Some(wgpu::BufferSize::new(imcol_bd_slot).unwrap()),
        });

        // ---------------------Shader Sources---------------------
        let wgsl_src = include_str!("../shaders/gemm_op/gemm_dldi.wgsl");
        let backward_deriv_shader =
            gpu_instance
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("cbackward_derivative_shader"),
                    source: wgpu::ShaderSource::Wgsl(wgsl_src.into()),
                });

        let backward_deriv_bind_group =
            gpu_instance
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("backward deriv bg"),
                    layout: &backward_deriv_bind_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: param_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: conv_deriv_swap_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: backward_deriv_dir_binding,
                        },
                    ],
                });

        // ---------------------Update Meta---------------------

        for layer_i in 0..conv_info.n_layers - 1 {
            let backward_deriv_dir = Im2ColDir_BD::new(&conv_info, layer_i);

            gpu_instance.queue.write_buffer(
                &backward_deriv_dir_buffer,
                layer_i as u64 * imcol_bg_slot,
                bytemuck::bytes_of(&backward_deriv_dir),
            );
        }

        // ---------------------Pipeline Layout---------------------

        let backward_deriv_pipeline_layout =
            gpu_instance
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("backward_deriv_pipeline_layout"),
                    bind_group_layouts: &[&backward_deriv_bind_layout],
                    push_constant_ranges: &[],
                });

        let backward_deriv_pipeline =
            gpu_instance
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("cbackward_deriv_pipeline"),
                    layout: Some(&backward_deriv_pipeline_layout),
                    module: &backward_deriv_shader,
                    entry_point: Some("main"),
                    cache: None,
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                });

        // +--------------------------------------------------------+
        // |                                                        |
        // |                 Backward Pooling Pass                  |
        // |                                                        |
        // +--------------------------------------------------------+

        let backward_pool_bind_layout =
            gpu_instance
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("c_b_p_bind_layout"),
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
                                    std::mem::size_of::<PoolDir>() as u64,
                                ),
                            },
                            count: None,
                        },
                    ],
                });

        let backward_pool_slot = ((std::mem::size_of::<PoolDir>() as u64 + gpu_instance.align - 1)
            / gpu_instance.align)
            * gpu_instance.align;

        let backward_pool_buffer_size = backward_pool_slot * (conv_info.n_layers - 1) as u64;

        let backward_pool_dir_buffer = gpu_instance.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cbackwards_pool_dir_buf"),
            size: backward_pool_buffer_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ---------------------Special Binding---------------------
        let backward_pool_dir_binding = wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &backward_pool_dir_buffer,
            offset: 0,
            size: Some(wgpu::BufferSize::new(backward_pool_slot).unwrap()),
        });

        // ---------------------Shader Sources---------------------
        let wgsl_src = include_str!("../shaders/conv_op/unpool.wgsl");
        let backward_pool_shader =
            gpu_instance
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("cbackward_pool_shader"),
                    source: wgpu::ShaderSource::Wgsl(wgsl_src.into()),
                });

        let backward_pool_bind_group =
            gpu_instance
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("backward pool bg"),
                    layout: &backward_pool_bind_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: pool_idx_storage_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: conv_deriv_swap_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: backward_pool_dir_binding,
                        },
                    ],
                });

        // ---------------------Update Meta---------------------

        for layer_i in 0..conv_info.n_layers - 1 {
            let pool_dir = PoolDir::backward_new(&conv_info, layer_i);

            gpu_instance.queue.write_buffer(
                &backward_pool_dir_buffer,
                layer_i as u64 * backward_pool_slot,
                bytemuck::bytes_of(&pool_dir),
            );
        }

        // ---------------------Pipeline Layout---------------------

        let backward_pool_pipeline_layout =
            gpu_instance
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("backward_pool_pipeline_layout"),
                    bind_group_layouts: &[&backward_pool_bind_layout],
                    push_constant_ranges: &[],
                });

        let backward_pool_pipeline =
            gpu_instance
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("cbackward_pool_pipeline"),
                    layout: Some(&backward_pool_pipeline_layout),
                    module: &backward_pool_shader,
                    entry_point: Some("main"),
                    cache: None,
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                });

        let gem_wg = vec![16, 16, 0];

        let forward_mat_pass_info = ConvPassInfo::new(
            forward_mat_dir_buffer,
            forward_mat_shader,
            forward_mat_bind_group,
            forward_mat_pipeline,
            imcol_mat_slot,
            gem_wg.clone(),
        );
        let forward_pool_pass_info = ConvPassInfo::new(
            forward_pool_dir_buffer,
            forward_pool_shader,
            forward_pool_bind_group,
            forward_pool_pipeline,
            pool_slot,
            vec![8, 8, 1],
        );

        let backward_gradient_pass_info = ConvPassInfo::new(
            backward_gradient_dir_buffer,
            backward_gradient_shader,
            backward_gradient_bind_group,
            backward_gradient_pipeline,
            imcol_bg_slot,
            gem_wg.clone(),
        );
        let backward_deriv_pass_info = ConvPassInfo::new(
            backward_deriv_dir_buffer,
            backward_deriv_shader,
            backward_deriv_bind_group,
            backward_deriv_pipeline,
            imcol_bd_slot,
            gem_wg.clone(),
        );
        let backward_pool_pass_info = ConvPassInfo::new(
            backward_pool_dir_buffer,
            backward_pool_shader,
            backward_pool_bind_group,
            backward_pool_pipeline,
            backward_pool_slot,
            vec![8, 8, 1],
        );

        let gradient_acc_pass_info = ConvPassInfo::new(
            gradient_acc_dir_buffer,
            gradient_acc_shader,
            gradient_acc_bind_group,
            gradient_acc_pipeline,
            gradient_acc_slot,
            vec![32, 0, 0],
        );

        return ConvDispatch {
            param_buffer,
            gradient_buffer,
            momentum_buffer,
            o_storage_buffer,
            pool_idx_storage_buffer,
            conv_output_swap_buffer,
            conv_deriv_swap_buffer,
            accumulate_buffer,
            out_buffer,

            forward_mat_pass_info,
            forward_pool_pass_info,

            backward_gradient_pass_info,
            backward_deriv_pass_info,
            backward_pool_pass_info,

            gradient_acc_pass_info,

            conv_info,
        };
    }

    pub fn accumulate_gradients(&self, gpu_instance: &GPUInstance) {
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

            let layer_i = 0;

            let dyn_off = layer_i as u32 * self.gradient_acc_pass_info.dir_slot_size as u32;
            pass.set_pipeline(&self.gradient_acc_pass_info.pipeline);
            pass.set_bind_group(0, &self.gradient_acc_pass_info.bind_group, &[dyn_off]);

            let gx = ceil_div(
                self.conv_info.conv_layers[layer_i].n_kernals
                    * self.conv_info.conv_layers[layer_i].kernal_info.size,
                self.gradient_acc_pass_info.workgroup_dim.x,
            );

            pass.dispatch_workgroups(gx as u32, 1, 1);
        }

        let backward_commands = encoder.finish();
        gpu_instance.queue.submit([backward_commands]);
    }

    pub fn backward_conv_mat(&self, gpu_instance: &GPUInstance) {
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

            let layer_i = self.conv_info.n_layers - 3;

            let dyn_off = layer_i as u32 * self.backward_gradient_pass_info.dir_slot_size as u32;
            pass.set_pipeline(&self.backward_gradient_pass_info.pipeline);
            pass.set_bind_group(0, &self.backward_gradient_pass_info.bind_group, &[dyn_off]);

            let conv_layer = &self.conv_info.conv_layers[layer_i];

            let gx = ceil_div(
                conv_layer.n_kernals,
                self.backward_gradient_pass_info.workgroup_dim.x,
            );
            let gy = ceil_div(
                conv_layer.kernal_info.size,
                self.backward_gradient_pass_info.workgroup_dim.y,
            );

            println!("{} {}", gy, conv_layer.acc_length);

            pass.dispatch_workgroups(gx as u32, gy as u32, conv_layer.acc_length as u32);
        }

        let backward_commands = encoder.finish();
        gpu_instance.queue.submit([backward_commands]);
    }

    pub fn backward_conv_mat_deriv(&self, gpu_instance: &GPUInstance) {
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

            let layer_i = self.conv_info.n_layers - 3;

            let dyn_off = layer_i as u32 * self.backward_deriv_pass_info.dir_slot_size as u32;
            pass.set_pipeline(&self.backward_deriv_pass_info.pipeline);
            pass.set_bind_group(0, &self.backward_deriv_pass_info.bind_group, &[dyn_off]);

            let conv_layer = &self.conv_info.conv_layers[layer_i];

            let gx = ceil_div(
                conv_layer.layer_dim[2],
                self.backward_deriv_pass_info.workgroup_dim.x,
            );
            let gy = ceil_div(
                conv_layer.layer_size_2d * self.conv_info.n_batches,
                self.backward_deriv_pass_info.workgroup_dim.y,
            );

            println!("{} {}", gy, conv_layer.acc_length);

            pass.dispatch_workgroups(gx as u32, gy as u32, 1 as u32);
        }

        let backward_commands = encoder.finish();
        gpu_instance.queue.submit([backward_commands]);
    }

    pub fn forward_conv_mat(&self, gpu_instance: &GPUInstance) {
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

            let layer_i = 0;
            for layer_i in 0..(self.conv_info.n_layers - 1) {
                {
                    let dyn_off = layer_i as u32 * self.forward_mat_pass_info.dir_slot_size as u32;
                    pass.set_pipeline(&self.forward_mat_pass_info.pipeline);
                    pass.set_bind_group(0, &self.forward_mat_pass_info.bind_group, &[dyn_off]);

                    let gx = ceil_div(
                        self.conv_info.conv_layers[layer_i].n_kernals,
                        self.forward_mat_pass_info.workgroup_dim.x,
                    );
                    let gy = ceil_div(
                        self.conv_info.conv_layers[layer_i].layer_size_2d
                            * self.conv_info.n_batches,
                        self.forward_mat_pass_info.workgroup_dim.y,
                    );

                    pass.dispatch_workgroups(gx as u32, gy as u32, 1);
                }

                {
                    let dyn_off = layer_i as u32 * self.forward_pool_pass_info.dir_slot_size as u32;
                    pass.set_pipeline(&self.forward_pool_pass_info.pipeline);

                    pass.set_bind_group(0, &self.forward_pool_pass_info.bind_group, &[dyn_off]);

                    let gx = ceil_div(
                        self.conv_info.conv_layers[layer_i].layer_dim[0],
                        self.forward_pool_pass_info.workgroup_dim.x,
                    );
                    let gy = ceil_div(
                        self.conv_info.conv_layers[layer_i].layer_dim[1],
                        self.forward_pool_pass_info.workgroup_dim.y,
                    );

                    pass.dispatch_workgroups(
                        gx as u32,
                        gy as u32,
                        (self.conv_info.conv_layers[layer_i].n_kernals * self.conv_info.n_batches)
                            as u32,
                    );
                }
            }
        }

        let forward_commands = encoder.finish();
        gpu_instance.queue.submit([forward_commands]);
    }

    pub fn set_data(&self, gpu_instance: &GPUInstance) {}

    pub fn read_back_act_single(&self, gpu_instance: &GPUInstance) {
        let mut encoder =
            gpu_instance
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("encoder"),
                });

        // encoder.copy_buffer_to_buffer(&self.accumulate_buffer, 0, &self.out_buffer, 0, 30000 as u64 * 4);
        encoder.copy_buffer_to_buffer(
            &self.conv_deriv_swap_buffer,
            0,
            &self.out_buffer,
            0,
            self.conv_info.activity_info.deriv_buffer_size as u64 * 4 * 2,
        );

        // encoder.copy_buffer_to_buffer(&self.conv_output_swap_buffer, 0, &self.out_buffer, 0, self.conv_info.activity_info.swap_buffer_size as u64 * 4 * 2);

        gpu_instance.queue.submit(Some(encoder.finish()));

        let slice = self.out_buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| ());
        gpu_instance.device.poll(wgpu::PollType::Wait).unwrap();

        // Now it's safe to read.
        let data = slice.get_mapped_range();
        let out: &[f32] = bytemuck::cast_slice(&data);

        // let start_idx = 0;
        let start_idx = self.conv_info.activity_info.deriv_buffer_size
            + self.conv_info.activity_info.batch_swap_buffer_size;

        // let start_idx = self.conv_info.activity_info.strides[1];

        let layer_size = 28;
        let check_n_layers = 30;

        let mut prev_idx = start_idx;
        let mut curr_idx = prev_idx + layer_size;

        while curr_idx <= start_idx + layer_size * check_n_layers {
            if ((prev_idx - start_idx) / layer_size) % layer_size == 0 {
                println!("{}", (prev_idx - start_idx) / (layer_size * layer_size));
            }
            println!(
                "{} activity buffer: {:?}",
                (((prev_idx - start_idx) / layer_size) % layer_size),
                &out[prev_idx..curr_idx]
            );
            prev_idx += layer_size;
            curr_idx += layer_size;
        }

        drop(data);
        self.out_buffer.unmap();
    }
}
