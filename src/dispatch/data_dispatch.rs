use bytemuck::{Pod, Zeroable};
use std::fs;
use std::num::NonZeroU64;
use wgpu::util::DeviceExt;

use crate::data_reader::*;
use crate::datatypes::*;
use crate::datatypes::{nn_datatypes::*, workgroup::*};
use crate::dispatch::conv_dispatch::ConvDispatch;
use crate::dispatch::gpu_instance::*;
use crate::dispatch::nn_dispatch::NNDispatch;
use crate::functions::*;
use crate::gpu_dirs::nn_dirs::*;

pub struct DataPassInfo {
    pub dir_buffer: wgpu::Buffer, // for metas

    pub shader: wgpu::ShaderModule,

    pub bind_group: wgpu::BindGroup,

    pub pipeline: wgpu::ComputePipeline,

    pub dir_slot_size: u64,

    pub workgroup_dim: WorkgroupDim,
}

impl DataPassInfo {
    pub fn new(
        b: wgpu::Buffer,
        s: wgpu::ShaderModule,
        bg: wgpu::BindGroup,
        p: wgpu::ComputePipeline,
        ss: u64,
        workgroup_dim: Vec<usize>,
    ) -> Self {
        return DataPassInfo {
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

pub struct DataDispatch {
    data_buffer: wgpu::Buffer,   // holds current batch of data
    metric_buffer: wgpu::Buffer, // gets accuracy and metrics
    out_buffer: wgpu::Buffer,    // retrieves parameters and debug stuff

    error_pass_info: DataPassInfo,
    test_pass_info: DataPassInfo,

    pub data_reader: DataReader,
}

impl DataDispatch {
    pub fn new_nn(
        gpu_instance: &GPUInstance,
        data_constructor: &DataConstructor,
        nn_dispatch: &NNDispatch,
    ) -> Self {
        let mut data_reader = DataReader::construct(data_constructor);

        data_reader.initialise_params_mnist_letters();
        data_reader.load_batch_mnist_letters();

        let test_metrics = TestMetrics::zero();

        // ---------------------Buffer Stuff---------------------
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

        let out_buffer = gpu_instance.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("out_buf"),
            size: (2048 * 2048 * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
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

        let error_dir = ErrorDir::new(&nn_dispatch.nn_info);

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
                        resource: nn_dispatch.get_act_buffer_ref().as_entire_binding(),
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
                        resource: nn_dispatch.get_act_buffer_ref().as_entire_binding(),
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

        let error_pass_info = DataPassInfo::new(
            error_dir_buffer.clone(),
            error_shader,
            error_bind_group,
            error_pipeline,
            error_slot,
            vec![64, 0, 0],
        );

        let test_pass_info = DataPassInfo::new(
            error_dir_buffer.clone(),
            test_shader,
            test_bind_group,
            test_pipeline,
            error_slot,
            vec![64, 0, 0],
        );

        return DataDispatch {
            data_buffer,
            metric_buffer,
            out_buffer,

            error_pass_info,
            test_pass_info,

            data_reader,
        };
    }

    pub fn new_convnn(
        gpu_instance: &GPUInstance,
        data_constructor: &DataConstructor,
        conv_dispatch: &ConvDispatch,
        nn_dispatch: &NNDispatch,
    ) -> Self {
        let mut data_reader = DataReader::construct(data_constructor);

        data_reader.initialise_params_mnist_letters();
        data_reader.load_batch_mnist_letters();

        let test_metrics = TestMetrics::zero();

        // ---------------------Buffer Stuff---------------------
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

        let out_buffer = gpu_instance.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("out_buf"),
            size: (2048 * 2048 * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
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

        let error_dir = ErrorDir::new(&nn_dispatch.nn_info);

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
                        resource: nn_dispatch.get_act_buffer_ref().as_entire_binding(),
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
                        resource: nn_dispatch.get_act_buffer_ref().as_entire_binding(),
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

        let error_pass_info = DataPassInfo::new(
            error_dir_buffer.clone(),
            error_shader,
            error_bind_group,
            error_pipeline,
            error_slot,
            vec![64, 0, 0],
        );

        let test_pass_info = DataPassInfo::new(
            error_dir_buffer.clone(),
            test_shader,
            test_bind_group,
            test_pipeline,
            error_slot,
            vec![64, 0, 0],
        );

        return DataDispatch {
            data_buffer,
            metric_buffer,
            out_buffer,

            error_pass_info,
            test_pass_info,

            data_reader,
        };
    }

    pub fn set_data_nn(&self, gpu_instance: &GPUInstance, nn_dispatch: &NNDispatch) {
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

        let nn_info = &nn_dispatch.nn_info;

        for batch_i in 0..nn_info.n_batches {
            let write_i_swap = batch_i * nn_info.activity_info.a_length;
            let write_i_storage =
                batch_i * nn_info.activity_info.a_length + nn_info.activity_info.s_start;

            let read_i = batch_i * (self.data_reader.data_value_size + 1) + curr_batch_start + 1; // plus one to skip the label

            encoder.copy_buffer_to_buffer(
                &self.data_buffer,
                (read_i * 4) as u64,
                &nn_dispatch.get_act_buffer_ref(),
                (write_i_swap * 4) as u64,
                (self.data_reader.data_value_size * 4) as u64,
            );

            encoder.copy_buffer_to_buffer(
                &self.data_buffer,
                (read_i * 4) as u64,
                &nn_dispatch.get_act_buffer_ref(),
                (write_i_storage * 4) as u64,
                (self.data_reader.data_value_size * 4) as u64,
            );
        }

        gpu_instance.queue.submit([encoder.finish()]);
    }

    pub fn set_data_convnn(&self, gpu_instance: &GPUInstance, conv_dispatch: &ConvDispatch) {
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

        let conv_info = &conv_dispatch.conv_info;

        for batch_i in 0..self.data_reader.n_batches {
            let write_i_swap = batch_i * conv_info.activity_info.batch_swap_buffer_size;
            let write_i_storage = batch_i * conv_info.activity_info.dim[0].tens_length;

            let read_i = batch_i * (self.data_reader.data_value_size + 1) + curr_batch_start + 1; // plus one to skip the label

            encoder.copy_buffer_to_buffer(
                &self.data_buffer,
                (read_i * 4) as u64,
                &conv_dispatch.get_act_buffer_ref(),
                (write_i_swap * 4) as u64,
                (self.data_reader.data_value_size * 4) as u64,
            );

            encoder.copy_buffer_to_buffer(
                &self.data_buffer,
                (read_i * 4) as u64,
                &conv_dispatch.get_storage_buffer_ref(),
                (write_i_storage * 4) as u64,
                (self.data_reader.data_value_size * 4) as u64,
            );
        }

        gpu_instance.queue.submit([encoder.finish()]);
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

            let params = ErrorPC::new(&self.data_reader, self.data_reader.n_batches);

            // println!("{}", params.batch_start_idx);

            pass.set_pipeline(&self.error_pass_info.pipeline);

            pass.set_bind_group(0, &self.error_pass_info.bind_group, &[]);

            pass.set_push_constants(0, bytemuck::bytes_of(&params));

            let gx = ceil_div(
                self.data_reader.n_batches,
                self.error_pass_info.workgroup_dim.x,
            );

            pass.dispatch_workgroups(gx as u32, 1, 1);
        }

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

            let params = ErrorPC::new(&self.data_reader, self.data_reader.n_batches);

            pass.set_pipeline(&self.test_pass_info.pipeline);

            pass.set_bind_group(0, &self.test_pass_info.bind_group, &[]);

            pass.set_push_constants(0, bytemuck::bytes_of(&params));

            let gx = ceil_div(
                self.data_reader.n_batches,
                self.test_pass_info.workgroup_dim.x,
            );

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
}
