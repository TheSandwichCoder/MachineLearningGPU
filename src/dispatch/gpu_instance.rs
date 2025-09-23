
pub struct GPUInstance{
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub align: u64,
}

// GPU stuff

impl GPUInstance{
    pub async fn new() -> Self{
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

        return GPUInstance{
            instance,
            adapter,
            device,
            queue,
            align,
        }
    }
}
        