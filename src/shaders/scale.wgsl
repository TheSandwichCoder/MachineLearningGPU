struct Buf { data: array<f32>, };

struct Meta {
  len: u32,
  _pad: vec3<u32>,    // keep 16-byte alignment for uniform buffers
  scale: f32,
};

@group(0) @binding(0) var<storage, read>       in_buf: Buf;
@group(0) @binding(1) var<storage, read_write> out_buf: Buf;
@group(0) @binding(2) var<uniform>             meta_thing: Meta;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i < meta_thing.len) {
    out_buf.data[i] = in_buf.data[i] * meta_thing.scale;
  }
}
