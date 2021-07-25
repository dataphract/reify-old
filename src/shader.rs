use std::{fmt, iter};

use crate::{util::ErrorOnDrop, vks};

#[derive(Default)]
pub struct DroppedGraphicsShaders;

impl fmt::Display for DroppedGraphicsShaders {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("GraphicsShaders must be manually destroyed with .destroy_with()")
    }
}

pub struct GraphicsShaders {
    pub bomb: ErrorOnDrop<DroppedGraphicsShaders>,

    pub vertex: vks::ShaderModule,
    pub fragment: vks::ShaderModule,
}

impl GraphicsShaders {
    pub unsafe fn destroy_with(self, device: &vks::Device) {
        let GraphicsShaders {
            mut bomb,
            vertex,
            fragment,
        } = self;

        bomb.disarm();

        for shader in iter::once(vertex).chain(iter::once(fragment)) {
            unsafe {
                device.destroy_shader_module(shader);
            }
        }
    }
}
