use erupt::vk;

use crate::vks;

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum ClearColorValue {
    Float32([f32; 4]),
    Int32([i32; 4]),
    Uint32([u32; 4]),
}

impl From<[f32; 4]> for ClearColorValue {
    fn from(f: [f32; 4]) -> Self {
        ClearColorValue::Float32(f)
    }
}

impl From<[i32; 4]> for ClearColorValue {
    fn from(i: [i32; 4]) -> Self {
        ClearColorValue::Int32(i)
    }
}

impl From<[u32; 4]> for ClearColorValue {
    fn from(u: [u32; 4]) -> Self {
        ClearColorValue::Uint32(u)
    }
}

impl From<ClearColorValue> for vk::ClearColorValue {
    fn from(val: ClearColorValue) -> Self {
        use ClearColorValue::*;

        match val {
            Float32(f) => vk::ClearColorValue { float32: f },
            Int32(i) => vk::ClearColorValue { int32: i },
            Uint32(u) => vk::ClearColorValue { uint32: u },
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ClearDepthStencilValue {
    depth: f32,
    stencil: u32,
}

impl From<ClearDepthStencilValue> for vk::ClearDepthStencilValue {
    fn from(val: ClearDepthStencilValue) -> Self {
        vk::ClearDepthStencilValue {
            depth: val.depth,
            stencil: val.stencil,
        }
    }
}

pub trait RenderPass {
    /// Returns the value used to clear color attachments.
    ///
    /// If `None`, then the initial contents of the attachment are undefined.
    fn clear_color_value(&self) -> Option<ClearColorValue> {
        None
    }

    /// Returns the value used to clear depth/stencil attachments.
    ///
    /// If `None`, then the initial contents of the attachment are undefined.
    fn clear_depth_stencil_value(&self) -> Option<ClearDepthStencilValue> {
        None
    }

    /// Records contents of the render pass to a command buffer.
    ///
    /// This excludes the actual beginning and ending of the render pass, as well as subpass transitions.
    fn record(&self, device: &vks::Device, cmdbuf: &mut vks::CommandBuffer);
}
