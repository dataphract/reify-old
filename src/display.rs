use erupt::vk;

use crate::{
    vks::{self, VkObject},
    Device, Surface,
};

pub struct Display {
    extent: vk::Extent2D,
    swapchain: vks::Swapchain,
}

impl Display {
    pub fn create(device: &Device, surface: &Surface) {
        let device_read = device.inner.read();
        let surface_read = surface.inner.read();
        let instance_read = device_read.instance.read_inner();

        let surf_caps = unsafe {
            instance_read
                .handle
                .get_physical_device_surface_capabilities_khr(
                    device_read.phys_device.inner.raw.handle(),
                    surface_read.raw.as_ref().expect("Surface missing").handle(),
                )
        }
        .expect("failed to query surface capabilities");

        let extent = surf_caps.current_extent;

        let create_info = vks::SwapchainCreateInfo {
            flags: todo!(),
            surface: todo!(),
            min_image_count: todo!(),
            image_format: todo!(),
            image_color_space: todo!(),
            image_extent: extent,
            image_array_layers: todo!(),
            image_usage: todo!(),
            image_sharing_mode: vk::SharingMode::EXCLUSIVE,
            queue_family_indices: &[],
            pre_transform: vk::SurfaceTransformFlagBitsKHR::IDENTITY_KHR,
            composite_alpha: todo!(),
            present_mode: todo!(),
            clipped: true,
            old_swapchain: None,
        };
    }
}
