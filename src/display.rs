use std::cmp;

use erupt::vk;

use crate::{
    vks::{self, VkObject},
    Device,
};

const SWAPCHAIN_CHOOSES_EXTENT: vk::Extent2D = vk::Extent2D {
    width: u32::MAX,
    height: u32::MAX,
};

pub struct DisplayInfo {
    pub min_image_count: u32,
    pub surface_format: vk::SurfaceFormatKHR,
    pub image_extent: vk::Extent2D,
    pub present_mode: vk::PresentModeKHR,
}

pub struct Display {
    info: DisplayInfo,

    image_views: Vec<vks::ImageView>,
    images: Vec<vks::Image>,
    swapchain: Option<vks::SwapchainKHR>,
    surface: Option<vks::SurfaceKHR>,
    device: Device,
}

impl Drop for Display {
    fn drop(&mut self) {
        let device_read = self.device.inner.read();

        for image_view in self.image_views.drain(..) {
            unsafe {
                // TODO: must allow commands to complete on image views before
                // destroying
                device_read.raw.destroy_image_view(image_view);
            }
        }

        // Swapchain images do not need to be explicitly destroyed, as they are
        // managed by the swapchain.

        if let Some(swapchain) = self.swapchain.take() {
            unsafe {
                device_read.raw.destroy_swapchain(swapchain);
            }
        }

        let instance_read = device_read.instance.read_inner();

        if let Some(surface) = self.surface.take() {
            unsafe {
                instance_read.handle.destroy_surface(surface);
            }
        }
    }
}

impl Display {
    // Safety: device and surface must be from same instance
    pub unsafe fn create(
        device: &Device,
        mut surface: vks::SurfaceKHR,
        phys_window_extent: vk::Extent2D,
    ) -> Display {
        let device_read = device.inner.read();
        let instance_read = device_read.instance.read_inner();

        let (surf_caps, surf_formats, surf_present_modes) = unsafe {
            let instance_handle = &instance_read.handle;
            let phys = &device_read.phys_device.inner.raw;

            let surface_supported = instance_handle
                .get_physical_device_surface_support_khr(
                    &phys,
                    device.present_family_id(),
                    &surface,
                )
                .expect("failed to verify physical device surface support");

            if !surface_supported {
                panic!("Surface not supported with this device.");
            }

            (
                instance_handle
                    .get_physical_device_surface_capabilities_khr(phys, &surface)
                    .expect("failed to query surface capabilities"),
                instance_handle
                    .get_physical_device_surface_formats_khr(phys, &surface)
                    .expect("failed to query surface formats"),
                instance_handle
                    .get_physical_device_surface_present_modes_khr(phys, &surface)
                    .expect("failed to query surface presentation modes"),
            )
        };

        let min_image_count = {
            // Try to keep an image free from the driver at all times.
            let desired = surf_caps.min_image_count + 1;

            if surf_caps.max_image_count == 0 {
                // No limit.
                desired
            } else {
                cmp::max(desired, surf_caps.max_image_count)
            }
        };

        // This is highly unlikely, but the spec doesn't require that
        // implementations support the identity transform.
        assert!(
            surf_caps
                .supported_transforms
                .contains(vk::SurfaceTransformFlagsKHR::IDENTITY_KHR),
            "surface must support IDENTITY_KHR transform",
        );

        let surface_format = *surf_formats
            .iter()
            .find(|sf| {
                sf.format == vk::Format::B8G8R8A8_SRGB
                    && sf.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR_KHR
            })
            .unwrap_or(&surf_formats[0]);

        let image_extent = if surf_caps.current_extent.width == SWAPCHAIN_CHOOSES_EXTENT.width
            && surf_caps.current_extent.height == SWAPCHAIN_CHOOSES_EXTENT.height
        {
            phys_window_extent
        } else {
            surf_caps.current_extent
        };

        let present_mode = if surf_present_modes
            .iter()
            .any(|&pm| pm == vk::PresentModeKHR::MAILBOX_KHR)
        {
            vk::PresentModeKHR::MAILBOX_KHR
        } else {
            // Implementations are required to support FIFO.
            vk::PresentModeKHR::FIFO_KHR
        };

        let mut create_info = vks::SwapchainCreateInfo {
            flags: vk::SwapchainCreateFlagsKHR::empty(),
            surface: &mut surface,
            min_image_count,
            image_format: surface_format.format,
            image_color_space: surface_format.color_space,
            image_extent,
            image_array_layers: 1,
            image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
            image_sharing_mode: vk::SharingMode::EXCLUSIVE,
            queue_family_indices: &[],
            pre_transform: vk::SurfaceTransformFlagBitsKHR::IDENTITY_KHR,
            composite_alpha: vk::CompositeAlphaFlagBitsKHR::OPAQUE_KHR,
            present_mode,
            clipped: true,
            old_swapchain: None,
        };

        let swapchain = unsafe { device_read.raw.create_swapchain_khr(&mut create_info) }
            .expect("failed to create swapchain");

        log::info!("Successfully created swapchain.");

        let images = unsafe { device_read.raw.get_swapchain_images_khr(&swapchain) }
            .expect("failed to get swapchain images");

        let image_views = images
            .iter()
            .map(|img| unsafe {
                device_read.raw.create_image_view(
                    &vks::ImageViewCreateInfoBuilder::new()
                        .flags(vk::ImageViewCreateFlags::empty())
                        .image(img)
                        .view_type(vk::ImageViewType::_2D)
                        .format(surface_format.format)
                        .components(vk::ComponentMapping {
                            r: vk::ComponentSwizzle::IDENTITY,
                            g: vk::ComponentSwizzle::IDENTITY,
                            b: vk::ComponentSwizzle::IDENTITY,
                            a: vk::ComponentSwizzle::IDENTITY,
                        })
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        }),
                )
            })
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        let info = DisplayInfo {
            min_image_count,
            surface_format,
            image_extent,
            present_mode,
        };

        Display {
            info,
            image_views,
            images,
            swapchain: Some(swapchain),
            surface: Some(surface),
            device: device.clone(),
        }
    }

    pub fn info(&self) -> &DisplayInfo {
        &self.info
    }
}
