use std::cmp;

use erupt::vk;

use crate::{
    vks::{self, VkObject, VkSyncObject},
    Device,
};

const SWAPCHAIN_CHOOSES_EXTENT: vk::Extent2D = vk::Extent2D {
    width: u32::MAX,
    height: u32::MAX,
};

pub struct DisplayInfo {
    min_image_count: u32,
    surface_format: vk::SurfaceFormatKHR,
    image_extent: vk::Extent2D,
    present_mode: vk::PresentModeKHR,
}

pub struct Display {
    info: DisplayInfo,

    swapchain: Option<vks::Swapchain>,
    surface: vks::Surface,
    device: Device,
}

impl Display {
    // Safety: device and surface must be from same instance
    pub unsafe fn create(
        device: &Device,
        mut surface: vks::Surface,
        phys_window_extent: vk::Extent2D,
    ) -> Display {
        let device_read = device.inner.read();
        let instance_read = device_read.instance.read_inner();

        let (surf_caps, surf_formats, surf_present_modes) = unsafe {
            let instance_handle = &instance_read.handle;
            let phys = &device_read.phys_device.inner.raw;

            let surface_supported = instance_handle
                .get_physical_device_surface_support_khr(
                    phys.handle(),
                    device.present_family_id(),
                    surface.handle(),
                )
                .expect("failed to verify physical device surface support");

            if !surface_supported {
                panic!("Surface not supported with this device.");
            }

            (
                instance_handle
                    .get_physical_device_surface_capabilities_khr(phys.handle(), surface.handle())
                    .expect("failed to query surface capabilities"),
                instance_handle
                    .get_physical_device_surface_formats_khr(phys.handle(), surface.handle())
                    .expect("failed to query surface formats"),
                instance_handle
                    .get_physical_device_surface_present_modes_khr(phys.handle(), surface.handle())
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
            surface: surface.handle_mut(),
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

        let swapchain = unsafe { device_read.raw.create_swapchain(&mut create_info) }
            .expect("failed to create swapchain");

        log::info!("Successfully created swapchain.");

        let info = DisplayInfo {
            min_image_count,
            surface_format,
            image_extent,
            present_mode,
        };

        Display {
            info,
            swapchain: Some(swapchain),
            surface,
            device: device.clone(),
        }
    }
}
