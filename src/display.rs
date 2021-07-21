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

/// A set of synchronization primitives used to control rendering a frame.
pub struct FrameInFlight {
    // If unsignaled, the frame is currently in flight.
    in_flight: vks::Fence,

    image_available: vks::Semaphore,
    render_complete: vks::Semaphore,

    // Only used if graphics and present queues are from different families.
    present_queue_ownership: vks::Semaphore,
}

impl FrameInFlight {
    fn destroy_with(self, device: &vks::Device) {
        let FrameInFlight {
            in_flight,
            image_available,
            render_complete,
            present_queue_ownership,
        } = self;

        unsafe {
            device.destroy_fence(in_flight);
            device.destroy_semaphore(image_available);
            device.destroy_semaphore(render_complete);
            device.destroy_semaphore(present_queue_ownership);
        }
    }
}

/// A swapchain image, along with the resources used to render to it.
pub struct SwapchainImage {
    present_commands: vks::CommandBuffer,
    graphics_commands: vks::CommandBuffer,
    framebuffer: Option<vks::Framebuffer>,
    view: vks::ImageView,
    image: vks::Image,
}

impl SwapchainImage {
    fn destroy_with(self, device: &vks::Device) {
        let SwapchainImage {
            framebuffer,
            view,
            // Command buffers are destroyed automatically along with their
            // owning pools.
            present_commands: _,
            graphics_commands: _,
            // Swapchain images are destroyed automatically along with their
            // owning swapchain.
            image: _,
        } = self;

        unsafe {
            if let Some(fb) = framebuffer {
                device.destroy_framebuffer(fb);
            }
            device.destroy_image_view(view);
        }
    }
}

pub struct DisplayInfo {
    pub min_image_count: u32,
    pub surface_format: vk::SurfaceFormatKHR,
    pub image_extent: vk::Extent2D,
    pub present_mode: vk::PresentModeKHR,
}

pub struct Display {
    info: DisplayInfo,

    frame: FrameInFlight,
    images: Vec<SwapchainImage>,
    swapchain: Option<vks::SwapchainKHR>,
    surface: Option<vks::SurfaceKHR>,
    device: Device,
}

impl Drop for Display {
    fn drop(&mut self) {
        let device_read = self.device.inner.read();

        // TODO
        // self.frame.destroy_with(&device_read.raw);

        for si in self.images.drain(..) {
            si.destroy_with(&device_read.raw)
        }

        if let Some(swapchain) = self.swapchain.take() {
            unsafe {
                device_read.raw.destroy_swapchain_khr(swapchain);
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

        log::info!("Retrieved {} images from swapchain.", images.len());

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

        let graphics_command_buffers = {
            let graphics_command_pool = device.graphics_command_pool();
            let mut pool_mut = graphics_command_pool
                .get_mut()
                .expect("failed to acquire command pool");
            let allocate_info = vks::CommandBufferAllocateInfoBuilder::new()
                .command_pool(&mut *pool_mut)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(image_views.len() as u32);

            unsafe { device_read.raw.allocate_command_buffers(&allocate_info) }
                .expect("failed to allocate command buffers")
        };

        let present_command_buffers = {
            let present_command_pool = device.present_command_pool();
            let mut pool_mut = present_command_pool
                .get_mut()
                .expect("failed to acquire command pool");
            let allocate_info = vks::CommandBufferAllocateInfoBuilder::new()
                .command_pool(&mut *pool_mut)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(image_views.len() as u32);

            unsafe { device_read.raw.allocate_command_buffers(&allocate_info) }
                .expect("failed to allocate command buffers")
        };

        let swapchain_images = images
            .into_iter()
            .zip(image_views)
            .zip(graphics_command_buffers)
            .zip(present_command_buffers)
            .map(
                |(((image, view), graphics_commands), present_commands)| SwapchainImage {
                    present_commands,
                    graphics_commands,
                    framebuffer: None,
                    view,
                    image,
                },
            )
            .collect::<Vec<_>>();

        let info = DisplayInfo {
            min_image_count,
            surface_format,
            image_extent,
            present_mode,
        };

        let fence_create_info =
            vk::FenceCreateInfoBuilder::new().flags(vk::FenceCreateFlags::SIGNALED);
        let in_flight = unsafe { device_read.raw.create_fence(&fence_create_info) }
            .expect("failed to create in_flight fence");

        let semaphore_create_info = vk::SemaphoreCreateInfoBuilder::new();
        let image_available = unsafe { device_read.raw.create_semaphore(&semaphore_create_info) }
            .expect("failed to create image_available semaphore");
        let render_complete = unsafe { device_read.raw.create_semaphore(&semaphore_create_info) }
            .expect("failed to create render_complete semaphore");
        let present_queue_ownership =
            unsafe { device_read.raw.create_semaphore(&semaphore_create_info) }
                .expect("failed to create present_queue_ownership semaphore");

        Display {
            frame: FrameInFlight {
                in_flight,
                image_available,
                render_complete,
                present_queue_ownership,
            },
            info,
            images: swapchain_images,
            swapchain: Some(swapchain),
            surface: Some(surface),
            device: device.clone(),
        }
    }

    pub fn rebuild_framebuffers(&mut self, render_pass: &vks::RenderPass) {
        for image in self.images.iter_mut() {
            unsafe {
                // Safety: raw handle does not outlive the block.
                let attachments = &[*image.view.handle()];

                let create_info = vks::FramebufferCreateInfoBuilder::new()
                    .flags(vk::FramebufferCreateFlags::empty())
                    .render_pass(render_pass)
                    .attachments(attachments)
                    .width(self.info.image_extent.width)
                    .height(self.info.image_extent.height)
                    .layers(1);

                let framebuffer = self
                    .device
                    .read_inner()
                    .raw
                    .create_framebuffer(&create_info)
                    .expect("failed to create framebuffer");

                if let Some(fb) = image.framebuffer.replace(framebuffer) {
                    self.device.read_inner().raw.destroy_framebuffer(fb);
                }
            }
        }
    }

    pub fn record_graphics_command_buffer(
        &mut self,
        render_pass: &vks::RenderPass,
        pipeline: &vks::Pipeline,
        index: usize,
    ) {
        let device_read = self.device.read_inner();

        let image = &mut self.images[index];
        let cmdbuf = &mut image.graphics_commands;
        let framebuffer = image.framebuffer.as_ref().unwrap();
        let graphics_present_differ =
            self.device.graphics_family_id() != self.device.present_family_id();

        let begin_info =
            vk::CommandBufferBeginInfoBuilder::new().flags(vk::CommandBufferUsageFlags::empty());

        unsafe {
            device_read
                .raw
                .begin_command_buffer(cmdbuf, &begin_info)
                .expect("failed to begin recording graphics command buffer");
        }

        if graphics_present_differ {
            // The EXTERNAL -> 0 subpass dependency was omitted, so insert a
            // barrier to perform the layout transition here.

            let subresource_range = vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: vk::REMAINING_MIP_LEVELS,
                base_array_layer: 0,
                layer_count: vk::REMAINING_ARRAY_LAYERS,
            };

            let pre_render_barrier = vks::ImageMemoryBarrierBuilder::new()
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                // The whole image is being cleared, so it's not necessary
                // to perform a QFOT here.
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(&image.image)
                .subresource_range(subresource_range);

            unsafe {
                device_read.raw.cmd_pipeline_barrier(
                    cmdbuf,
                    // Wait for any previous color output to complete before
                    // beginning output, precluding a WAW hazard.
                    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    // Safety: Produced raw handles do not outlive the block.
                    &[pre_render_barrier.into_inner()],
                );
            }
        }

        let clear_values = &[vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        }];
        let pass_info = vks::RenderPassBeginInfoBuilder::new()
            .render_pass(render_pass)
            .framebuffer(framebuffer)
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: self.info.image_extent,
            })
            .clear_values(clear_values);

        unsafe {
            device_read
                .raw
                .cmd_begin_render_pass(cmdbuf, &pass_info, vk::SubpassContents::INLINE);

            device_read
                .raw
                .cmd_bind_pipeline(cmdbuf, vk::PipelineBindPoint::GRAPHICS, pipeline);

            device_read.raw.cmd_draw(cmdbuf, 3, 1, 0, 0);
            device_read.raw.cmd_end_render_pass(cmdbuf);
        }

        if graphics_present_differ {
            // Need to perform a QFOT from the graphics queue to the present
            // queue. Layout transition can occur simultaneously.

            let subresource_range = vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: vk::REMAINING_MIP_LEVELS,
                base_array_layer: 0,
                layer_count: vk::REMAINING_ARRAY_LAYERS,
            };

            let post_render_barrier = vks::ImageMemoryBarrierBuilder::new()
                // Make all writes to the color attachment available.
                .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                .dst_access_mask(vk::AccessFlags::empty())
                // Transition to presentation layout.
                .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                // Release ownership from graphics queue to present queue.
                .src_queue_family_index(self.device.graphics_family_id())
                .dst_queue_family_index(self.device.present_family_id())
                .image(&image.image)
                .subresource_range(subresource_range);

            unsafe {
                device_read.raw.cmd_pipeline_barrier(
                    cmdbuf,
                    // Wait for color output to complete before ending the
                    // submission.
                    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    // Safety: Produced raw handles do not outlive the block.
                    &[post_render_barrier.into_inner()],
                );
            }
        }

        unsafe {
            device_read
                .raw
                .end_command_buffer(cmdbuf)
                .expect("failed to end recording graphics command buffer");
        }
    }

    pub fn record_command_buffers(
        &mut self,
        render_pass: &vks::RenderPass,
        pipeline: &vks::Pipeline,
    ) {
        let graphics_present_differ =
            self.device.graphics_family_id() != self.device.present_family_id();

        if graphics_present_differ {
            log::info!("using separate graphics and present queues");
        } else {
            log::info!("using shared graphics and present queue");
        }

        for index in 0..self.images.len() {
            self.record_graphics_command_buffer(render_pass, pipeline, index);
            let device_read = self.device.read_inner();
            if graphics_present_differ {
                let image = &mut self.images[index];
                let cmdbuf = &mut image.present_commands;

                let begin_info = vk::CommandBufferBeginInfoBuilder::new()
                    .flags(vk::CommandBufferUsageFlags::empty());

                unsafe {
                    device_read
                        .raw
                        .begin_command_buffer(cmdbuf, &begin_info)
                        .expect("failed to begin presentation command buffer");
                }

                let subresource_range = vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: vk::REMAINING_MIP_LEVELS,
                    base_array_layer: 0,
                    layer_count: vk::REMAINING_ARRAY_LAYERS,
                };

                unsafe {
                    let acquire_attachment_barrier = vk::ImageMemoryBarrierBuilder::new()
                        // No access masks needed: graphics command buffer
                        // submission signals a semaphore, which defines a
                        // memory dependency. (§6.4.1 Semaphore Signaling)
                        .src_access_mask(vk::AccessFlags::empty())
                        .dst_access_mask(vk::AccessFlags::empty())
                        .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                        .src_queue_family_index(self.device.graphics_family_id())
                        .dst_queue_family_index(self.device.present_family_id())
                        .image(*image.image.handle())
                        .subresource_range(subresource_range);

                    device_read.raw.cmd_pipeline_barrier(
                        cmdbuf,
                        vk::PipelineStageFlags::TOP_OF_PIPE,
                        vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[acquire_attachment_barrier],
                    );

                    device_read
                        .raw
                        .end_command_buffer(cmdbuf)
                        .expect("failed to end recording present command buffer");
                }
            }
        }
    }

    pub fn draw(&mut self) {
        log::trace!("drawing");
        let device_read = self.device.read_inner();

        let graphics_present_differ =
            self.device.graphics_family_id() != self.device.present_family_id();

        let acquired = unsafe {
            device_read.raw.acquire_next_image_khr(
                self.swapchain.as_mut().unwrap(),
                None,
                Some(&mut self.frame.image_available),
                None,
            )
        }
        .expect("failed to acquire next swapchain image");

        let graphics_queue = self.device.graphics_queue();
        let mut graphics_queue_write = graphics_queue.write_inner();

        // Submit graphics commands.
        unsafe {
            // Safety: raw handles do not outlive the block.

            let wait_semaphores = &[*self.frame.image_available.handle_mut()];
            // Only block rendering once ready to output to the color attachment.
            let wait_dst_stage_mask = &[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
            let command_buffers = &[*self.images[acquired.index as usize]
                .graphics_commands
                .handle_mut()];
            let signal_semaphores = &[*self.frame.render_complete.handle_mut()];

            let submit_info = vk::SubmitInfoBuilder::new()
                .wait_semaphores(wait_semaphores)
                .wait_dst_stage_mask(wait_dst_stage_mask)
                .command_buffers(command_buffers)
                .signal_semaphores(signal_semaphores);

            let submits = &[submit_info];

            log::trace!("submitting draw commands");
            device_read
                .raw
                .queue_submit(&mut graphics_queue_write.raw, submits, None)
                .expect("failed to submit graphics command buffer");
        }

        let present_queue = self.device.present_queue();
        let mut present_queue_write = present_queue.write_inner();

        if graphics_present_differ {
            // Submit present queue commands. This acquires the swapchain image
            // from the graphics queue.
            unsafe {
                let wait_semaphores = &[*self.frame.render_complete.handle_mut()];
                let wait_dst_stage_mask = &[vk::PipelineStageFlags::ALL_COMMANDS];
                let command_buffers = &[*self.images[acquired.index as usize]
                    .present_commands
                    .handle_mut()];
                let signal_semaphores = &[*self.frame.present_queue_ownership.handle_mut()];

                let submit_info = vk::SubmitInfoBuilder::new()
                    .wait_semaphores(wait_semaphores)
                    .wait_dst_stage_mask(wait_dst_stage_mask)
                    .command_buffers(command_buffers)
                    .signal_semaphores(signal_semaphores);

                let submits = &[submit_info];

                log::trace!("submitting presentation queue acquire");
                device_read
                    .raw
                    .queue_submit(&mut present_queue_write.raw, submits, None)
                    .expect("failed to submit command buffer");
            }
        }

        unsafe {
            let present_wait_semaphore = if graphics_present_differ {
                *self.frame.present_queue_ownership.handle_mut()
            } else {
                *self.frame.render_complete.handle_mut()
            };

            let wait_semaphores = &[present_wait_semaphore];
            let swapchains = &[*self.swapchain.as_mut().unwrap().handle_mut()];
            let image_indices = &[acquired.index];

            let present_info = vk::PresentInfoKHRBuilder::new()
                .wait_semaphores(wait_semaphores)
                .swapchains(swapchains)
                .image_indices(image_indices);

            log::trace!("presenting swapchain image");
            device_read
                .raw
                .queue_present_khr(&mut present_queue_write.raw, &present_info)
                .expect("failed to present swapchain image");
            device_read
                .raw
                .queue_wait_idle(&mut present_queue_write.raw)
                .expect("failed to wait for presentation queue to become idle");
        }
    }

    pub fn info(&self) -> &DisplayInfo {
        &self.info
    }
}
