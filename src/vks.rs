//! Safer (but still unsafe) wrappers over the Vulkan API.
//!
//! This module provides zero-cost wrappers for Vulkan objects and functions
//! which eliminate certain invalid usages of the API without incurring runtime
//! overhead. In particular, Rust's ownership and borrowing rules are used to
//! enforce Vulkan's external synchronization rules: functions which require
//! that a parameter be externally synchronized accept the object type either by
//! mutable reference or by value (in case the function may destroy the object).

use erupt::{utils::VulkanResult, vk, DeviceLoader, EntryLoader, InstanceLoader, LoaderError};

use std::{convert::TryInto, ffi::CStr, lazy::SyncOnceCell, time::Duration};

static ENTRY: SyncOnceCell<EntryLoader> = SyncOnceCell::new();

pub fn entry() -> &'static EntryLoader {
    ENTRY
        .get_or_try_init(|| EntryLoader::new())
        .expect("failed to load Vulkan dynamic library")
}

pub type VkResult<T> = std::result::Result<T, vk::Result>;

/// A Vulkan API object.
///
/// This trait represents a Vulkan object. Types which implement this trait have
/// exclusive ownership of a Vulkan object and may allow shared use of the
/// object via the `handle` method or exclusive use via the `handle_mut` method.
///
/// # Safety
///
/// Implementors of this trait must uphold the following invariants:
/// - The implementing type must not implement `Copy`.
/// - If a value of the implementing type can be duplicated (e.g., if the type is
///   `Clone`), duplicating the value must not duplicate the underlying raw
///   handle.
/// - The raw handle owned by values of the implementing type must not be
///   accessible except via the `handle` or `handle_mut` trait methods.
pub unsafe trait VkObject {
    /// The raw handle type.
    type Handle;

    /// Returns a reference to the underlying Vulkan object.
    ///
    /// # Safety
    ///
    /// The caller must uphold the following invariants:
    /// - The raw handle must not be used as a parameter in any Vulkan API call
    ///   which specifies an external synchronization requirement on that
    ///   parameter.
    /// - If the raw handle is copied, it is the caller's responsibility to
    ///   uphold Vulkan's external synchronization rules until the copy is
    ///   dropped.
    unsafe fn handle(&self) -> &Self::Handle;

    /// Returns a mutable reference to the underlying Vulkan object.
    ///
    /// # Safety
    ///
    /// The caller must uphold the following invariants:
    /// - If the raw handle is copied, it is the caller's responsibility to
    ///   uphold Vulkan's external synchronization rules until the copy is
    ///   dropped.
    unsafe fn handle_mut(&mut self) -> &mut Self::Handle;
}

/// A macro to define a wrapper type around a raw Vulkan handle and implement
/// `VkObject` for the wrapper type.
macro_rules! define_handle {
    ($(#[$outer:meta])? $v:vis struct $defty:ident ($raw:ty);) => {
        $(#[$outer])?
        $v struct $defty {
            raw: $raw,
        }

        unsafe impl VkObject for $defty {
            type Handle = $raw;

            #[inline(always)]
            unsafe fn handle(&self) -> &Self::Handle {
                &self.raw
            }

            #[inline(always)]
            unsafe fn handle_mut(&mut self) -> &mut Self::Handle {
                &mut self.raw
            }
        }

        impl $defty {
            unsafe fn new(raw: $raw) -> $defty {
                $defty { raw }
            }
        }
    };
}

// TODO: Unclear if there's an ergonomic way to delegate builder methods from
// &[&VkObject] to &[VkObject::Handle]. It would probably require builders to be
// self-referential, which is not great.
/// Defines a builder struct which delegates the selected methods to an `erupt`
/// builder.
///
/// This allows `VkObject`'s borrowing semantics to integrate with builders
/// without fully rewriting every struct.
macro_rules! define_delegated_builder {
    (
        $(#[$outer_meta:meta])*
        $ty_vis:vis struct $typename:ident<'a> {
            inner: vk::$typename2:ident<'a>,
        }

        impl $typename3:ident {
            $(
                $(#[$method_meta:meta])*
                $method_vis:vis fn $method_name:ident($method_arg:ty) -> Self;
            )*
        }
    ) => {
        $(#[$outer_meta])*
        $ty_vis struct $typename<'a> {
            inner: vk::$typename<'a>,
        }

        impl<'a> $typename2<'a> {
            pub fn new() -> Self {
                Self {
                    inner: vk::$typename::new(),
                }
            }

            /// Returns the underlying `erupt` builder.
            ///
            /// # Safety
            ///
            /// This discards lifetime information associated with `VkObject`
            /// handles that have been passed into the builder. Callers must
            /// ensure that any handles referenced by this builder outlive the
            /// value returned from this method.
            pub unsafe fn into_inner(self) -> vk::$typename<'a> {
                self.inner
            }

            $(
                $(#[$method_meta])*
                $method_vis fn $method_name(mut self, $method_name: $method_arg) -> Self {
                    self.inner = self.inner.$method_name($method_name);
                    self
                }
            )*
        }
    };
}

// ============================================================================

pub struct Instance {
    // Boxed to reduce stack space usage.
    loader: Box<InstanceLoader>,
}

impl Drop for Instance {
    fn drop(&mut self) {
        unsafe {
            // Safety: instance is not used again.
            self.loader.destroy_instance(None);
        }
    }
}

unsafe impl VkObject for Instance {
    type Handle = InstanceLoader;

    unsafe fn handle(&self) -> &Self::Handle {
        &self.loader
    }

    unsafe fn handle_mut(&mut self) -> &mut Self::Handle {
        &mut self.loader
    }
}

impl Instance {
    /// Creates a new Vulkan instance.
    #[inline]
    pub fn create(create_info: &vk::InstanceCreateInfo) -> Result<Instance, LoaderError> {
        let loader = unsafe { InstanceLoader::new(entry(), create_info, None) }?;
        Ok(Instance {
            loader: Box::new(loader),
        })
    }

    #[inline]
    pub fn enumerate_physical_devices(&self) -> VkResult<Vec<PhysicalDevice>> {
        unsafe {
            Ok(self
                .loader
                .enumerate_physical_devices(None)
                .result()?
                .into_iter()
                .map(|raw| PhysicalDevice::new(raw))
                .collect())
        }
    }

    /// Reports capabilities of a physical device.
    ///
    /// # Safety
    ///
    /// The caller must uphold the following invariants:
    /// - `phys_device` must be a physical device handle associated with this
    ///   instance.
    #[inline]
    pub unsafe fn get_physical_device_features(
        &self,
        phys_device: &PhysicalDevice,
    ) -> vk::PhysicalDeviceFeatures {
        unsafe {
            self.loader
                .get_physical_device_features(*phys_device.handle())
        }
    }

    /// Returns properties of a physical device.
    ///
    /// # Safety
    ///
    /// The caller must uphold the following invariants:
    /// - `phys_device` must be a physical device handle associated with this
    ///   instance.
    #[inline]
    pub unsafe fn get_physical_device_properties(
        &self,
        phys_device: &PhysicalDevice,
    ) -> vk::PhysicalDeviceProperties {
        unsafe {
            self.loader
                .get_physical_device_properties(*phys_device.handle())
        }
    }

    /// Reports properties of the queues of the specified physical device.
    ///
    /// # Safety
    ///
    /// The caller must uphold the following invariants:
    /// - `phys_device` must be a physical device handle associated with this
    ///   instance.
    #[inline]
    pub unsafe fn get_physical_device_queue_family_properties(
        &self,
        phys_device: &PhysicalDevice,
    ) -> Vec<vk::QueueFamilyProperties> {
        unsafe {
            self.loader
                .get_physical_device_queue_family_properties(*phys_device.handle(), None)
        }
    }

    /// Reports memory information for the specified physical device.
    #[inline]
    pub fn get_physical_device_memory_properties(
        &self,
        phys_device: &PhysicalDevice,
    ) -> vk::PhysicalDeviceMemoryProperties {
        unsafe {
            self.loader
                .get_physical_device_memory_properties(*phys_device.handle())
        }
    }

    /// Queries if presentation is supported.
    ///
    /// # Safety
    ///
    /// The caller must uphold the following invariants:
    /// - `phys_device` must be a physical device handle associated with this
    ///   instance.
    /// - `surface` must be a surface handle associated with this instance.
    #[inline]
    pub unsafe fn get_physical_device_surface_support_khr(
        &self,
        phys_device: &PhysicalDevice,
        queue_family_index: u32,
        surface: &SurfaceKHR,
    ) -> VkResult<bool> {
        unsafe {
            self.loader
                .get_physical_device_surface_support_khr(
                    *phys_device.handle(),
                    queue_family_index,
                    *surface.handle(),
                )
                .result()
        }
    }

    /// Queries surface capabilities.
    ///
    /// # Safety
    ///
    /// The caller must uphold the following invariants:
    /// - `phys_device` must be a physical device handle associated with this
    ///   instance.
    /// - `surface` must be a surface handle associated with this instance.
    #[inline]
    pub unsafe fn get_physical_device_surface_capabilities_khr(
        &self,
        phys_device: &PhysicalDevice,
        surface: &SurfaceKHR,
    ) -> VkResult<vk::SurfaceCapabilitiesKHR> {
        unsafe {
            self.loader
                .get_physical_device_surface_capabilities_khr(
                    *phys_device.handle(),
                    *surface.handle(),
                )
                .result()
        }
    }

    /// Queries color formats supported by a surface.
    ///
    /// # Safety
    ///
    /// The caller must uphold the following invariants:
    /// - `phys_device` must be a physical device handle associated with this
    ///   instance.
    /// - `surface` must be a surface handle associated with this instance.
    #[inline]
    pub unsafe fn get_physical_device_surface_formats_khr(
        &self,
        phys_device: &PhysicalDevice,
        surface: &SurfaceKHR,
    ) -> VkResult<Vec<vk::SurfaceFormatKHR>> {
        unsafe {
            self.loader
                .get_physical_device_surface_formats_khr(
                    *phys_device.handle(),
                    *surface.handle(),
                    None,
                )
                .result()
        }
    }

    /// Queries supported presentation modes.
    ///
    /// # Safety
    ///
    /// The caller must uphold the following invariants:
    /// - `phys_device` must be a physical device handle associated with this
    ///   instance.
    /// - `surface` must be a surface handle associated with this instance.
    #[inline]
    pub unsafe fn get_physical_device_surface_present_modes_khr(
        &self,
        phys_device: &PhysicalDevice,
        surface: &SurfaceKHR,
    ) -> VkResult<Vec<vk::PresentModeKHR>> {
        unsafe {
            self.loader
                .get_physical_device_surface_present_modes_khr(
                    *phys_device.handle(),
                    *surface.handle(),
                    None,
                )
                .result()
        }
    }

    /// Creates a new device instance.
    ///
    /// # Safety
    ///
    /// The caller must uphold the following invariants:
    /// - `phys_device` must be a physical device handle associated with this
    ///   instance.
    #[inline]
    pub unsafe fn create_device(
        &self,
        phys_device: &PhysicalDevice,
        create_info: &vk::DeviceCreateInfo,
    ) -> Result<Device, LoaderError> {
        unsafe {
            let loader = DeviceLoader::new(&self.loader, *phys_device.handle(), create_info, None)?;
            Ok(Device::new(loader))
        }
    }

    #[inline]
    pub unsafe fn create_debug_utils_messenger(
        &self,
        info: &vk::DebugUtilsMessengerCreateInfoEXT,
    ) -> VkResult<DebugUtilsMessengerEXT> {
        unsafe {
            let debug_messenger = self
                .loader
                .create_debug_utils_messenger_ext(info, None)
                .result()?;

            Ok(DebugUtilsMessengerEXT::new(debug_messenger))
        }
    }

    pub unsafe fn destroy_debug_utils_messenger(&self, mut messenger: DebugUtilsMessengerEXT) {
        // Safety:
        // - Access to debug messenger is externally synchronized via ownership.
        unsafe {
            self.loader
                .destroy_debug_utils_messenger_ext(Some(*messenger.handle_mut()), None);
        }
    }

    pub unsafe fn create_xcb_surface_khr(
        &self,
        create_info: &vk::XcbSurfaceCreateInfoKHR,
    ) -> VkResult<SurfaceKHR> {
        unsafe {
            self.loader
                .create_xcb_surface_khr(create_info, None)
                .result()
                .map(|s| SurfaceKHR::new(s))
        }
    }

    pub unsafe fn create_xlib_surface_khr(
        &self,
        create_info: &vk::XlibSurfaceCreateInfoKHR,
    ) -> VkResult<SurfaceKHR> {
        unsafe {
            self.loader
                .create_xlib_surface_khr(create_info, None)
                .result()
                .map(|s| SurfaceKHR::new(s))
        }
    }

    pub unsafe fn destroy_surface(&self, mut surface: SurfaceKHR) {
        // Safety:
        // - Access to surface is externally synchronized via ownership.
        unsafe {
            self.loader
                .destroy_surface_khr(Some(*surface.handle_mut()), None);
        }
    }
}

// ============================================================================

define_handle! {
    /// An opaque handle to a Vulkan physical device object.
    pub struct PhysicalDevice(vk::PhysicalDevice);
}

// ============================================================================

pub struct Device {
    loader: Box<DeviceLoader>,
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe {
            // Safety: device is not used again.
            self.loader.destroy_device(None);
        }
    }
}

unsafe impl VkObject for Device {
    type Handle = DeviceLoader;

    unsafe fn handle(&self) -> &Self::Handle {
        &self.loader
    }

    unsafe fn handle_mut(&mut self) -> &mut Self::Handle {
        &mut self.loader
    }
}

impl Device {
    /// Constructs a new `Device` which owns the logical device object
    /// associated with `loader`.
    fn new(loader: DeviceLoader) -> Device {
        Device {
            loader: Box::new(loader),
        }
    }

    // ------------------------------------------------------------------------

    /// Retrieves a queue handle from a device.
    ///
    /// # Safety
    ///
    /// **This function must not be called multiple times with the same
    /// inputs.** Doing so violates the ownership semantics of the `Queue`
    /// handle by producing multiple values with the same underlying handle.
    /// It is therefore recommended to call `get_device_queue` once for every
    /// queue immediately upon device construction, and not again after that.
    ///
    /// Additionally, the caller must uphold the following invariants:
    /// - `queue_family_index` must be one of the queue family indices specified
    ///   when the device was created.
    /// - `queue_index` must be less than the number of queues created for the
    ///   specified queue family when the device was created.
    /// - The `flags` field of the `vk::DeviceCreateInfo` used to create the
    ///   device must have been zero.
    pub unsafe fn get_device_queue(&self, queue_family_index: u32, queue_index: u32) -> Queue {
        unsafe {
            Queue::new(
                self.loader
                    .get_device_queue(queue_family_index, queue_index),
            )
        }
    }

    pub unsafe fn queue_submit(
        &self,
        queue: &mut Queue,
        submits: &[vk::SubmitInfoBuilder<'_>],
        fence: Option<&mut Fence>,
    ) -> VkResult<()> {
        unsafe {
            self.loader
                .queue_submit(*queue.handle_mut(), submits, fence.map(|f| *f.handle_mut()))
                .result()
        }
    }

    pub unsafe fn queue_wait_idle(&self, queue: &mut Queue) -> VkResult<()> {
        unsafe { self.loader.queue_wait_idle(*queue.handle_mut()).result() }
    }

    // ------------------------------------------------------------------------

    /// Creates an image view from an existing image.
    ///
    /// # Safety
    ///
    /// // TODO
    pub unsafe fn create_image_view(
        &self,
        create_info: &ImageViewCreateInfoBuilder<'_>,
    ) -> VkResult<ImageView> {
        unsafe {
            self.loader
                .create_image_view(&create_info.inner, None)
                .result()
                .map(|v| ImageView::new(v))
        }
    }

    /// Destroys an image view object.
    ///
    /// # Safety
    ///
    /// The caller must uphold the following invariants:
    /// - All submitted commands that refer to `image_view` must have completed
    ///   execution.
    /// - `image_view` must be a handle to an image view object associated with
    ///   this device.
    pub unsafe fn destroy_image_view(&self, mut image_view: ImageView) {
        unsafe {
            self.loader
                .destroy_image_view(Some(*image_view.handle_mut()), None)
        }
    }

    // ------------------------------------------------------------------------

    /// Creates a framebuffer from an existing image.
    ///
    /// # Safety
    ///
    /// // TODO
    pub unsafe fn create_framebuffer(
        &self,
        create_info: &FramebufferCreateInfoBuilder<'_>,
    ) -> VkResult<Framebuffer> {
        unsafe {
            self.loader
                .create_framebuffer(&create_info.inner, None)
                .result()
                .map(|v| Framebuffer::new(v))
        }
    }

    /// Destroys a framebuffer object.
    ///
    /// # Safety
    ///
    /// The caller must uphold the following invariants:
    /// - All submitted commands that refer to `framebuffer` must have completed
    ///   execution.
    /// - `framebuffer` must be a handle to an framebuffer object associated with
    ///   this device.
    pub unsafe fn destroy_framebuffer(&self, mut framebuffer: Framebuffer) {
        unsafe {
            self.loader
                .destroy_framebuffer(Some(*framebuffer.handle_mut()), None)
        }
    }

    // ------------------------------------------------------------------------

    /// Creates a new shader module object.
    ///
    /// # Safety
    ///
    /// - TODO: must destroy before destroying parent device
    pub unsafe fn create_shader_module(
        &self,
        create_info: &vk::ShaderModuleCreateInfo,
    ) -> VkResult<ShaderModule> {
        unsafe {
            self.loader
                .create_shader_module(create_info, None)
                .result()
                .map(|sm| ShaderModule::new(sm))
        }
    }

    /// Destroys a shader module.
    ///
    /// # Safety
    ///
    /// - `module` must be a shader module created by this device.
    pub unsafe fn destroy_shader_module(&self, mut module: ShaderModule) {
        unsafe {
            self.loader
                .destroy_shader_module(Some(*module.handle_mut()), None);
        }
    }

    // ------------------------------------------------------------------------

    /// Creates a new render pass object.
    ///
    /// # Safety
    ///
    /// - TODO: destroy before destroying parent
    pub unsafe fn create_render_pass(
        &self,
        create_info: &vk::RenderPassCreateInfoBuilder<'_>,
    ) -> VkResult<RenderPass> {
        unsafe {
            self.loader
                .create_render_pass(create_info, None)
                .result()
                .map(|rp| RenderPass::new(rp))
        }
    }

    /// Destroys a render pass object.
    ///
    /// # Safety
    ///
    /// - `render_pass` must be a handle to a render pass object associated with
    ///   this device.
    pub unsafe fn destroy_render_pass(&self, mut render_pass: RenderPass) {
        unsafe {
            self.loader
                .destroy_render_pass(Some(*render_pass.handle_mut()), None);
        }
    }

    // ------------------------------------------------------------------------

    /// Creates a new pipeline layout object.
    ///
    /// # Safety
    ///
    /// - TODO: destroy before destroying parent
    pub unsafe fn create_pipeline_layout(
        &self,
        create_info: &vk::PipelineLayoutCreateInfoBuilder<'_>,
    ) -> VkResult<PipelineLayout> {
        unsafe {
            self.loader
                .create_pipeline_layout(create_info, None)
                .result()
                .map(|pl| PipelineLayout::new(pl))
        }
    }

    /// Destroys a pipeline layout object.
    ///
    /// # Safety
    ///
    /// - `pipeline_layout` must be a handle to a pipeline layout object
    ///   associated with this instance.
    pub unsafe fn destroy_pipeline_layout(&self, mut pipeline_layout: PipelineLayout) {
        unsafe {
            self.loader
                .destroy_pipeline_layout(Some(*pipeline_layout.handle_mut()), None)
        }
    }

    // ------------------------------------------------------------------------

    /// Creates graphics pipelines.
    ///
    /// # Safety
    ///
    /// - TODO: lots of handles
    pub unsafe fn create_graphics_pipelines(
        &self,
        create_infos: &[vk::GraphicsPipelineCreateInfoBuilder<'_>],
    ) -> VkResult<Vec<Pipeline>> {
        unsafe {
            Ok(self
                .loader
                .create_graphics_pipelines(None, create_infos, None)
                .result()?
                .into_iter()
                .map(|p| Pipeline::new(p))
                .collect())
        }
    }

    /// Destroys a pipeline object.
    ///
    /// # Safety
    ///
    /// - TODO: destroy before parent device and depended handles
    pub unsafe fn destroy_pipeline(&self, mut pipeline: Pipeline) {
        unsafe {
            self.loader
                .destroy_pipeline(Some(*pipeline.handle_mut()), None);
        }
    }

    // ------------------------------------------------------------------------

    /// Creates a new command pool object.
    ///
    /// # Safety
    ///
    /// - TODO: destroy before parent device
    pub unsafe fn create_command_pool(
        &self,
        create_info: &vk::CommandPoolCreateInfo,
    ) -> VkResult<CommandPool> {
        unsafe {
            self.loader
                .create_command_pool(create_info, None)
                .result()
                .map(|c| CommandPool::new(c))
        }
    }

    /// Destroys a command pool object.
    ///
    /// # Safety
    ///
    /// The caller must uphold the following invariants:
    /// - `command_pool` must be a handle to a command pool object associated
    ///   with this device.
    /// - No command buffer allocated from this command pool may be in the
    ///   pending state.
    pub unsafe fn destroy_command_pool(&self, mut command_pool: CommandPool) {
        unsafe {
            self.loader
                .destroy_command_pool(Some(*command_pool.handle_mut()), None)
        }
    }

    // ------------------------------------------------------------------------

    /// Allocates command buffers from an existing command pool.
    ///
    /// # Safety
    ///
    /// The caller must uphold the following invariants:
    /// - `allocate_info.command_pool` must be a handle to a command pool object
    ///   associated with this device.
    pub unsafe fn allocate_command_buffers(
        &self,
        allocate_info: &CommandBufferAllocateInfoBuilder,
    ) -> VkResult<Vec<CommandBuffer>> {
        if allocate_info.inner.command_buffer_count == 0 {
            panic!("allocate_command_buffers: command_buffer_count must not be 0");
        }

        unsafe {
            Ok(self
                .loader
                .allocate_command_buffers(&allocate_info.inner)
                .result()?
                .into_iter()
                .map(|cb| CommandBuffer::new(cb))
                .collect())
        }
    }

    /// Begins recording a command buffer.
    ///
    /// # Safety
    ///
    /// The caller must uphold the following invariants:
    /// - `command_buffer` must be a handle to a command buffer object
    ///   associated with this device.
    #[inline]
    pub unsafe fn begin_command_buffer(
        &self,
        command_buffer: &mut CommandBuffer,
        begin_info: &vk::CommandBufferBeginInfoBuilder<'_>,
    ) -> VkResult<()> {
        unsafe {
            // Safety: command_buffer is externally synchronized via mutable reference.
            self.loader
                .begin_command_buffer(*command_buffer.handle_mut(), begin_info)
                .result()?;
        }

        Ok(())
    }

    /// Finishes recording a command buffer.
    ///
    /// # Safety
    ///
    /// The caller must uphold the following invariants:
    /// - `command_buffer` must be a handle to a command buffer object
    ///   associated with this device.
    /// - `command_buffer` must be in the recording state.
    /// - TODO: more valid usage details
    #[inline]
    pub unsafe fn end_command_buffer(&self, command_buffer: &mut CommandBuffer) -> VkResult<()> {
        unsafe {
            self.loader
                .end_command_buffer(*command_buffer.handle_mut())
                .result()
        }
    }

    #[inline]
    pub unsafe fn cmd_pipeline_barrier(
        &self,
        command_buffer: &mut CommandBuffer,
        src_stage_mask: vk::PipelineStageFlags,
        dst_stage_mask: vk::PipelineStageFlags,
        dependency_flags: vk::DependencyFlags,
        memory_barriers: &[vk::MemoryBarrierBuilder<'_>],
        buffer_memory_barriers: &[vk::BufferMemoryBarrierBuilder<'_>],
        image_memory_barriers: &[vk::ImageMemoryBarrierBuilder<'_>],
    ) {
        unsafe {
            self.loader.cmd_pipeline_barrier(
                *command_buffer.handle_mut(),
                src_stage_mask,
                dst_stage_mask,
                Some(dependency_flags),
                memory_barriers,
                buffer_memory_barriers,
                image_memory_barriers,
            );
        }
    }

    #[inline]
    pub unsafe fn cmd_begin_render_pass(
        &self,
        command_buffer: &mut CommandBuffer,
        begin_info: &RenderPassBeginInfoBuilder<'_>,
        contents: vk::SubpassContents,
    ) {
        unsafe {
            self.loader.cmd_begin_render_pass(
                *command_buffer.handle_mut(),
                &begin_info.inner,
                contents,
            );
        }
    }

    pub unsafe fn cmd_bind_pipeline(
        &self,
        command_buffer: &mut CommandBuffer,
        bind_point: vk::PipelineBindPoint,
        pipeline: &Pipeline,
    ) {
        unsafe {
            self.loader.cmd_bind_pipeline(
                *command_buffer.handle_mut(),
                bind_point,
                *pipeline.handle(),
            );
        }
    }

    pub unsafe fn cmd_draw(
        &self,
        command_buffer: &mut CommandBuffer,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) {
        unsafe {
            self.loader.cmd_draw(
                *command_buffer.handle_mut(),
                vertex_count,
                instance_count,
                first_vertex,
                first_instance,
            );
        }
    }

    pub unsafe fn cmd_end_render_pass(&self, command_buffer: &mut CommandBuffer) {
        unsafe {
            self.loader
                .cmd_end_render_pass(*command_buffer.handle_mut());
        }
    }

    // ------------------------------------------------------------------------

    /// Creates a new fence object.
    ///
    /// # Safety
    ///
    /// // TODO
    pub unsafe fn create_fence(&self, create_info: &vk::FenceCreateInfo) -> VkResult<Fence> {
        unsafe {
            self.loader
                .create_fence(create_info, None)
                .result()
                .map(|f| Fence::new(f))
        }
    }

    /// Destroys a fence object.
    ///
    /// # Safety
    ///
    /// The caller must uphold the following invariants:
    /// - `fence` must be a handle to a fence object associated with this device.
    /// - All queue submission commands that refer to `fence` must have
    ///   completed execution.
    pub unsafe fn destroy_fence(&self, mut fence: Fence) {
        unsafe { self.loader.destroy_fence(Some(*fence.handle_mut()), None) }
    }

    /// Queries the status of a fence.
    ///
    /// # Safety
    ///
    /// The caller must uphold the following invariants:
    /// - `fence` must be a handle to a fence object associated with this device.
    pub unsafe fn get_fence_status(&self, fence: &Fence) -> VkResult<FenceStatus> {
        unsafe {
            match self.loader.get_fence_status(*fence.handle()).raw {
                vk::Result::SUCCESS => Ok(FenceStatus::Signaled),
                vk::Result::NOT_READY => Ok(FenceStatus::Unsignaled),
                e => Err(e),
            }
        }
    }

    pub unsafe fn reset_fences(&self, fences: &[vk::Fence]) -> VkResult<()> {
        unsafe { self.loader.reset_fences(fences).result() }
    }

    pub unsafe fn wait_for_fences(
        &self,
        fences: &[vk::Fence],
        wait_all: bool,
        timeout: Option<Duration>,
    ) -> VkResult<FenceWaitStatus> {
        let timeout_ns: u64 = timeout
            .map(|dur| {
                dur.as_nanos()
                    .try_into()
                    .expect("wait_for_fences: timeout overflowed u64")
            })
            .unwrap_or(u64::MAX);

        unsafe {
            match self
                .loader
                .wait_for_fences(fences, wait_all, timeout_ns)
                .raw
            {
                vk::Result::SUCCESS => Ok(FenceWaitStatus::Signaled),
                vk::Result::TIMEOUT => Ok(FenceWaitStatus::TimedOut),
                e => Err(e),
            }
        }
    }

    // ------------------------------------------------------------------------

    pub unsafe fn create_semaphore(
        &self,
        create_info: &vk::SemaphoreCreateInfoBuilder<'_>,
    ) -> VkResult<Semaphore> {
        unsafe {
            self.loader
                .create_semaphore(create_info, None)
                .result()
                .map(|s| Semaphore::new(s))
        }
    }

    pub unsafe fn destroy_semaphore(&self, mut semaphore: Semaphore) {
        unsafe {
            self.loader
                .destroy_semaphore(Some(*semaphore.handle_mut()), None);
        }
    }

    // ------------------------------------------------------------------------

    /// Creates a swapchain.
    ///
    /// # Safety
    ///
    /// The caller must uphold the following invariants:
    /// - `create_info.surface` must be a handle to a surface associated with
    ///   this instance.
    /// - If `create_info.old_swapchain` is `Some(old)`, then `old` must be a
    ///   handle to a non-retired swapchain associated with
    ///   `create_info.surface`.
    #[inline]
    pub unsafe fn create_swapchain_khr(
        &self,
        create_info: &mut SwapchainCreateInfo<'_, '_>,
    ) -> VkResult<SwapchainKHR> {
        unsafe {
            let raw_create_info = vk::SwapchainCreateInfoKHRBuilder::new()
                .flags(create_info.flags)
                // Safety: surface is externally synchronized via mutable borrow.
                .surface(*create_info.surface.handle_mut())
                .min_image_count(create_info.min_image_count)
                .image_format(create_info.image_format)
                .image_color_space(create_info.image_color_space)
                .image_extent(create_info.image_extent)
                .image_array_layers(create_info.image_array_layers)
                .image_usage(create_info.image_usage)
                .image_sharing_mode(create_info.image_sharing_mode)
                .queue_family_indices(create_info.queue_family_indices)
                .pre_transform(create_info.pre_transform)
                .composite_alpha(create_info.composite_alpha)
                .present_mode(create_info.present_mode)
                .clipped(create_info.clipped)
                // Safety: old swapchain, if any, is externally synchronized via ownership.
                .old_swapchain(
                    create_info
                        .old_swapchain
                        .take()
                        .map(|mut os| *os.handle_mut())
                        .unwrap_or(vk::SwapchainKHR::null()),
                );

            let swapchain = self
                .loader
                .create_swapchain_khr(&raw_create_info, None)
                .result()?;

            Ok(SwapchainKHR::new(swapchain))
        }
    }

    /// Destroy a swapchain object.
    ///
    /// # Safety
    ///
    /// The caller must uphold the following invariants:
    /// - `swapchain` must be a swapchain object associated with this instance.
    pub unsafe fn destroy_swapchain_khr(&self, mut swapchain: SwapchainKHR) {
        unsafe {
            // Safety:
            // - `swapchain` is externally synchronized, as it is received by value.
            self.loader
                .destroy_swapchain_khr(Some(*swapchain.handle_mut()), None);
        }
    }

    /// Obtains the array of presentable images associated with a swapchain.
    ///
    /// # Safety
    ///
    /// The caller must uphold the following invariants:
    /// - `swapchain` must be a handle to a swapchain object associated with
    ///   this logical device.
    #[inline]
    pub unsafe fn get_swapchain_images_khr(
        &self,
        swapchain: &SwapchainKHR,
    ) -> VkResult<Vec<Image>> {
        unsafe {
            Ok(self
                .loader
                .get_swapchain_images_khr(*swapchain.handle(), None)
                .result()?
                .into_iter()
                .map(|img| Image::new(img))
                .collect())
        }
    }

    pub unsafe fn acquire_next_image_khr(
        &self,
        swapchain: &mut SwapchainKHR,
        timeout: Option<Duration>,
        semaphore: Option<&mut Semaphore>,
        fence: Option<&mut Fence>,
    ) -> VkResult<AcquiredImage> {
        let timeout_ns: u64 = timeout
            .map(|t| {
                t.as_nanos()
                    .try_into()
                    .expect("acquire_next_image_khr: timeout overflowed u64")
            })
            .unwrap_or(u64::MAX);

        let VulkanResult { raw, value } = unsafe {
            self.loader.acquire_next_image_khr(
                *swapchain.handle_mut(),
                timeout_ns,
                semaphore.map(|s| *s.handle_mut()),
                fence.map(|f| *f.handle_mut()),
            )
        };

        match value {
            Some(v) => Ok(AcquiredImage {
                status: raw,
                index: v,
            }),
            None => Err(raw),
        }
    }

    pub unsafe fn queue_present_khr(
        &self,
        queue: &mut Queue,
        present_info: &vk::PresentInfoKHRBuilder<'_>,
    ) -> VkResult<()> {
        unsafe {
            self.loader
                .queue_present_khr(*queue.handle_mut(), present_info)
                .result()
        }
    }
}

// ============================================================================

define_handle! {
    /// An opaque handle to a Vulkan queue object.
    pub struct Queue(vk::Queue);
}

// ============================================================================

define_handle! {
    /// An opaque handle to a Vulkan image object.
    pub struct Image(vk::Image);
}

// ============================================================================

define_handle! {
    /// An opaque handle to a Vulkan image view object.
    pub struct ImageView(vk::ImageView);
}

define_delegated_builder! {
    pub struct ImageViewCreateInfoBuilder<'a> {
        inner: vk::ImageViewCreateInfoBuilder<'a>,
    }

    impl ImageViewCreateInfoBuilder {
        pub fn flags(vk::ImageViewCreateFlags) -> Self;
        pub fn view_type(vk::ImageViewType) -> Self;
        pub fn format(vk::Format) -> Self;
        pub fn components(vk::ComponentMapping) -> Self;
        pub fn subresource_range(vk::ImageSubresourceRange) -> Self;
    }
}

impl<'a> ImageViewCreateInfoBuilder<'a> {
    pub fn image(mut self, image: &'a Image) -> Self {
        self.inner = self.inner.image(unsafe { *image.handle() });
        self
    }
}

// ============================================================================

define_handle! {
    /// An opaque handle to a Vulkan framebuffer object.
    pub struct Framebuffer(vk::Framebuffer);
}

define_delegated_builder! {
    pub struct FramebufferCreateInfoBuilder<'a> {
        inner: vk::FramebufferCreateInfoBuilder<'a>,
    }

    impl FramebufferCreateInfoBuilder {
        pub fn flags(vk::FramebufferCreateFlags) -> Self;
        pub fn attachments(&'a [vk::ImageView]) -> Self;
        pub fn width(u32) -> Self;
        pub fn height(u32) -> Self;
        pub fn layers(u32) -> Self;
    }
}

impl<'a> FramebufferCreateInfoBuilder<'a> {
    pub fn render_pass(mut self, render_pass: &'a RenderPass) -> Self {
        self.inner = self.inner.render_pass(unsafe { *render_pass.handle() });
        self
    }
}

// ============================================================================

define_handle! {
    /// An opaque handle to a Vulkan shader module.
    pub struct ShaderModule(vk::ShaderModule);
}

// ============================================================================

define_handle! {
    /// An opaque handle to a sampler object.
    pub struct Sampler(vk::Sampler);
}

// ============================================================================

define_handle! {
    /// An opaque handle to a descriptor set layout object.
    pub struct DescriptorSetLayout(vk::DescriptorSetLayout);
}

// ============================================================================

define_handle! {
    /// An opaque handle to a pipeline layout object.
    pub struct PipelineLayout(vk::PipelineLayout);
}

// ============================================================================

define_handle! {
    /// An opaque handle to a render pass object.
    pub struct RenderPass(vk::RenderPass);
}

define_delegated_builder! {
    pub struct RenderPassBeginInfoBuilder<'a> {
        inner: vk::RenderPassBeginInfoBuilder<'a>,
    }

    impl RenderPassBeginInfoBuilder {
        pub fn render_area(vk::Rect2D) -> Self;
        pub fn clear_values(&'a [vk::ClearValue]) -> Self;
    }
}

impl<'a> RenderPassBeginInfoBuilder<'a> {
    pub fn render_pass(mut self, render_pass: &'a RenderPass) -> Self {
        self.inner = self.inner.render_pass(unsafe { *render_pass.handle() });
        self
    }

    pub fn framebuffer(mut self, framebuffer: &'a Framebuffer) -> Self {
        self.inner = self.inner.framebuffer(unsafe { *framebuffer.handle() });
        self
    }
}

// ============================================================================

define_handle! {
    /// An opaque handle to a Vulkan pipeline object.
    pub struct Pipeline(vk::Pipeline);
}

define_delegated_builder! {
    pub struct PipelineShaderStageCreateInfoBuilder<'a> {
        inner: vk::PipelineShaderStageCreateInfoBuilder<'a>,
    }

    impl PipelineShaderStageCreateInfoBuilder {
        pub fn stage(vk::ShaderStageFlagBits) -> Self;
        pub fn name(&'a CStr) -> Self;
        pub fn specialization_info(&'a vk::SpecializationInfo) -> Self;
    }
}

impl<'a> PipelineShaderStageCreateInfoBuilder<'a> {
    pub fn module(mut self, module: &'a ShaderModule) -> PipelineShaderStageCreateInfoBuilder<'a> {
        self.inner = self.inner.module(unsafe { *module.handle() });
        self
    }
}

define_delegated_builder! {
    pub struct GraphicsPipelineCreateInfoBuilder<'a> {
        inner: vk::GraphicsPipelineCreateInfoBuilder<'a>,
    }

    impl GraphicsPipelineCreateInfoBuilder {
        pub fn flags(vk::PipelineCreateFlags) -> Self;
        pub fn stages(&'a [vk::PipelineShaderStageCreateInfoBuilder<'a>]) -> Self;
        pub fn vertex_input_state(&'a vk::PipelineVertexInputStateCreateInfo) -> Self;
        pub fn input_assembly_state(&'a vk::PipelineInputAssemblyStateCreateInfo) -> Self;
        pub fn tessellation_state(&'a vk::PipelineTessellationStateCreateInfo) -> Self;
        pub fn viewport_state(&'a vk::PipelineViewportStateCreateInfo) -> Self;
        pub fn rasterization_state(&'a vk::PipelineRasterizationStateCreateInfo) -> Self;
        pub fn multisample_state(&'a vk::PipelineMultisampleStateCreateInfo) -> Self;
        pub fn depth_stencil_state(&'a vk::PipelineDepthStencilStateCreateInfo) -> Self;
        pub fn color_blend_state(&'a vk::PipelineColorBlendStateCreateInfo) -> Self;
        pub fn dynamic_state(&'a vk::PipelineDynamicStateCreateInfo) -> Self;
        pub fn subpass(u32) -> Self;
        pub fn base_pipeline_index(i32) -> Self;
    }
}

impl<'a> GraphicsPipelineCreateInfoBuilder<'a> {
    pub fn layout(mut self, layout: &'a PipelineLayout) -> Self {
        self.inner = self.inner.layout(unsafe { *layout.handle() });
        self
    }

    pub fn render_pass(mut self, render_pass: &'a RenderPass) -> Self {
        self.inner = self.inner.render_pass(unsafe { *render_pass.handle() });
        self
    }

    pub fn base_pipeline_handle(mut self, base_pipeline_handle: &'a Pipeline) -> Self {
        self.inner = self
            .inner
            .base_pipeline_handle(unsafe { *base_pipeline_handle.handle() });
        self
    }
}

// ============================================================================

define_handle! {
    /// An opaque handle to a Vulkan command pool object.
    pub struct CommandPool(vk::CommandPool);
}

// ============================================================================

define_handle! {
    /// An opaque handle to a Vulkan command buffer object.
    pub struct CommandBuffer(vk::CommandBuffer);
}

define_delegated_builder! {
    pub struct CommandBufferAllocateInfoBuilder<'a> {
        inner: vk::CommandBufferAllocateInfoBuilder<'a>,
    }

    impl CommandBufferAllocateInfoBuilder {
        pub fn level(vk::CommandBufferLevel) -> Self;
        pub fn command_buffer_count(u32) -> Self;
    }
}

impl<'a> CommandBufferAllocateInfoBuilder<'a> {
    pub fn command_pool(mut self, command_pool: &'a mut CommandPool) -> Self {
        // Safety: command_pool is externally synchronized via mutable reference
        self.inner = self
            .inner
            .command_pool(unsafe { *command_pool.handle_mut() });
        self
    }
}

// ============================================================================

define_handle! {
    /// An opaque handle to a fence object.
    pub struct Fence(vk::Fence);
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum FenceStatus {
    Unsignaled,
    Signaled,
}

impl FenceStatus {
    pub fn is_signaled(&self) -> bool {
        *self == Self::Signaled
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum FenceWaitStatus {
    TimedOut,
    Signaled,
}

impl FenceWaitStatus {
    pub fn is_signaled(&self) -> bool {
        *self == Self::Signaled
    }
}

// ============================================================================

define_handle! {
    /// An opaque handle to a Vulkan semaphore object.
    pub struct Semaphore(vk::Semaphore);
}

// ============================================================================

define_delegated_builder! {
    pub struct ImageMemoryBarrierBuilder<'a> {
        inner: vk::ImageMemoryBarrierBuilder<'a>,
    }

    impl ImageMemoryBarrierBuilder {
        pub fn src_access_mask(vk::AccessFlags) -> Self;
        pub fn dst_access_mask(vk::AccessFlags) -> Self;
        pub fn old_layout(vk::ImageLayout) -> Self;
        pub fn new_layout(vk::ImageLayout) -> Self;
        pub fn src_queue_family_index(u32) -> Self;
        pub fn dst_queue_family_index(u32) -> Self;
        pub fn subresource_range(vk::ImageSubresourceRange) -> Self;
    }
}

impl<'a> ImageMemoryBarrierBuilder<'a> {
    pub fn image(mut self, image: &Image) -> Self {
        self.inner = self.inner.image(unsafe { *image.handle() });
        self
    }
}

// ============================================================================

define_handle! {
    /// An opaque handle to a Vulkan surface object.
    pub struct SurfaceKHR(vk::SurfaceKHR);
}

// ============================================================================

define_handle! {
    /// An opaque handle to a Vulkan swapchain object.
    pub struct SwapchainKHR(vk::SwapchainKHR);
}

pub struct SwapchainCreateInfo<'surf, 'queues> {
    pub flags: vk::SwapchainCreateFlagsKHR,
    pub surface: &'surf mut SurfaceKHR,
    pub min_image_count: u32,
    pub image_format: vk::Format,
    pub image_color_space: vk::ColorSpaceKHR,
    pub image_extent: vk::Extent2D,
    pub image_array_layers: u32,
    pub image_usage: vk::ImageUsageFlags,
    pub image_sharing_mode: vk::SharingMode,
    pub queue_family_indices: &'queues [u32],
    pub pre_transform: vk::SurfaceTransformFlagBitsKHR,
    pub composite_alpha: vk::CompositeAlphaFlagBitsKHR,
    pub present_mode: vk::PresentModeKHR,
    pub clipped: bool,
    pub old_swapchain: Option<SwapchainKHR>,
}

pub struct AcquiredImage {
    pub status: vk::Result,
    pub index: u32,
}

// ============================================================================

define_handle! {
    pub struct DebugUtilsMessengerEXT(vk::DebugUtilsMessengerEXT);
}

// ============================================================================
