#![deny(unsafe_op_in_unsafe_fn)]
#![feature(once_cell)]

mod debug_utils;
mod display;
pub mod vks;

use std::{
    ffi::{CStr, CString},
    sync::Arc,
};

use arrayvec::ArrayVec;
use erupt::vk;
use parking_lot::{RwLock, RwLockReadGuard};
use raw_window_handle::RawWindowHandle;
use vks::VkObject;

pub use debug_utils::DebugMessenger;
pub use display::Display;

const LAYER_NAME_VALIDATION: &[u8] = b"VK_LAYER_KHRONOS_validation\0";

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct ApiVersion {
    inner: u32,
}

impl ApiVersion {
    pub const V1_0_0: ApiVersion = ApiVersion {
        inner: vk::make_api_version(0, 1, 0, 0),
    };

    pub const V1_1_0: ApiVersion = ApiVersion {
        inner: vk::make_api_version(0, 1, 1, 0),
    };

    pub const V1_2_0: ApiVersion = ApiVersion {
        inner: vk::make_api_version(0, 1, 2, 0),
    };

    pub const fn major(&self) -> u32 {
        vk::api_version_major(self.inner)
    }

    pub const fn minor(&self) -> u32 {
        vk::api_version_minor(self.inner)
    }

    pub const fn patch(&self) -> u32 {
        vk::api_version_patch(self.inner)
    }

    pub const fn from_u32(version: u32) -> ApiVersion {
        ApiVersion { inner: version }
    }

    pub const fn as_u32(&self) -> u32 {
        self.inner
    }
}

fn i8_slice_to_cstr(slice: &[i8]) -> Option<&CStr> {
    if slice.contains(&0) {
        // Safety: NUL byte is known to exist in bounds of the slice.
        Some(unsafe { CStr::from_ptr(slice.as_ptr()) })
    } else {
        None
    }
}

/// A type representing functionality which may be either an extension or part
/// of the core Vulkan API, depending on version.
pub enum ExtensionFn<T> {
    Core,
    Extension(T),
}

pub(crate) struct InstanceInner {
    // Underlying instance. Destroys the instance when dropped.
    handle: vks::Instance,
}

impl InstanceInner {
    pub fn handle(&self) -> &vks::Instance {
        &self.handle
    }
}

#[derive(Clone)]
pub struct Instance {
    inner: Arc<RwLock<InstanceInner>>,
}

impl Instance {
    /// Lists the set of required extensions for the current platform.
    fn required_extensions(api_version: ApiVersion) -> Vec<&'static CStr> {
        let entry = vks::entry();
        let instance_extensions =
            unsafe { entry.enumerate_instance_extension_properties(None, None) }
                .expect("failed to enumerate instance extension properties");

        let mut extensions = Vec::new();

        extensions.push(vk::KHR_SURFACE_EXTENSION_NAME);
        if cfg!(all(
            unix,
            not(target_os = "android"),
            not(target_os = "macos")
        )) {
            extensions.push(vk::KHR_WAYLAND_SURFACE_EXTENSION_NAME);
            extensions.push(vk::KHR_XCB_SURFACE_EXTENSION_NAME);
            extensions.push(vk::KHR_XLIB_SURFACE_EXTENSION_NAME);
        } else {
            unimplemented!("only tested on linux at the moment, sorry :(");
        }

        extensions.push(vk::EXT_DEBUG_UTILS_EXTENSION_NAME);
        extensions.push(vk::EXT_DISPLAY_SURFACE_COUNTER_EXTENSION_NAME);
        extensions.push(vk::KHR_DISPLAY_EXTENSION_NAME);

        if api_version < ApiVersion::V1_1_0 {
            extensions.push(vk::KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
            extensions.push(vk::KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);
        }

        let mut extension_cstrs = extensions
            .into_iter()
            .map(|ptr| unsafe { CStr::from_ptr(ptr) })
            .collect::<Vec<_>>();

        extension_cstrs.retain(|&wanted| {
            for inst_ext in instance_extensions.iter() {
                let inst_ext_cstr = i8_slice_to_cstr(&inst_ext.extension_name)
                    .expect("extension name is not NUL-terminated");

                if wanted == inst_ext_cstr {
                    return true;
                }
            }

            log::warn!("Extension not found: {}", wanted.to_string_lossy());
            false
        });

        extension_cstrs
    }

    /// Lists the set of required layers.
    fn required_layers() -> Vec<&'static CStr> {
        let entry = vks::entry();
        let instance_layers = unsafe { entry.enumerate_instance_layer_properties(None) }
            .expect("failed to enumerate instance layer properties");

        let mut layers = Vec::new();
        if cfg!(debug_assertions) {
            layers.push(CStr::from_bytes_with_nul(LAYER_NAME_VALIDATION).unwrap());
        }

        layers.retain(|&layer| {
            for inst_layer in instance_layers.iter() {
                let inst_layer_cstr = i8_slice_to_cstr(&inst_layer.layer_name)
                    .expect("layer name is not NUL-terminated");

                if layer == inst_layer_cstr {
                    return true;
                }
            }

            log::warn!("Layer not found: {}", layer.to_string_lossy());
            false
        });

        layers
    }

    pub fn create<S>(app_name: S, app_version: u32) -> Instance
    where
        S: AsRef<str>,
    {
        let entry = vks::entry();

        let driver_api_version = match unsafe { entry.enumerate_instance_version() }.result() {
            Ok(version) => ApiVersion::from_u32(version),
            Err(e) => panic!("failed to query instance version: {}", e),
        };

        let app_name = CString::new(app_name.as_ref()).unwrap();
        let app_info = vk::ApplicationInfoBuilder::new()
            .application_name(app_name.as_c_str())
            .application_version(app_version)
            .engine_name(CStr::from_bytes_with_nul(b"reify\0").unwrap())
            .engine_version(1)
            .api_version({ std::cmp::min(driver_api_version, ApiVersion::V1_2_0) }.as_u32());

        let extensions = Self::required_extensions(driver_api_version);
        let layers = Self::required_layers();

        let instance_handle = {
            let ext_ptrs = extensions.iter().map(|&s| s.as_ptr()).collect::<Vec<_>>();
            let layer_ptrs = layers.iter().map(|&s| s.as_ptr()).collect::<Vec<_>>();

            let create_info = vk::InstanceCreateInfoBuilder::new()
                .flags(vk::InstanceCreateFlags::empty())
                .application_info(&app_info)
                .enabled_layer_names(&layer_ptrs)
                .enabled_extension_names(&ext_ptrs);

            vks::Instance::create(&create_info).expect("failed to create raw Vulkan instance: {}")
        };

        let instance_extensions =
            unsafe { entry.enumerate_instance_extension_properties(None, None) }
                .expect("failed to enumerate instance extension properties");

        Instance {
            inner: Arc::new(RwLock::new(InstanceInner {
                handle: instance_handle,
            })),
        }
    }

    pub(crate) fn read_inner(&self) -> RwLockReadGuard<InstanceInner> {
        self.inner.read()
    }

    /// Initializes a debug messenger for this instance.
    ///
    /// # Safety
    ///
    /// The created debug messenger must be destroyed before this instance is
    /// dropped.
    // TODO: safe to create multiple debug messengers?
    pub fn create_debug_messenger(&self) -> DebugMessenger {
        DebugMessenger::new(self.clone())
    }

    #[cfg(any(
        target_os = "linux",
        target_os = "dragonfly",
        target_os = "freebsd",
        target_os = "netbsd",
        target_os = "openbsd",
    ))]
    pub fn create_surface(&self, window: RawWindowHandle) -> vks::SurfaceKHR {
        let read_lock = self.inner.read();

        let raw_surface = match window {
            RawWindowHandle::Xlib(xlib) => {
                let create_info = vk::XlibSurfaceCreateInfoKHRBuilder::new()
                    .flags(vk::XlibSurfaceCreateFlagsKHR::empty())
                    .dpy(xlib.display as *mut _)
                    .window(xlib.window);

                unsafe {
                    read_lock
                        .handle
                        .handle()
                        .create_xlib_surface_khr(&create_info, None)
                        .expect("failed to create Xlib window surface")
                }
            }
            RawWindowHandle::Xcb(xcb) => {
                let create_info = vk::XcbSurfaceCreateInfoKHRBuilder::new()
                    .flags(vk::XcbSurfaceCreateFlagsKHR::empty())
                    .window(xcb.window)
                    .connection(xcb.connection);

                unsafe {
                    read_lock
                        .handle
                        .handle()
                        .create_xcb_surface_khr(&create_info, None)
                        .expect("failed to create XCB window surface")
                }
            }
            RawWindowHandle::Wayland(_) => todo!(),
            _ => panic!("unrecognized window handle"),
        };

        drop(read_lock);

        unsafe { vks::SurfaceKHR::new(raw_surface) }
    }

    #[cfg(not(any(
        target_os = "linux",
        target_os = "dragonfly",
        target_os = "freebsd",
        target_os = "netbsd",
        target_os = "openbsd",
    )))]
    pub fn create_surface(&self, window: RawWindowHandle) -> Surface {
        compile_error!("Unsupported platform (only linux is supported).");
    }

    pub fn enumerate_physical_devices(&self, surface: &vks::SurfaceKHR) -> Vec<PhysicalDevice> {
        let read_lock = self.inner.read();

        let devices = match read_lock.handle.enumerate_physical_devices() {
            Ok(devices) => devices,
            Err(e) => {
                log::error!("failed to enumerate physical devices: {}", e);
                Vec::new()
            }
        };

        drop(read_lock);

        devices
            .into_iter()
            .map(|raw| {
                // Safety: Physical device handle comes from the correct instance.
                unsafe { PhysicalDevice::new(self.clone(), raw, surface) }
            })
            .collect()
    }
}

pub struct PhysicalDeviceInner {
    instance: Instance,
    raw: vks::PhysicalDevice,
    queue_families: Vec<vk::QueueFamilyProperties>,
    graphics_queue_family: u32,
    transfer_queue_family: u32,
    present_queue_family: u32,
}

fn is_dedicated_transfer(props: &vk::QueueFamilyProperties) -> bool {
    props.queue_flags.contains(vk::QueueFlags::TRANSFER)
        && !props.queue_flags.contains(vk::QueueFlags::GRAPHICS)
        && !props.queue_flags.contains(vk::QueueFlags::COMPUTE)
}

fn is_dedicated_graphics(props: &vk::QueueFamilyProperties) -> bool {
    props.queue_flags.contains(vk::QueueFlags::GRAPHICS)
        && !props.queue_flags.contains(vk::QueueFlags::COMPUTE)
}

#[derive(Clone)]
pub struct PhysicalDevice {
    // Physical device handles do not need to be externally synchronized.
    inner: Arc<PhysicalDeviceInner>,
}

impl PhysicalDevice {
    /// Create a new physical device with the provided instance and raw physical
    /// device handle.
    ///
    /// # Safety
    ///
    /// The raw physical device handle must have been provided by `instance`.
    unsafe fn new(
        instance: Instance,
        phys_device: vks::PhysicalDevice,
        surface: &vks::SurfaceKHR,
    ) -> PhysicalDevice {
        let read_lock = instance.inner.read();

        let queue_families = unsafe {
            read_lock
                .handle
                .get_physical_device_queue_family_properties(&phys_device)
        };

        enum QueueSelection {
            Dedicated(u32),
            General(u32),
        }

        let mut graphics_queue = None;
        let mut transfer_queue = None;
        let mut present_queue = None;

        for (index, qf) in queue_families.iter().enumerate() {
            match graphics_queue {
                // Already have a dedicated queue.
                Some(QueueSelection::Dedicated(_)) => (),

                Some(QueueSelection::General(_)) => {
                    if is_dedicated_graphics(qf) {
                        graphics_queue = Some(QueueSelection::Dedicated(index as u32));
                    }
                }

                None => {
                    if is_dedicated_graphics(qf) {
                        graphics_queue = Some(QueueSelection::Dedicated(index as u32));
                    } else if qf.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                        graphics_queue = Some(QueueSelection::General(index as u32));
                    }
                }
            }

            match transfer_queue {
                // Already have a dedicated queue.
                Some(QueueSelection::Dedicated(_)) => (),

                Some(QueueSelection::General(_)) => {
                    if is_dedicated_transfer(qf) {
                        transfer_queue = Some(QueueSelection::Dedicated(index as u32));
                    }
                }

                None => {
                    if is_dedicated_transfer(qf) {
                        transfer_queue = Some(QueueSelection::Dedicated(index as u32));
                    } else if qf.queue_flags.contains(vk::QueueFlags::TRANSFER) {
                        transfer_queue = Some(QueueSelection::General(index as u32));
                    }
                }
            }

            // Safety:
            // - Queue family index provided by physical device.
            // - No external synchronization requirement.
            if unsafe {
                read_lock
                    .handle()
                    .get_physical_device_surface_support_khr(&phys_device, index as u32, &surface)
                    .expect(&format!(
                        "failed to query queue family {} for surface support",
                        index
                    ))
            } {
                present_queue = Some(index);
            }
        }

        drop(read_lock);

        let graphics_queue_family = match graphics_queue {
            Some(QueueSelection::Dedicated(d)) => {
                log::info!("Using dedicated graphics queue family (index = {})", d);
                d
            }
            Some(QueueSelection::General(g)) => {
                log::info!(
                    "Using general-purpose graphics queue family (index = {})",
                    g
                );
                g
            }
            None => panic!("No queue families support graphics operations."),
        };

        let transfer_queue_family = match transfer_queue {
            Some(QueueSelection::Dedicated(d)) => {
                log::info!("Using dedicated transfer queue family (index = {})", d);
                d
            }
            Some(QueueSelection::General(g)) => {
                log::info!(
                    "Using general-purpose transfer queue family (index = {})",
                    g
                );
                g
            }
            None => panic!("No queue families support transfer operations."),
        };

        let present_queue_family = match present_queue {
            Some(p) => {
                log::info!("Using queue family {} for presentation", p);
                p as u32
            }
            None => panic!("No queue families support presenting to the window surface"),
        };

        PhysicalDevice {
            inner: Arc::new(PhysicalDeviceInner {
                instance,
                raw: phys_device,
                queue_families,
                graphics_queue_family,
                transfer_queue_family,
                present_queue_family,
            }),
        }
    }

    pub fn properties(&self) -> vk::PhysicalDeviceProperties {
        // Safety: No external synchronization requirement.
        unsafe {
            self.inner
                .instance
                .read_inner()
                .handle
                .get_physical_device_properties(&self.inner.raw)
        }
    }

    pub fn features(&self) -> vk::PhysicalDeviceFeatures {
        // Safety: No external synchronization requirement.
        unsafe {
            self.inner
                .instance
                .read_inner()
                .handle
                .get_physical_device_features(&self.inner.raw)
        }
    }

    pub fn create_device(&self) -> Device {
        let mut unique_queue_families = UniqueQueueFamilies::default();
        let graphics = unique_queue_families
            .get_or_insert(self.inner.graphics_queue_family, SINGLE_QUEUE_PRIORITY)
            as u8;
        let transfer = unique_queue_families
            .get_or_insert(self.inner.transfer_queue_family, SINGLE_QUEUE_PRIORITY)
            as u8;
        let present = unique_queue_families
            .get_or_insert(self.inner.present_queue_family, SINGLE_QUEUE_PRIORITY)
            as u8;

        let phys_device_features = vk::PhysicalDeviceFeaturesBuilder::new();
        let enabled_layer_names = &[LAYER_NAME_VALIDATION.as_ptr() as *const i8];
        let device_create_info = vk::DeviceCreateInfoBuilder::new()
            .flags(vk::DeviceCreateFlags::empty())
            .queue_create_infos(unique_queue_families.infos())
            .enabled_layer_names(enabled_layer_names)
            // TODO: need to check ahead of time that this is available
            .enabled_extension_names(&[vk::KHR_SWAPCHAIN_EXTENSION_NAME])
            .enabled_features(&phys_device_features);

        // Safety: no external synchronization requirement.
        let raw_device = unsafe {
            self.inner
                .instance
                .read_inner()
                .handle
                .create_device(&self.inner.raw, &device_create_info)
                .expect("failed to create logical device")
        };

        log::info!("Successfully created logical device.");

        let inner = Arc::new(RwLock::new(DeviceInner {
            raw: raw_device,
            phys_device: self.clone(),
            instance: self.inner.instance.clone(),
        }));

        let inner_read = inner.read();

        let mut family_ids = ArrayVec::new();
        let mut queues = ArrayVec::new();
        for (info, id) in unique_queue_families
            .infos()
            .into_iter()
            .zip(unique_queue_families.family_ids.iter())
        {
            let raw_queue = unsafe {
                vks::Queue::new(
                    inner_read
                        .raw
                        .handle()
                        .get_device_queue(info.queue_family_index, 0),
                )
            };

            let queue = Queue {
                inner: Arc::new(RwLock::new(QueueInner {
                    device: inner.clone(),
                    raw: raw_queue,
                })),
            };

            family_ids.push(*id);
            queues.push(queue);
        }

        drop(inner_read);

        let queues = DeviceQueues {
            family_ids,
            queues,
            graphics,
            transfer,
            present,
        };

        Device { inner, queues }
    }
}

// Graphics, compute, transfer, present
const MAX_DEVICE_QUEUES: usize = 4;

static SINGLE_QUEUE_PRIORITY: &'static [f32] = &[1.0];

#[derive(Default)]
struct UniqueQueueFamilies<'a> {
    family_ids: ArrayVec<u32, MAX_DEVICE_QUEUES>,
    infos: ArrayVec<vk::DeviceQueueCreateInfoBuilder<'a>, MAX_DEVICE_QUEUES>,
}

impl<'a> UniqueQueueFamilies<'a> {
    fn get_or_insert(&mut self, family_id: u32, priorities: &'a [f32]) -> usize {
        match self
            .family_ids
            .iter()
            .enumerate()
            .find(|(_, &qf)| qf == family_id)
        {
            Some((i, _)) => i,
            None => {
                let i = self.family_ids.len();
                self.family_ids.push(family_id);
                self.infos.push(
                    vk::DeviceQueueCreateInfoBuilder::new()
                        .flags(vk::DeviceQueueCreateFlags::empty())
                        .queue_family_index(family_id)
                        .queue_priorities(priorities),
                );
                i
            }
        }
    }

    fn infos(&self) -> &[vk::DeviceQueueCreateInfoBuilder<'a>] {
        self.infos.as_slice()
    }
}

#[derive(Clone)]
struct DeviceQueues {
    graphics: u8,
    transfer: u8,
    present: u8,

    family_ids: ArrayVec<u32, MAX_DEVICE_QUEUES>,
    queues: ArrayVec<Queue, MAX_DEVICE_QUEUES>,
}

impl DeviceQueues {
    pub fn graphics(&self) -> Queue {
        self.queues[self.graphics as usize].clone()
    }

    pub fn graphics_family_id(&self) -> u32 {
        self.family_ids[self.graphics as usize]
    }

    pub fn transfer(&self) -> Queue {
        self.queues[self.transfer as usize].clone()
    }

    pub fn transfer_family_id(&self) -> u32 {
        self.family_ids[self.transfer as usize]
    }

    pub fn present(&self) -> Queue {
        self.queues[self.present as usize].clone()
    }

    pub fn present_family_id(&self) -> u32 {
        self.family_ids[self.present as usize]
    }
}

pub struct DeviceInner {
    // NOTE: sensitive drop order.
    raw: vks::Device,
    phys_device: PhysicalDevice,
    instance: Instance,
}

#[derive(Clone)]
pub struct Device {
    // NOTE: sensitive drop order.
    queues: DeviceQueues,
    inner: Arc<RwLock<DeviceInner>>,
}

impl Device {
    pub fn read_inner(&self) -> RwLockReadGuard<'_, DeviceInner> {
        self.inner.read()
    }

    pub fn graphics_queue(&self) -> Queue {
        self.queues.graphics()
    }

    pub fn graphics_family_id(&self) -> u32 {
        self.queues.graphics_family_id()
    }

    pub fn transfer_queue(&self) -> Queue {
        self.queues.transfer()
    }

    pub fn transfer_family_id(&self) -> u32 {
        self.queues.transfer_family_id()
    }

    pub fn present_queue(&self) -> Queue {
        self.queues.present()
    }

    pub fn present_family_id(&self) -> u32 {
        self.queues.present_family_id()
    }

    // Safety: device and surface must be from same instance
    pub unsafe fn create_display(
        &self,
        surface: vks::SurfaceKHR,
        phys_window_extent: vk::Extent2D,
    ) -> Display {
        unsafe { Display::create(self, surface, phys_window_extent) }
    }

    unsafe fn create_render_pass(&self, target: &Display) -> vks::RenderPass {
        let color_attachment = vk::AttachmentDescriptionBuilder::new()
            .format(target.info().surface_format.format)
            .samples(vk::SampleCountFlagBits::_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);

        let attachment_reference = vk::AttachmentReferenceBuilder::new()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

        let color_attachments = &[attachment_reference];
        let subpass = vk::SubpassDescriptionBuilder::new()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(color_attachments);

        let attachments = &[color_attachment];
        let subpasses = &[subpass];
        let render_pass_info = vk::RenderPassCreateInfoBuilder::new()
            .attachments(attachments)
            .subpasses(subpasses);

        let render_pass = unsafe { self.inner.read().raw.create_render_pass(&render_pass_info) }
            .expect("failed to create render pass");

        render_pass
    }

    pub unsafe fn create_pipeline(
        &self,
        vert_spv: &[u32],
        frag_spv: &[u32],
        target: &Display,
    ) -> Pipeline {
        let device_read = self.inner.read();

        let vert_module = unsafe {
            device_read.raw.create_shader_module(
                &vk::ShaderModuleCreateInfoBuilder::new()
                    .flags(vk::ShaderModuleCreateFlags::empty())
                    .code(vert_spv),
            )
        }
        .expect("failed to create vertex shader module");

        let frag_module = unsafe {
            device_read.raw.create_shader_module(
                &vk::ShaderModuleCreateInfoBuilder::new()
                    .flags(vk::ShaderModuleCreateFlags::empty())
                    .code(frag_spv),
            )
        }
        .expect("failed to create fragment shader module");

        let vert_stage = vks::PipelineShaderStageCreateInfoBuilder::new()
            .stage(vk::ShaderStageFlagBits::VERTEX)
            .name(&CStr::from_bytes_with_nul(b"main\0").unwrap())
            .module(&vert_module);

        let frag_stage = vks::PipelineShaderStageCreateInfoBuilder::new()
            .stage(vk::ShaderStageFlagBits::FRAGMENT)
            .name(&CStr::from_bytes_with_nul(b"main\0").unwrap())
            .module(&frag_module);

        let vertex_input = vk::PipelineVertexInputStateCreateInfoBuilder::new()
            .vertex_binding_descriptions(&[])
            .vertex_attribute_descriptions(&[]);

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfoBuilder::new()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);

        let viewport = vk::ViewportBuilder::new()
            .x(0.0)
            .y(0.0)
            .width(target.info().image_extent.width as f32)
            .height(target.info().image_extent.height as f32)
            .min_depth(0.0)
            .max_depth(1.0);

        let scissor = vk::Rect2DBuilder::new()
            .offset(vk::Offset2D { x: 0, y: 0 })
            .extent(target.info().image_extent);

        let viewports = &[viewport];
        let scissors = &[scissor];
        let viewport_state = vk::PipelineViewportStateCreateInfoBuilder::new()
            .viewports(viewports)
            .scissors(scissors);

        let rasterization_state = vk::PipelineRasterizationStateCreateInfoBuilder::new()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(vk::CullModeFlags::BACK)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .depth_bias_enable(false);

        let multisample_state = vk::PipelineMultisampleStateCreateInfoBuilder::new()
            .sample_shading_enable(false)
            .rasterization_samples(vk::SampleCountFlagBits::_1)
            .min_sample_shading(1.0)
            .sample_mask(&[])
            .alpha_to_coverage_enable(false)
            .alpha_to_one_enable(false);

        let color_blend_attachment = vk::PipelineColorBlendAttachmentStateBuilder::new()
            .color_write_mask(
                vk::ColorComponentFlags::R
                    | vk::ColorComponentFlags::G
                    | vk::ColorComponentFlags::B
                    | vk::ColorComponentFlags::A,
            )
            .blend_enable(false);

        let attachments = &[color_blend_attachment];
        let color_blend = vk::PipelineColorBlendStateCreateInfoBuilder::new()
            .attachments(attachments)
            .logic_op_enable(false);

        let pipeline_layout_info = vk::PipelineLayoutCreateInfoBuilder::new()
            .set_layouts(&[])
            .push_constant_ranges(&[]);

        let pipeline_layout = unsafe {
            self.inner
                .read()
                .raw
                .create_pipeline_layout(&pipeline_layout_info)
        }
        .expect("failed to create pipeline layout");

        let render_pass = unsafe { self.create_render_pass(target) };

        let pipeline = unsafe {
            // Safety: copied handles do not outlive the block.
            let stages = &[vert_stage.into_inner(), frag_stage.into_inner()];

            let pipeline_info = vks::GraphicsPipelineCreateInfoBuilder::new()
                .stages(stages)
                .vertex_input_state(&vertex_input)
                .input_assembly_state(&input_assembly)
                .viewport_state(&viewport_state)
                .rasterization_state(&rasterization_state)
                .multisample_state(&multisample_state)
                .color_blend_state(&color_blend)
                .layout(&pipeline_layout)
                .render_pass(&render_pass);

            // Safety: copied handles do not outlive the block.
            let pipeline_infos = &[pipeline_info.into_inner()];
            device_read
                .raw
                .create_graphics_pipelines(pipeline_infos)
                .expect("failed to create pipeline")
                .into_iter()
                .next()
                .unwrap()
        };

        Pipeline {
            inner: Arc::new(RwLock::new(PipelineInner {
                pipeline: Some(pipeline),
                layout: Some(pipeline_layout),
                pass: Some(render_pass),
                device: self.clone(),
            })),
        }
    }
}

struct QueueInner {
    // NOTE: sensitive drop order

    // Queue does not need to be manually destroyed, but it must not be used
    // after the underlying device is destroyed.
    raw: vks::Queue,

    // Hold a reference to the `DeviceInner` (rather than the `Device`) to avoid
    // circular `Arc`s.
    device: Arc<RwLock<DeviceInner>>,
}

#[derive(Clone)]
pub struct Queue {
    inner: Arc<RwLock<QueueInner>>,
}

struct SurfaceInner {
    instance: Instance,
    raw: Option<vks::SurfaceKHR>,
}

impl Drop for SurfaceInner {
    fn drop(&mut self) {
        let instance = self.instance.inner.read();

        match self.raw.take() {
            Some(raw) => unsafe {
                // Safety:
                // - Swapchain objects have handles to this surface, so it will not be
                //   dropped before they are.
                // - Host access synchronization guaranteed by mutable receiver.
                instance.handle.destroy_surface(raw);
            },
            None => {
                log::warn!("Possible leak of Surface");
            }
        }
    }
}

#[derive(Clone)]
pub struct Surface {
    inner: Arc<RwLock<SurfaceInner>>,
}

pub struct SwapchainCreateInfo {
    pub flags: vk::SwapchainCreateFlagsKHR,
    pub surface: Surface,
    pub min_image_count: u32,
    pub image_format: vk::Format,
    pub image_color_space: vk::ColorSpaceKHR,
    pub image_extent: vk::Extent2D,
    pub image_array_layers: u32,
    pub image_usage: vk::ImageUsageFlags,
    pub pre_transform: vk::SurfaceTransformFlagsKHR,
    pub composite_alpha: vk::CompositeAlphaFlagsKHR,
    pub present_mode: vk::PresentModeKHR,
    pub clipped: bool,
}

struct SwapchainInner {
    // NOTE: sensitive drop order.
    raw: Option<vks::SwapchainKHR>,
    device: Device,
    surface: Surface,
}

impl Drop for SwapchainInner {
    fn drop(&mut self) {
        if let Some(swapchain) = self.raw.take() {
            let device_read = self.device.inner.read();

            unsafe { device_read.raw.destroy_swapchain(swapchain) }
        }
    }
}

#[derive(Clone)]
pub struct Swapchain {
    inner: Arc<RwLock<SwapchainInner>>,
}

pub struct PipelineInner {
    pipeline: Option<vks::Pipeline>,
    layout: Option<vks::PipelineLayout>,
    pass: Option<vks::RenderPass>,
    device: Device,
}

impl Drop for PipelineInner {
    fn drop(&mut self) {
        let device_read = self.device.inner.read();

        if let Some(pipeline) = self.pipeline.take() {
            unsafe {
                device_read.raw.destroy_pipeline(pipeline);
            }
        }

        if let Some(layout) = self.layout.take() {
            unsafe {
                device_read.raw.destroy_pipeline_layout(layout);
            }
        }

        if let Some(pass) = self.pass.take() {
            unsafe {
                device_read.raw.destroy_render_pass(pass);
            }
        }
    }
}

impl PipelineInner {
    pub fn render_pass(&self) -> &vks::RenderPass {
        self.pass.as_ref().unwrap()
    }
}

pub struct Pipeline {
    inner: Arc<RwLock<PipelineInner>>,
}

impl Pipeline {
    pub fn read_inner(&self) -> RwLockReadGuard<'_, PipelineInner> {
        self.inner.read()
    }
}
