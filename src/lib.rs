#![deny(unsafe_op_in_unsafe_fn)]
#![feature(once_cell)]

mod debug_utils;
mod vks;

use std::{
    ffi::{CStr, CString},
    ops::Deref,
    sync::Arc,
};

use arrayvec::ArrayVec;
use ash::{
    extensions::{ext, khr},
    version::{DeviceV1_0, EntryV1_0, InstanceV1_0},
    vk,
};
use parking_lot::{RwLock, RwLockReadGuard};
use raw_window_handle::RawWindowHandle;
use vks::{VkObject, VkSyncObject};

pub use debug_utils::DebugMessenger;

const LAYER_NAME_VALIDATION: &[u8] = b"VK_LAYER_KHRONOS_validation\0";

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct ApiVersion {
    inner: u32,
}

impl ApiVersion {
    pub const V1_0_0: ApiVersion = ApiVersion {
        inner: vk::make_version(1, 0, 0),
    };

    pub const V1_1_0: ApiVersion = ApiVersion {
        inner: vk::make_version(1, 1, 0),
    };

    pub const V1_2_0: ApiVersion = ApiVersion {
        inner: vk::make_version(1, 2, 0),
    };

    pub const fn major(&self) -> u32 {
        vk::version_major(self.inner)
    }

    pub const fn minor(&self) -> u32 {
        vk::version_minor(self.inner)
    }

    pub const fn patch(&self) -> u32 {
        vk::version_patch(self.inner)
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
    // NOTE: Drop correctness depends on field order!

    // Extensions.
    get_physical_device_properties: ExtensionFn<vk::KhrGetPhysicalDeviceProperties2Fn>,
    external_memory_capabilities: ExtensionFn<vk::KhrExternalMemoryCapabilitiesFn>,
    display: khr::Display,
    debug_utils: ext::DebugUtils,

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
        let instance_extensions = entry
            .enumerate_instance_extension_properties()
            .expect("failed to enumerate instance extension properties");

        let mut extensions = Vec::new();

        extensions.push(khr::Surface::name());
        if cfg!(all(
            unix,
            not(target_os = "android"),
            not(target_os = "macos")
        )) {
            extensions.push(khr::WaylandSurface::name());
            extensions.push(khr::XcbSurface::name());
            extensions.push(khr::XlibSurface::name());
        } else {
            unimplemented!("only tested on linux at the moment, sorry :(");
        }

        extensions.push(ext::DebugUtils::name());
        extensions.push(vk::KhrGetPhysicalDeviceProperties2Fn::name());
        extensions.push(vk::ExtDisplaySurfaceCounterFn::name());
        extensions.push(khr::Display::name());

        if api_version < ApiVersion::V1_1_0 {
            extensions.push(vk::KhrExternalMemoryCapabilitiesFn::name());
        }

        extensions.retain(|&ext| {
            for inst_ext in instance_extensions.iter() {
                let inst_ext_cstr = i8_slice_to_cstr(&inst_ext.extension_name)
                    .expect("extension name is not NUL-terminated");

                if ext == inst_ext_cstr {
                    return true;
                }
            }

            log::warn!("Extension not found: {}", ext.to_string_lossy());
            false
        });

        extensions
    }

    /// Lists the set of required layers.
    fn required_layers() -> Vec<&'static CStr> {
        let entry = vks::entry();
        let instance_layers = entry
            .enumerate_instance_layer_properties()
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

        let driver_api_version = match entry.try_enumerate_instance_version() {
            Ok(Some(version)) => ApiVersion::from_u32(version),
            Ok(None) => ApiVersion::V1_0_0,
            Err(e) => panic!("failed to query instance version: {}", e),
        };

        let app_name = CString::new(app_name.as_ref()).unwrap();
        let app_info = vk::ApplicationInfo::builder()
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

            let create_info = vk::InstanceCreateInfo::builder()
                .flags(vk::InstanceCreateFlags::empty())
                .application_info(&app_info)
                .enabled_layer_names(&layer_ptrs)
                .enabled_extension_names(&ext_ptrs);

            match unsafe { entry.create_instance(&create_info, None) } {
                // Safety: raw handle is not accessible anywhere else.
                Ok(raw) => unsafe { vks::Instance::new(raw) },
                Err(e) => panic!("failed to create raw Vulkan instance: {}", e),
            }
        };

        let instance_extensions = entry
            .enumerate_instance_extension_properties()
            .expect("failed to enumerate instance extension properties");

        let get_physical_device_properties = if driver_api_version >= ApiVersion::V1_1_0 {
            ExtensionFn::Core
        } else {
            if !extensions
                .iter()
                .any(|&ext| ext == vk::KhrGetPhysicalDeviceProperties2Fn::name())
            {
                panic!("missing required extension: VK_KHR_get_physical_device_properties2");
            }

            let load_fn = |name: &CStr| unsafe {
                let raw_handle = instance_handle.handle().raw().handle();
                let addr = entry.get_instance_proc_addr(raw_handle, name.as_ptr());
                std::mem::transmute::<vk::PFN_vkVoidFunction, *const std::ffi::c_void>(addr)
            };

            ExtensionFn::Extension(vk::KhrGetPhysicalDeviceProperties2Fn::load(load_fn))
        };

        let external_memory_capabilities = if driver_api_version >= ApiVersion::V1_1_0 {
            ExtensionFn::Core
        } else {
            if !extensions
                .iter()
                .any(|&ext| ext == vk::KhrExternalMemoryCapabilitiesFn::name())
            {
                panic!("missing required extension: VK_KHR_external_memory_capabilities");
            }

            let load_fn = |name: &CStr| unsafe {
                let raw_handle = instance_handle.handle().raw().handle();
                let addr = entry.get_instance_proc_addr(raw_handle, name.as_ptr());
                std::mem::transmute::<vk::PFN_vkVoidFunction, *const std::ffi::c_void>(addr)
            };

            ExtensionFn::Extension(vk::KhrExternalMemoryCapabilitiesFn::load(load_fn))
        };

        if !extensions.contains(&khr::Display::name()) {
            panic!("missing required extension: VK_KHR_display");
        }

        let display = khr::Display::new(entry, unsafe { instance_handle.handle().raw() });

        if !instance_extensions.iter().any(|props| {
            i8_slice_to_cstr(&props.extension_name).unwrap() == ext::DebugUtils::name()
        }) {
            panic!("missing required extension: VK_EXT_debug_utils")
        }

        let debug_utils = ext::DebugUtils::new(entry, unsafe { instance_handle.handle().raw() });

        Instance {
            inner: Arc::new(RwLock::new(InstanceInner {
                get_physical_device_properties,
                external_memory_capabilities,
                display,
                debug_utils,
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
    pub fn create_surface(&self, window: RawWindowHandle) -> Surface {
        let read_lock = self.inner.read();
        let instance = read_lock.handle.handle();

        let raw_surface = match window {
            RawWindowHandle::Xlib(xlib) => {
                let xlib_ext = khr::XlibSurface::new(vks::entry(), unsafe { instance.raw() });
                let create_info = vk::XlibSurfaceCreateInfoKHR::builder()
                    .flags(vk::XlibSurfaceCreateFlagsKHR::empty())
                    .dpy(xlib.display as *mut _)
                    .window(xlib.window);

                unsafe {
                    vks::Surface::new(
                        xlib_ext
                            .create_xlib_surface(&create_info, None)
                            .expect("failed to create Xlib window surface"),
                    )
                }
            }
            RawWindowHandle::Xcb(xcb) => {
                let xcb_ext = khr::XcbSurface::new(vks::entry(), unsafe { instance.raw() });
                let create_info = vk::XcbSurfaceCreateInfoKHR::builder()
                    .flags(vk::XcbSurfaceCreateFlagsKHR::empty())
                    .window(xcb.window)
                    .connection(xcb.connection);

                unsafe {
                    vks::Surface::new(
                        xcb_ext
                            .create_xcb_surface(&create_info, None)
                            .expect("failed to create XCB window surface"),
                    )
                }
            }
            RawWindowHandle::Wayland(_) => todo!(),
            _ => panic!("unrecognized window handle"),
        };

        drop(read_lock);

        Surface {
            inner: Arc::new(RwLock::new(SurfaceInner {
                instance: self.clone(),
                raw: raw_surface,
            })),
        }
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

    pub fn enumerate_physical_devices(&self, surface: &Surface) -> Vec<PhysicalDevice> {
        let read_lock = self.inner.read();

        let devices = match unsafe { read_lock.handle.handle().raw().enumerate_physical_devices() }
        {
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
                unsafe {
                    PhysicalDevice::new(
                        self.clone(),
                        vks::PhysicalDevice::new(raw),
                        &surface.inner.read().raw,
                    )
                }
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
        surface: &vks::Surface,
    ) -> PhysicalDevice {
        let read_lock = instance.inner.read();

        let queue_families = unsafe {
            read_lock
                .handle
                .get_physical_device_queue_family_properties(phys_device.handle())
        };

        enum QueueSelection {
            Dedicated(u32),
            General(u32),
        }

        let mut graphics_queue = None;
        let mut transfer_queue = None;
        let mut present_queue = None;

        let surface_ext =
            unsafe { khr::Surface::new(vks::entry(), &*read_lock.handle.handle().raw()) };

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
                surface_ext
                    .get_physical_device_surface_support(
                        *phys_device.handle().raw(),
                        index as u32,
                        *surface.handle().raw(),
                    )
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
                .get_physical_device_properties(self.inner.raw.handle())
        }
    }

    pub fn features(&self) -> vk::PhysicalDeviceFeatures {
        // Safety: No external synchronization requirement.
        unsafe {
            self.inner
                .instance
                .read_inner()
                .handle
                .get_physical_device_features(self.inner.raw.handle())
        }
    }

    pub fn create_device(&self) -> Device {
        let mut unique_queue_families = UniqueQueueFamilies::default();
        let graphics = unique_queue_families.get_or_insert(self.inner.graphics_queue_family) as u8;
        let transfer = unique_queue_families.get_or_insert(self.inner.transfer_queue_family) as u8;
        let present = unique_queue_families.get_or_insert(self.inner.present_queue_family) as u8;

        let phys_device_features = vk::PhysicalDeviceFeatures::builder();
        let enabled_layer_names = &[LAYER_NAME_VALIDATION.as_ptr() as *const i8];
        let device_create_info = vk::DeviceCreateInfo::builder()
            .flags(vk::DeviceCreateFlags::empty())
            .queue_create_infos(unique_queue_families.infos())
            .enabled_layer_names(enabled_layer_names)
            .enabled_extension_names(&[])
            .enabled_features(&phys_device_features);

        // Safety: no external synchronization requirement.
        let raw_device = unsafe {
            self.inner
                .instance
                .read_inner()
                .handle
                .create_device(self.inner.raw.handle(), &device_create_info)
                .expect("failed to create logical device")
        };

        log::info!("Successfully created logical device.");

        let inner = Arc::new(RwLock::new(DeviceInner {
            raw: raw_device,
            phys_device: self.clone(),
            instance: self.inner.instance.clone(),
        }));

        let inner_lock = inner.read();

        let mut queues = ArrayVec::new();
        for info in unique_queue_families.infos() {
            let raw_queue = unsafe {
                vks::Queue::new(
                    inner_lock
                        .raw
                        .handle()
                        .raw()
                        .get_device_queue(info.queue_family_index, 0),
                )
            };

            let queue = Queue {
                inner: Arc::new(RwLock::new(QueueInner {
                    device: inner.clone(),
                    raw: raw_queue,
                })),
            };

            queues.push(queue);
        }

        drop(inner_lock);

        let queues = DeviceQueues {
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

#[derive(Default)]
struct UniqueQueueFamilies {
    family_ids: ArrayVec<u32, MAX_DEVICE_QUEUES>,
    infos: ArrayVec<vk::DeviceQueueCreateInfo, MAX_DEVICE_QUEUES>,
}

impl UniqueQueueFamilies {
    fn get_or_insert(&mut self, family_id: u32) -> usize {
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
                    vk::DeviceQueueCreateInfo::builder()
                        .flags(vk::DeviceQueueCreateFlags::empty())
                        .queue_family_index(family_id)
                        .queue_priorities(&[1.0])
                        .build(),
                );
                i
            }
        }
    }

    fn infos(&self) -> &[vk::DeviceQueueCreateInfo] {
        self.infos.as_slice()
    }
}

#[derive(Clone)]
struct DeviceQueues {
    // NOTE: sensitive drop order.
    graphics: u8,
    transfer: u8,
    present: u8,

    queues: ArrayVec<Queue, MAX_DEVICE_QUEUES>,
}

impl DeviceQueues {
    pub fn graphics(&self) -> Queue {
        self.queues[self.graphics as usize].clone()
    }

    pub fn transfer(&self) -> Queue {
        self.queues[self.transfer as usize].clone()
    }

    pub fn present(&self) -> Queue {
        self.queues[self.present as usize].clone()
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
    pub fn graphics_queue(&self) -> Queue {
        self.queues.graphics()
    }

    pub fn transfer_queue(&self) -> Queue {
        self.queues.transfer()
    }

    pub fn present_queue(&self) -> Queue {
        self.queues.present()
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
    raw: vks::Surface,
}

impl Drop for SurfaceInner {
    fn drop(&mut self) {
        let instance = self.instance.inner.read();

        // Safety:
        // - Swapchain objects have handles to this surface, so it will not be
        //   dropped before they are.
        // - Host access synchronization guaranteed by mutable receiver.
        unsafe {
            let surface_ext = khr::Surface::new(vks::entry(), &*instance.handle.handle().raw());
            surface_ext.destroy_surface(*self.raw.handle_mut().raw_mut(), None);
        }
    }
}

pub struct Surface {
    inner: Arc<RwLock<SurfaceInner>>,
}

struct SwapchainInner {
    surface: Surface,
}

struct Swapchain {
    inner: Arc<RwLock<SwapchainInner>>,
}
