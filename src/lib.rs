#![deny(unsafe_op_in_unsafe_fn)]

mod debug_utils;

use std::{
    ffi::{CStr, CString},
    ops::Deref,
    sync::Arc,
};

use arrayvec::ArrayVec;
use ash::{
    extensions::{ext, khr},
    version::{DeviceV1_0, EntryV1_0, InstanceV1_0, InstanceV1_1},
    vk,
};
use parking_lot::{
    MappedRwLockReadGuard, MappedRwLockWriteGuard, RwLock, RwLockReadGuard, RwLockWriteGuard,
};
use raw_window_handle::RawWindowHandle;

use crate::debug_utils::DebugMessenger;

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

/// Wrapper type to destroy a Vulkan instance when dropped.
///
/// This allows the instance to be destroyed at a time defined by Rust's drop
/// order.
///
/// Implements `Deref<Target = ash::Instance>`.
struct InstanceDropper(ash::Instance);

impl Deref for InstanceDropper {
    type Target = ash::Instance;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Drop for InstanceDropper {
    fn drop(&mut self) {
        unsafe { self.0.destroy_instance(None) };
    }
}

struct InstanceInner {
    // NOTE: Drop correctness depends on field order!

    // Extensions.
    get_physical_device_properties: ExtensionFn<vk::KhrGetPhysicalDeviceProperties2Fn>,
    external_memory_capabilities: ExtensionFn<vk::KhrExternalMemoryCapabilitiesFn>,
    display: khr::Display,
    debug_utils: ext::DebugUtils,

    // Underlying instance. Destroys the instance when dropped.
    raw: InstanceDropper,

    // Loaded dynamic library. Must be dropped *after* all Vulkan API calls have
    // completed.
    entry: ash::Entry,
}

#[derive(Clone)]
pub struct Instance {
    inner: Arc<RwLock<InstanceInner>>,
}

impl Instance {
    /// Lists the set of required extensions for the current platform.
    fn required_extensions(entry: &ash::Entry, api_version: ApiVersion) -> Vec<&'static CStr> {
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
    fn required_layers(entry: &ash::Entry) -> Vec<&'static CStr> {
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
        let entry = unsafe { ash::Entry::new() }.expect("failed to load Vulkan dlib");

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

        let extensions = Self::required_extensions(&entry, driver_api_version);
        let layers = Self::required_layers(&entry);

        let raw_instance = {
            let ext_ptrs = extensions.iter().map(|&s| s.as_ptr()).collect::<Vec<_>>();
            let layer_ptrs = layers.iter().map(|&s| s.as_ptr()).collect::<Vec<_>>();

            let create_info = vk::InstanceCreateInfo::builder()
                .flags(vk::InstanceCreateFlags::empty())
                .application_info(&app_info)
                .enabled_layer_names(&layer_ptrs)
                .enabled_extension_names(&ext_ptrs);

            match unsafe { entry.create_instance(&create_info, None) } {
                // Immediately wrap in case the rest of the function panics.
                Ok(raw) => InstanceDropper(raw),
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
                let addr = entry.get_instance_proc_addr(raw_instance.handle(), name.as_ptr());
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
                let addr = entry.get_instance_proc_addr(raw_instance.handle(), name.as_ptr());
                std::mem::transmute::<vk::PFN_vkVoidFunction, *const std::ffi::c_void>(addr)
            };

            ExtensionFn::Extension(vk::KhrExternalMemoryCapabilitiesFn::load(load_fn))
        };

        if !extensions.contains(&khr::Display::name()) {
            panic!("missing required extension: VK_KHR_display");
        }

        let display = khr::Display::new(&entry, &*raw_instance);

        if !instance_extensions.iter().any(|props| {
            i8_slice_to_cstr(&props.extension_name).unwrap() == ext::DebugUtils::name()
        }) {
            panic!("missing required extension: VK_EXT_debug_utils")
        }

        let debug_utils = ext::DebugUtils::new(&entry, &*raw_instance);

        let instance = InstanceInner {
            get_physical_device_properties,
            external_memory_capabilities,
            display,
            debug_utils,
            raw: raw_instance,
            entry,
        };

        Instance {
            inner: Arc::new(RwLock::new(instance)),
        }
    }

    /// Acquires a read lock on the instance, returning the underlying instance
    /// handle.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the returned raw instance handle is only
    /// used for Vulkan API calls which do not specify an external
    /// synchronization requirement.
    ///
    /// If the raw instance handle is copied out of the returned guard, the
    /// caller must ensure that further access to the handle is externally
    /// synchronized.
    pub(crate) unsafe fn read_raw(&self) -> MappedRwLockReadGuard<ash::Instance> {
        RwLockReadGuard::map(self.inner.read(), |inst| &inst.raw.0)
    }

    /// Acquires a write lock on the instance, returning the underlying instance
    /// handle.
    ///
    /// # Safety
    ///
    /// If the raw instance handle is copied out of the returned guard, the
    /// caller must ensure that further access to the handle is externally
    /// synchronized.
    pub(crate) unsafe fn write_raw(&self) -> MappedRwLockWriteGuard<ash::Instance> {
        RwLockWriteGuard::map(self.inner.write(), |inst| &mut inst.raw.0)
    }

    /// Acquires a write lock on the instance, returning the `DebugUtils` extension.
    ///
    /// # Safety
    ///
    /// If the raw instance handle is copied out of the extension object in the
    /// returned guard, the caller must ensure that further access to the handle
    /// is externally synchronized.
    pub(crate) unsafe fn write_ext_debug_utils_raw(
        &self,
    ) -> MappedRwLockWriteGuard<ext::DebugUtils> {
        RwLockWriteGuard::map(self.inner.write(), |inst| &mut inst.debug_utils)
    }

    /// Initializes a debug messenger for this instance.
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
        let inner = self.inner.read();

        let raw = match window {
            RawWindowHandle::Xlib(xlib) => {
                let xlib_ext = khr::XlibSurface::new(&inner.entry, &*inner.raw);
                let create_info = vk::XlibSurfaceCreateInfoKHR::builder()
                    .flags(vk::XlibSurfaceCreateFlagsKHR::empty())
                    .dpy(xlib.display as *mut _)
                    .window(xlib.window);

                unsafe { xlib_ext.create_xlib_surface(&create_info, None) }
                    .expect("failed to create Xlib window surface")
            }
            RawWindowHandle::Xcb(xcb) => {
                let xcb_ext = khr::XcbSurface::new(&inner.entry, &*inner.raw);
                let create_info = vk::XcbSurfaceCreateInfoKHR::builder()
                    .flags(vk::XcbSurfaceCreateFlagsKHR::empty())
                    .window(xcb.window)
                    .connection(xcb.connection);

                unsafe { xcb_ext.create_xcb_surface(&create_info, None) }
                    .expect("failed to create XCB window surface")
            }
            RawWindowHandle::Wayland(_) => todo!(),
            _ => panic!("unrecognized window handle"),
        };

        Surface {
            inner: Arc::new(RwLock::new(SurfaceInner {
                instance: self.clone(),
                raw,
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
    pub fn create_surface(&self, window: RawWindowHandle) -> () {
        compile_error!("Unsupported platform (only linux is supported).");
    }

    pub fn enumerate_physical_devices(&self, surface: &Surface) -> Vec<PhysicalDevice> {
        let inner = self.inner.read();

        // Safety: Instance is externally synchronized by locking.
        let raw_devices = match unsafe { inner.raw.enumerate_physical_devices() } {
            Ok(devices) => devices,
            Err(e) => {
                log::error!("failed to enumerate physical devices: {}", e);
                Vec::new()
            }
        };

        // Drop the lock so that it may be re-acquired for physical device initialization.
        drop(inner);

        raw_devices
            .into_iter()
            .map(|raw| {
                // Safety: Physical device handle comes from the correct instance.
                unsafe { PhysicalDevice::new(self.clone(), raw, surface.inner.read().raw) }
            })
            .collect()
    }
}

pub struct PhysicalDeviceInner {
    instance: Instance,
    raw: vk::PhysicalDevice,
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
        raw: vk::PhysicalDevice,
        surface: vk::SurfaceKHR,
    ) -> PhysicalDevice {
        // Safety: raw instance handle is not copied out of the guard.
        let queue_families = unsafe {
            instance
                .read_raw()
                .get_physical_device_queue_family_properties(raw)
        };

        enum QueueSelection {
            Dedicated(u32),
            General(u32),
        }

        let mut graphics_queue = None;
        let mut transfer_queue = None;
        let mut present_queue = None;

        let surface_ext = {
            let instance = instance.inner.read();
            khr::Surface::new(&instance.entry, &*instance.raw)
        };

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
                    .get_physical_device_surface_support(raw, index as u32, surface)
                    .expect(&format!(
                        "failed to query queue family {} for surface support",
                        index
                    ))
            } {
                present_queue = Some(index);
            }
        }

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
                raw,
                queue_families,
                graphics_queue_family,
                transfer_queue_family,
                present_queue_family,
            }),
        }
    }

    pub fn properties(&self) -> vk::PhysicalDeviceProperties {
        let mut props = vk::PhysicalDeviceProperties2::default();

        let instance = self.instance();
        let instance_inner = instance.inner.read();

        match &instance_inner.get_physical_device_properties {
            // Safety: No external synchronization requirement.
            ExtensionFn::Core => unsafe {
                instance_inner
                    .raw
                    .get_physical_device_properties2(self.inner.raw, &mut props)
            },
            ExtensionFn::Extension(ext) => unsafe {
                ext.get_physical_device_properties2_khr(self.inner.raw, &mut props)
            },
        }

        props.properties
    }

    pub fn features(&self) -> vk::PhysicalDeviceFeatures {
        // Safety: No external synchronization requirement.
        unsafe {
            self.instance()
                .read_raw()
                .get_physical_device_features(self.inner.raw)
        }
    }

    #[inline]
    fn instance(&self) -> Instance {
        self.inner.instance.clone()
    }

    // Graphics, compute, transfer, present
    const MAX_NEEDED_QUEUE_FAMILIES: usize = 4;

    pub fn create_device(&self) -> Device {
        let mut unique_queue_families: ArrayVec<u32, { Self::MAX_NEEDED_QUEUE_FAMILIES }> =
            ArrayVec::new();
        for family in [
            self.inner.graphics_queue_family,
            self.inner.transfer_queue_family,
            self.inner.present_queue_family,
        ] {
            if !unique_queue_families.contains(&family) {
                unique_queue_families.push(family);
            }
        }

        let queue_create_infos: ArrayVec<
            vk::DeviceQueueCreateInfo,
            { Self::MAX_NEEDED_QUEUE_FAMILIES },
        > = unique_queue_families
            .iter()
            .map(|qf| {
                vk::DeviceQueueCreateInfo::builder()
                    .flags(vk::DeviceQueueCreateFlags::empty())
                    .queue_family_index(*qf)
                    .queue_priorities(&[1.0])
                    .build()
            })
            .collect();

        let phys_device_features = vk::PhysicalDeviceFeatures::builder();
        let enabled_layer_names = &[LAYER_NAME_VALIDATION.as_ptr() as *const i8];
        let device_create_info = vk::DeviceCreateInfo::builder()
            .flags(vk::DeviceCreateFlags::empty())
            .queue_create_infos(&queue_create_infos)
            .enabled_layer_names(enabled_layer_names)
            .enabled_extension_names(&[])
            .enabled_features(&phys_device_features);

        // Safety: no external synchronization requirement.
        let raw_device = unsafe {
            self.instance()
                .read_raw()
                .create_device(self.inner.raw, &device_create_info, None)
                .expect("failed to create logical device")
        };

        log::info!("Successfully created logical device.");

        Device {
            inner: Arc::new(DeviceInner {
                instance: self.instance().clone(),
                phys_device: self.clone(),
                raw: raw_device,
            }),
        }
    }
}

pub struct DeviceInner {
    // NOTE: sensitive drop order.
    raw: ash::Device,
    phys_device: PhysicalDevice,
    instance: Instance,
}

impl Drop for DeviceInner {
    fn drop(&mut self) {
        // Safety:
        // - All child objects created on the device have a handle to it, so it
        //   will not be destroyed until said child objects have been.
        // - No allocation callbacks are provided at logical device
        //   construction, so no callbacks are provided at destruction.
        // - Host access is guaranteed to be synchronized by the mutable
        //   receiver.
        unsafe { self.raw.destroy_device(None) }
    }
}

#[derive(Clone)]
pub struct Device {
    inner: Arc<DeviceInner>,
}

impl Device {
    /// Returns a queue owned by the logical device.
    ///
    /// # Safety
    ///
    /// - The family index must have been specified at logical device creation.
    /// - The queue index must be less than the number of queues requested at
    ///   logical device creation.
    unsafe fn owned_queue(&self, family: u32, queue: u32) -> Queue {
        let raw = unsafe { self.inner.raw.get_device_queue(family, queue) };

        Queue {
            inner: Arc::new(RwLock::new(QueueInner {
                device: self.clone(),
                raw,
            })),
        }
    }

    pub fn graphics_queue(&self) -> Queue {
        let family = self.inner.phys_device.inner.graphics_queue_family;

        // Safety:
        // - Queue family index was specified at construction.
        // - Queue index is less than the number of requested queues.
        // - Queue flags were empty at device creation.
        unsafe { self.owned_queue(family, 0) }
    }

    pub fn present_queue(&self) -> Queue {
        let family = self.inner.phys_device.inner.present_queue_family;

        // Safety:
        // - Queue family index was specified at construction.
        // - Queue index is less than the number of requested queues.
        // - Queue flags were empty at device creation.
        unsafe { self.owned_queue(family, 0) }
    }
}

struct QueueInner {
    device: Device,
    raw: vk::Queue,
}

pub struct Queue {
    inner: Arc<RwLock<QueueInner>>,
}

struct SurfaceInner {
    instance: Instance,
    raw: vk::SurfaceKHR,
}

impl Drop for SurfaceInner {
    fn drop(&mut self) {
        let instance = self.instance.inner.read();

        // Safety:
        // - Swapchain objects have handles to this surface, so it will not be
        //   dropped before they are.
        // - Host access synchronization guaranteed by mutable receiver.
        unsafe {
            let surface_ext = khr::Surface::new(&instance.entry, &*instance.raw);
            surface_ext.destroy_surface(self.raw, None);
        }
    }
}

pub struct Surface {
    inner: Arc<RwLock<SurfaceInner>>,
}
