//! Safer (but still unsafe) wrappers over the Vulkan API.
//!
//! This module provides zero-cost wrappers for Vulkan objects and functions
//! which eliminate certain invalid usages of the API without incurring runtime
//! overhead. In particular, the `Handle` and `HandleMut` types are used to
//! enforce Vulkan's external synchronization rules: functions which require
//! that a parameter be externally synchronized accept a `HandleMut`, which may
//! not be obtained without exclusive access to the object.

use erupt::{vk, DeviceLoader, EntryLoader, InstanceLoader, LoaderError};

use std::lazy::SyncOnceCell;

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
/// exclusive ownership of a Vulkan object and may allow use of the object via
/// the `handle` method.
///
/// # Safety
///
/// Implementors of this trait must uphold the following invariants:
/// - The implementing type must not implement `Copy`.
/// - If a value of the implementing type can be duplicated (e.g., if the type is
///   `Clone`), duplicating the value must not duplicate the underlying raw
///   handle.
/// - The raw handle owned by values of the implementing type must not be
///   accessible except through the `handle` trait method or the `handle_mut`
///   method on `VkSyncObject`, which extends this trait.
pub unsafe trait VkObject {
    /// The raw handle type.
    type Raw;

    /// Returns a `Handle` to the underlying Vulkan object.
    fn handle<'a>(&'a self) -> Handle<'a, Self::Raw>;
}

/// A Vulkan API object which has external synchronization requirements.
///
/// This trait extends `VkObject` to represent Vulkan objects that require
/// external synchronization during at least one Vulkan API call.
///
/// # Safety
///
/// Implementors of this trait must uphold the following invariants:
/// - The implementing type must not implement `Copy`.
/// - If a value of the implementing type can be duplicated (e.g., if the type is
///   `Clone`), duplicating the value must not duplicate the underlying raw
///   handle.
/// - The raw handle owned by values of the implementing type must not be
///   accessible except through the `handle_mut` trait method or the `handle`
///   method on `VkObject`, which is extended by this trait.
pub unsafe trait VkSyncObject: VkObject {
    /// Returns a `HandleMut` to the underlying Vulkan object.
    fn handle_mut<'a>(&'a mut self) -> HandleMut<'a, Self::Raw>;
}

/// A handle to a Vulkan object which does not guarantee external synchronization.
pub struct Handle<'a, T> {
    raw: &'a T,
}

impl<'a, T> Handle<'a, T> {
    /// Returns a raw handle to the associated Vulkan object.
    ///
    /// # Safety
    ///
    /// The caller must uphold the following invariants:
    /// - The raw handle must not be used as a parameter in any Vulkan API call
    ///   which specifies an external synchronization requirement on that
    ///   parameter.
    /// - No copy of the returned raw handle may outlive the `Handle` value.
    #[inline(always)]
    pub(crate) unsafe fn raw(&self) -> &T {
        self.raw
    }
}

/// A handle to a Vulkan object which guarantees external synchronization.
pub struct HandleMut<'a, T> {
    raw: &'a mut T,
}

impl<'a, T> HandleMut<'a, T> {
    /// Returns a raw handle to the associated Vulkan object.
    ///
    /// # Safety
    ///
    /// The caller must uphold the following invariants:
    /// - No copy of the returned raw handle may outlive the `Handle` value.
    #[inline(always)]
    pub(crate) unsafe fn raw_mut(&self) -> &T {
        self.raw
    }
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
            type Raw = $raw;

            #[inline(always)]
            fn handle<'a>(&'a self) -> Handle<'a, Self::Raw> {
                Handle { raw: &self.raw }
            }
        }
    };
}

/// A macro to define a wrapper type around a raw Vulkan handle and implement
/// `VkObject` and `VkSyncObject` for the wrapper type.
macro_rules! define_sync_handle {
    ($(#[$outer:meta])? $v:vis struct $defty:ident($raw:ty);) => {
        define_handle! {
            $(#[$outer])?
            $v struct $defty($raw);
        }

        unsafe impl VkSyncObject for $defty {
            #[inline(always)]
            fn handle_mut<'a>(&'a mut self) -> HandleMut<'a, Self::Raw> {
                HandleMut { raw: &mut self.raw }
            }
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
    type Raw = InstanceLoader;

    fn handle<'a>(&'a self) -> Handle<'a, Self::Raw> {
        Handle { raw: &self.loader }
    }
}

unsafe impl VkSyncObject for Instance {
    fn handle_mut<'a>(&'a mut self) -> HandleMut<'a, Self::Raw> {
        HandleMut {
            raw: &mut self.loader,
        }
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
        phys_device: Handle<'_, vk::PhysicalDevice>,
    ) -> vk::PhysicalDeviceFeatures {
        unsafe { self.loader.get_physical_device_features(*phys_device.raw()) }
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
        phys_device: Handle<'_, vk::PhysicalDevice>,
    ) -> vk::PhysicalDeviceProperties {
        unsafe {
            self.loader
                .get_physical_device_properties(*phys_device.raw())
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
        phys_device: Handle<'_, vk::PhysicalDevice>,
    ) -> Vec<vk::QueueFamilyProperties> {
        unsafe {
            self.loader
                .get_physical_device_queue_family_properties(*phys_device.raw(), None)
        }
    }

    /// Queries is presentation is supported.
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
        phys_device: Handle<'_, vk::PhysicalDevice>,
        queue_family_index: u32,
        surface: Handle<'_, vk::SurfaceKHR>,
    ) -> VkResult<bool> {
        unsafe {
            self.loader
                .get_physical_device_surface_support_khr(
                    *phys_device.raw(),
                    queue_family_index,
                    *surface.raw(),
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
        phys_device: Handle<'_, vk::PhysicalDevice>,
        surface: Handle<'_, vk::SurfaceKHR>,
    ) -> VkResult<vk::SurfaceCapabilitiesKHR> {
        unsafe {
            self.loader
                .get_physical_device_surface_capabilities_khr(*phys_device.raw(), *surface.raw())
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
        phys_device: Handle<'_, vk::PhysicalDevice>,
        surface: Handle<'_, vk::SurfaceKHR>,
    ) -> VkResult<Vec<vk::SurfaceFormatKHR>> {
        unsafe {
            self.loader
                .get_physical_device_surface_formats_khr(*phys_device.raw(), *surface.raw(), None)
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
        phys_device: Handle<'_, vk::PhysicalDevice>,
        surface: Handle<'_, vk::SurfaceKHR>,
    ) -> VkResult<Vec<vk::PresentModeKHR>> {
        unsafe {
            self.loader
                .get_physical_device_surface_present_modes_khr(
                    *phys_device.raw(),
                    *surface.raw(),
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
        phys_device: Handle<'_, vk::PhysicalDevice>,
        create_info: &vk::DeviceCreateInfo,
    ) -> Result<Device, LoaderError> {
        unsafe {
            let loader = DeviceLoader::new(&self.loader, *phys_device.raw(), create_info, None)?;
            Ok(Device::new(loader))
        }
    }

    #[inline]
    pub unsafe fn create_debug_utils_messenger(
        &self,
        info: &vk::DebugUtilsMessengerCreateInfoEXT,
    ) -> VkResult<DebugMessenger> {
        unsafe {
            let debug_messenger = self
                .loader
                .create_debug_utils_messenger_ext(info, None)
                .result()?;

            Ok(DebugMessenger::new(debug_messenger))
        }
    }

    pub unsafe fn destroy_debug_utils_messenger(
        &self,
        messenger: HandleMut<'_, vk::DebugUtilsMessengerEXT>,
    ) {
        // Safety:
        // - Access to debug messenger is externally synchronized via `HandleMut`.
        unsafe {
            self.loader
                .destroy_debug_utils_messenger_ext(Some(*messenger.raw_mut()), None);
        }
    }

    pub unsafe fn destroy_surface(&self, mut surface: Surface) {
        // Safety:
        // - Access to surface is externally synchronized via ownership.
        unsafe {
            self.loader
                .destroy_surface_khr(Some(*surface.handle_mut().raw_mut()), None);
        }
    }
}

// ============================================================================

define_handle! {
    /// An opaque handle to a Vulkan physical device object.
    pub struct PhysicalDevice(vk::PhysicalDevice);
}

impl PhysicalDevice {
    /// Constructs a new `PhysicalDevice` which owns the physical device object
    /// associated with the raw handle `raw`.
    ///
    /// # Safety
    ///
    /// The caller must uphold the following invariants:
    /// - `raw` must not be used as a parameter to any Vulkan API call after
    ///   this constructor is called.
    /// - `raw` must not be used to create another `PhysicalDevice` object.
    pub unsafe fn new(raw: vk::PhysicalDevice) -> PhysicalDevice {
        PhysicalDevice { raw }
    }
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
    type Raw = DeviceLoader;

    fn handle<'a>(&'a self) -> Handle<'a, Self::Raw> {
        Handle { raw: &self.loader }
    }
}

unsafe impl VkSyncObject for Device {
    fn handle_mut<'a>(&'a mut self) -> HandleMut<'a, Self::Raw> {
        HandleMut {
            raw: &mut self.loader,
        }
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

    /// Create a swapchain.
    ///
    /// # Safety
    ///
    /// The caller must uphold the following invariants:
    /// - `device` must be a handle to a device object associated with this
    ///   instance.
    /// - `create_info.surface` must be a handle to a surface associated with
    ///   this instance.
    /// - If `create_info.old_swapchain` is `Some(old)`, then `old` must be a
    ///   handle to a non-retired swapchain associated with
    ///   `create_info.surface`.
    #[inline]
    pub unsafe fn create_swapchain(
        &self,
        create_info: &mut SwapchainCreateInfo<'_, '_>,
    ) -> VkResult<Swapchain> {
        unsafe {
            let raw_create_info = vk::SwapchainCreateInfoKHRBuilder::new()
                .flags(create_info.flags)
                // Safety: surface is externally synchronized via `HandleMut`.
                .surface(*create_info.surface.raw_mut())
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
                // Safety: old swapchain, if any, is externally synchronized via `HandleMut`.
                .old_swapchain(
                    create_info
                        .old_swapchain
                        .take()
                        .map(|mut os| *os.handle_mut().raw_mut())
                        .unwrap_or(vk::SwapchainKHR::null()),
                );

            let swapchain = self
                .loader
                .create_swapchain_khr(&raw_create_info, None)
                .result()?;

            Ok(Swapchain::new(swapchain))
        }
    }

    /// Destroy a swapchain object.
    ///
    /// # Safety
    ///
    /// The caller must uphold the following invariants:
    /// - `swapchain` must be a swapchain object associated with this instance.
    pub unsafe fn destroy_swapchain(&self, mut swapchain: Swapchain) {
        unsafe {
            // Safety:
            // - `swapchain` is externally synchronized, as it is received by value.
            self.loader
                .destroy_swapchain_khr(Some(*swapchain.handle_mut().raw_mut()), None);
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
        swapchain: Handle<'_, vk::SwapchainKHR>,
    ) -> VkResult<Vec<Image>> {
        unsafe {
            Ok(self
                .loader
                .get_swapchain_images_khr(*swapchain.raw(), None)
                .result()?
                .into_iter()
                .map(|img| Image::new(img))
                .collect())
        }
    }
}

// ============================================================================

define_sync_handle! {
    /// An opaque handle to a Vulkan queue object.
    pub struct Queue(vk::Queue);
}

impl Queue {
    /// Constructs a new `Queue` which owns the queue associated with the raw
    /// handle `raw`.
    ///
    /// # Safety
    ///
    /// The caller must uphold the following invariants:
    /// - `raw` must not be used as a parameter to any Vulkan API call after
    ///   this constructor is called.
    /// - `raw` must not be used to create another `Queue` object.
    /// - TODO: must not be used after parent `Device` is destroyed
    pub unsafe fn new(raw: vk::Queue) -> Queue {
        Queue { raw }
    }
}

// ============================================================================

define_sync_handle! {
    /// An opaque handle to a Vulkan image object.
    pub struct Image(vk::Image);
}

impl Image {
    /// Constructs a new `Image` which owns the image associated with the raw
    /// handle `raw`.
    ///
    /// # Safety
    ///
    /// The caller must uphold the following invariants:
    /// - `raw` must not be used as a parameter to any Vulkan API call after
    ///   this constructor is called.
    /// - `raw` must not be used to create another `Image` object.
    /// - If the image was created by a device, it must be destroyed by a call
    ///   to `Device::destroy_image`.
    /// - If the image was obtained from a swapchain with `Device::get_swapchain_images_khr`,
    ///   it must not be used after the associated swapchain has been destroyed.
    pub unsafe fn new(raw: vk::Image) -> Image {
        Image { raw }
    }
}

// ============================================================================

define_sync_handle! {
    /// An opaque handle to a Vulkan surface object.
    pub struct Surface(vk::SurfaceKHR);
}

impl Surface {
    /// Constructs a new `Surface` which owns the surface associated with the
    /// raw handle `raw`.
    ///
    /// # Safety
    ///
    /// The caller must uphold the following invariants:
    /// - `raw` must not be used as a parameter to any Vulkan API call after
    ///   this constructor is called.
    /// - `raw` must not be used to create another `Surface` object.
    /// - TODO: must be destroyed before instance is destroyed
    pub unsafe fn new(raw: vk::SurfaceKHR) -> Surface {
        Surface { raw }
    }
}

// ============================================================================

define_sync_handle! {
    /// An opaque handle to a Vulkan swapchain object.
    pub struct Swapchain(vk::SwapchainKHR);
}

impl Swapchain {
    /// Constructs a new `Swapchain` which owns the swapchain associated with
    /// the raw handle `raw`.
    ///
    /// # Safety
    ///
    /// The caller must uphold the following invariants:
    /// - `raw` must not be used as a parameter to any Vulkan API call after
    ///   this constructor is called.
    /// - `raw` must not be used to create another `Swapchain` object.
    pub unsafe fn new(raw: vk::SwapchainKHR) -> Swapchain {
        Swapchain { raw }
    }
}

pub struct SwapchainCreateInfo<'surf, 'queues> {
    pub flags: vk::SwapchainCreateFlagsKHR,
    pub surface: HandleMut<'surf, vk::SurfaceKHR>,
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
    pub old_swapchain: Option<Swapchain>,
}

// ============================================================================

define_sync_handle! {
    pub struct DebugMessenger(vk::DebugUtilsMessengerEXT);
}

impl DebugMessenger {
    /// Constructs a new `DebugMessenger` which owns the debug utils messenger
    /// associated with the raw handle `raw`.
    ///
    /// # Safety
    ///
    /// The caller must uphold the following invariants:
    /// - `raw` must not be used as a parameter to any Vulkan API call after
    ///   this constructor is called.
    /// - `raw` must not be used to create another `DebugMessenger` object.
    /// - TODO: must be destroyed before instance is destroyed
    pub unsafe fn new(raw: vk::DebugUtilsMessengerEXT) -> DebugMessenger {
        DebugMessenger { raw }
    }
}

// ============================================================================
