//! Safer (but still unsafe) wrappers over the Vulkan API.
//!
//! This module provides zero-cost wrappers for Vulkan objects and functions
//! which eliminate certain invalid usages of the API without incurring runtime
//! overhead. In particular, the `Handle` and `HandleMut` types are used to
//! enforce Vulkan's external synchronization rules: functions which require
//! that a parameter be externally synchronized accept a `HandleMut`, which may
//! not be obtained without exclusive access to the object.

use ash::{
    extensions::{ext, khr},
    prelude::VkResult,
    version::{DeviceV1_0, EntryV1_0, InstanceV1_0},
    vk,
};

use std::lazy::SyncOnceCell;

static ENTRY: SyncOnceCell<ash::Entry> = SyncOnceCell::new();

pub fn entry() -> &'static ash::Entry {
    ENTRY
        .get_or_try_init(|| unsafe { ash::Entry::new() })
        .expect("failed to load Vulkan dynamic library")
}

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

define_sync_handle! {
    /// An opaque handle to a Vulkan instance.
    pub struct Instance(ash::Instance);
}

impl Instance {
    /// Constructs a new `Instance` which owns the Vulkan instance associated
    /// with the raw handle `raw`.
    ///
    /// # Safety
    ///
    /// The caller must guarantee that `raw` is not used as a parameter to any
    /// Vulkan API call after this constructor is called.
    #[inline]
    pub unsafe fn new(raw: ash::Instance) -> Instance {
        Instance { raw }
    }

    #[inline]
    pub fn enumerate_physical_devices(&self) -> ash::prelude::VkResult<Vec<PhysicalDevice>> {
        unsafe {
            Ok(self
                .handle()
                .raw()
                .enumerate_physical_devices()?
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
        unsafe {
            self.handle()
                .raw()
                .get_physical_device_features(*phys_device.raw())
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
        phys_device: Handle<'_, vk::PhysicalDevice>,
    ) -> vk::PhysicalDeviceProperties {
        unsafe {
            self.handle()
                .raw()
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
            self.handle()
                .raw()
                .get_physical_device_queue_family_properties(*phys_device.raw())
        }
    }

    #[inline]
    pub unsafe fn create_device(
        &self,
        phys_device: Handle<'_, vk::PhysicalDevice>,
        create_info: &vk::DeviceCreateInfo,
    ) -> VkResult<Device> {
        unsafe {
            Ok(Device::new(self.handle().raw().create_device(
                *phys_device.raw(),
                create_info,
                None,
            )?))
        }
    }

    pub unsafe fn create_debug_utils_messenger(
        &self,
        info: &vk::DebugUtilsMessengerCreateInfoEXT,
    ) -> DebugMessenger {
        let entry = entry();
        let instance = self.handle();

        unsafe {
            let debug_utils = ext::DebugUtils::new(entry, instance.raw());
            DebugMessenger::new(
                debug_utils
                    .create_debug_utils_messenger(info, None)
                    .expect("failed to create debug messenger"),
            )
        }
    }

    pub unsafe fn destroy_debug_utils_messenger(
        &self,
        messenger: HandleMut<'_, vk::DebugUtilsMessengerEXT>,
    ) {
        let entry = entry();
        let instance = self.handle();

        // Safety:
        // - DebugUtils instance handle is obtained from `Handle`.
        // - Access to debug messenger is externally synchronized via `HandleMut`.
        unsafe {
            let debug_utils = ext::DebugUtils::new(entry, instance.raw());
            debug_utils.destroy_debug_utils_messenger(*messenger.raw_mut(), None);
        }
    }
}

impl Drop for Instance {
    fn drop(&mut self) {
        // Safety: raw handle is externally synchronized by InstanceHandle contract.
        unsafe { self.raw.destroy_instance(None) }
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

define_sync_handle! {
    /// An opaque handle to a Vulkan logical device object.
    pub struct Device(ash::Device);
}

impl Device {
    /// Constructs a new `Device` which owns the logical device object
    /// associated with the raw handle `raw`.
    ///
    /// # Safety
    ///
    /// The caller must uphold the following invariants:
    /// - `raw` must not be used as a parameter to any Vulkan API call after
    ///   this constructor is called.
    /// - `raw` must not be used to create another `Device` object.
    pub unsafe fn new(raw: ash::Device) -> Device {
        Device { raw }
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe { self.raw.destroy_device(None) }
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
