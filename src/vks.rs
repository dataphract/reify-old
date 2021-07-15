//! Safer (but still unsafe) wrappers over the Vulkan API.
//!
//! This module provides zero-cost wrappers for Vulkan objects and functions
//! which eliminate certain invalid usages of the API without incurring runtime
//! overhead. In particular, Rust's ownership and borrowing rules are used to
//! enforce Vulkan's external synchronization rules: functions which require
//! that a parameter be externally synchronized accept the object type either by
//! mutable reference or by value (in case the function may destroy the object).

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
    /// - No copy of the returned raw handle may be used in any Vulkan API call
    ///   once the borrow expires.
    unsafe fn handle(&self) -> &Self::Handle;

    /// Returns a mutable reference to the underlying Vulkan object.
    ///
    /// # Safety
    ///
    /// The caller must uphold the following invariants:
    /// - No copy of the returned raw handle may be used in any Vulkan API call
    ///   once the mutable borrow expires.
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

    /// Creates an image view from an existing image.
    ///
    /// # Safety
    ///
    /// // TODO
    pub unsafe fn create_image_view(
        &self,
        create_info: &ImageViewCreateInfo,
    ) -> VkResult<ImageView> {
        unsafe {
            let builder = vk::ImageViewCreateInfoBuilder::new()
                .flags(create_info.flags)
                .image(*create_info.image.handle())
                .view_type(create_info.view_type)
                .format(create_info.format)
                .components(create_info.components)
                .subresource_range(create_info.subresource_range);
            self.loader
                .create_image_view(&builder, None)
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
    pub unsafe fn destroy_swapchain(&self, mut swapchain: SwapchainKHR) {
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
}

// ============================================================================

define_handle! {
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

define_handle! {
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

define_handle! {
    /// An opaque handle to a Vulkan image view object.
    pub struct ImageView(vk::ImageView);
}

impl ImageView {
    /// Constructs a new `Image` which owns the image view associated with the
    /// raw handle `raw`.
    ///
    /// # Safety
    ///
    /// The caller must uphold the following invariants:
    /// - `raw` must not be used as a parameter to any Vulkan API call after
    ///   this constructor is called.
    /// - `raw` must not be used to create another `ImageView` object.
    /// // TODO: destroy before parent device is destroyed
    pub unsafe fn new(raw: vk::ImageView) -> ImageView {
        ImageView { raw }
    }
}

pub struct ImageViewCreateInfo<'img> {
    pub flags: vk::ImageViewCreateFlags,
    pub image: &'img Image,
    pub view_type: vk::ImageViewType,
    pub format: vk::Format,
    pub components: vk::ComponentMapping,
    pub subresource_range: vk::ImageSubresourceRange,
}

// ============================================================================

define_handle! {
    /// An opaque handle to a Vulkan shader module.
    pub struct ShaderModule(vk::ShaderModule);
}

impl ShaderModule {
    /// Constructs a new `ShaderModule` which owns the shader module associated
    /// with the raw handle `raw`.
    ///
    /// # Safety
    ///
    /// The caller must uphold the following invariants:
    /// - `raw` must not be used as a parameter to any Vulkan API call after
    ///   this constructor is called.
    /// - `raw` must not be used to create another `ShaderModule` object.
    /// - TODO: destroy before parent device is destroyed
    pub unsafe fn new(raw: vk::ShaderModule) -> ShaderModule {
        ShaderModule { raw }
    }
}

// ============================================================================

define_handle! {
    /// An opaque handle to a Vulkan surface object.
    pub struct SurfaceKHR(vk::SurfaceKHR);
}

impl SurfaceKHR {
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
    pub unsafe fn new(raw: vk::SurfaceKHR) -> SurfaceKHR {
        SurfaceKHR { raw }
    }
}

// ============================================================================

define_handle! {
    /// An opaque handle to a Vulkan swapchain object.
    pub struct SwapchainKHR(vk::SwapchainKHR);
}

impl SwapchainKHR {
    /// Constructs a new `Swapchain` which owns the swapchain associated with
    /// the raw handle `raw`.
    ///
    /// # Safety
    ///
    /// The caller must uphold the following invariants:
    /// - `raw` must not be used as a parameter to any Vulkan API call after
    ///   this constructor is called.
    /// - `raw` must not be used to create another `Swapchain` object.
    pub unsafe fn new(raw: vk::SwapchainKHR) -> SwapchainKHR {
        SwapchainKHR { raw }
    }
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

// ============================================================================

define_handle! {
    pub struct DebugUtilsMessengerEXT(vk::DebugUtilsMessengerEXT);
}

impl DebugUtilsMessengerEXT {
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
    pub unsafe fn new(raw: vk::DebugUtilsMessengerEXT) -> DebugUtilsMessengerEXT {
        DebugUtilsMessengerEXT { raw }
    }
}

// ============================================================================
