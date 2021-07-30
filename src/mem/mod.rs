pub mod buddy;
pub use buddy::{BuddyAllocator, BuddyBlock, BuddyBuilder, BuddyError};

use std::{
    collections::LinkedList,
    fmt,
    iter::FromIterator,
    marker::PhantomData,
    num::NonZeroU32,
    sync::{Arc, Weak},
};

use arrayvec::ArrayVec;
use erupt::vk;
use parking_lot::{MappedRwLockReadGuard, RwLock, RwLockReadGuard};

use crate::{util::ErrorOnDrop, vks, Device};

// Vulkan implementations are required by the spec to support at least this many
// separate memory allocations. See §42.1 Limit Requirements, Table 53 Required
// Limits, `maxMemoryAllocationCount`.
const MAX_MEMORY_ALLOCATION_COUNT: usize = 4096;

pub struct PhysicalDeviceMemoryProperties {
    types: ArrayVec<vk::MemoryType, { vk::MAX_MEMORY_TYPES as usize }>,
    heaps: ArrayVec<vk::MemoryHeap, { vk::MAX_MEMORY_HEAPS as usize }>,
}

impl From<vk::PhysicalDeviceMemoryProperties> for PhysicalDeviceMemoryProperties {
    fn from(props: vk::PhysicalDeviceMemoryProperties) -> Self {
        PhysicalDeviceMemoryProperties {
            types: ArrayVec::from_iter(
                props
                    .memory_types
                    .iter()
                    .copied()
                    .take(props.memory_type_count as usize),
            ),
            heaps: ArrayVec::from_iter(
                props
                    .memory_heaps
                    .iter()
                    .copied()
                    .take(props.memory_heap_count as usize),
            ),
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq)]
pub struct MemoryConfig {
    // TODO: how to handle using same heap for both? relevant on e.g. integrated
    // graphics on CPU using host memory
    pub min_host_memory: u64,
    pub min_device_memory: u64,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct HostVisible {
    type_index: u32,
    heap_index: u32,
    is_host_coherent: bool,
    is_host_cached: bool,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct DeviceLocal {
    type_index: u32,
    heap_index: u32,
    is_host_visible: bool,
    is_host_coherent: bool,
}

impl PhysicalDeviceMemoryProperties {
    fn heap_size(&self, heap_index: u32) -> u64 {
        self.heaps.get(heap_index as usize).unwrap().size
    }

    fn type_properties(&self, type_index: u32) -> vk::MemoryPropertyFlags {
        self.types.get(type_index as usize).unwrap().property_flags
    }

    pub fn select_memory_types(&self, cfg: MemoryConfig) -> MemoryTypes {
        // Quoted from the spec, §11.2.1, "Device Memory Properties":
        //
        //   For each pair of elements X and Y returned in memoryTypes, X must
        //   be placed at a lower index position than Y if:
        //   - the set of bit flags returned in the propertyFlags member of X is
        //     a strict subset of the set of bit flags returned in the
        //     propertyFlags member of Y; or
        //   - the propertyFlags members of X and Y are equal, and X belongs to
        //     a memory heap with greater performance (as determined in an
        //     implementation-specific manner) ; or
        //   - the propertyFlags members of Y includes
        //     VK_MEMORY_PROPERTY_DEVICE_COHERENT_BIT_AMD or
        //     VK_MEMORY_PROPERTY_DEVICE_UNCACHED_BIT_AMD and X does not

        // Host memory is used for staging buffers in transfer operations. It
        // must have the HOST_VISIBLE property. HOST_CACHED is strongly
        // preferred when available. Memory with the DEVICE_LOCAL property
        // should be avoided where possible, though UMA systems may well
        // have DEVICE_LOCAL + HOST_CACHED.
        let mut host: Option<HostVisible> = None;

        // Device memory is used for the majority of a resource's lifetime.
        // It must have the DEVICE_LOCAL property. Memory with any HOST_*
        // properties should be avoided where possible.
        let mut device: Option<DeviceLocal> = None;

        for ty_id in 0u32..self.types.len() as u32 {
            let memty = &self.types[ty_id as usize];
            let flags = memty.property_flags;
            let heap_id = memty.heap_index;

            let is_host_visible = flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE);
            let is_host_coherent = flags.contains(vk::MemoryPropertyFlags::HOST_COHERENT);
            let is_host_cached = flags.contains(vk::MemoryPropertyFlags::HOST_CACHED);
            let is_device_local = flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL);

            if is_host_visible && self.heap_size(heap_id) > cfg.min_host_memory {
                match host {
                    Some(ref old) => {
                        let better_host_cached = is_host_cached && !old.is_host_cached;
                        let same_host_cached = is_host_cached == old.is_host_cached;
                        let larger_heap = self.heap_size(heap_id) > self.heap_size(old.heap_index);

                        if better_host_cached || same_host_cached && larger_heap {
                            host = Some(HostVisible {
                                type_index: ty_id,
                                heap_index: memty.heap_index,
                                is_host_coherent,
                                is_host_cached,
                            });
                        }
                    }

                    None => {
                        host = Some(HostVisible {
                            type_index: ty_id,
                            heap_index: memty.heap_index,
                            is_host_coherent,
                            is_host_cached,
                        })
                    }
                }
            }

            if is_device_local && self.heap_size(heap_id) > cfg.min_device_memory {
                match device {
                    Some(ref old) => {
                        let same_host_visible = is_host_visible == old.is_host_visible;
                        let better_host_visible = !is_host_visible && old.is_host_visible;
                        let same_host_coherent = is_host_coherent == old.is_host_coherent;
                        let better_host_coherent = !is_host_coherent && old.is_host_coherent;

                        let larger_heap = self.heap_size(heap_id) > self.heap_size(old.heap_index);

                        if better_host_visible
                            || better_host_coherent
                            || (same_host_visible && same_host_coherent && larger_heap)
                        {
                            device = Some(DeviceLocal {
                                type_index: ty_id,
                                heap_index: heap_id,
                                is_host_visible,
                                is_host_coherent,
                            });
                        }
                    }

                    None => {
                        device = Some(DeviceLocal {
                            type_index: ty_id,
                            heap_index: heap_id,
                            is_host_visible,
                            is_host_coherent,
                        });
                    }
                }
            }
        }

        MemoryTypes {
            host: host.expect("No suitable host-visible memory type found."),
            device: device.expect("No suitable device-local memory type found."),
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct MemoryTypes {
    host: HostVisible,
    device: DeviceLocal,
}

pub trait MemoryType: private::Sealed {
    fn type_index(&self) -> u32;
    fn heap_index(&self) -> u32;
}

impl private::Sealed for HostVisible {}

impl MemoryType for HostVisible {
    fn type_index(&self) -> u32 {
        self.type_index
    }

    fn heap_index(&self) -> u32 {
        self.heap_index
    }
}

impl private::Sealed for DeviceLocal {}

impl MemoryType for DeviceLocal {
    fn type_index(&self) -> u32 {
        self.type_index
    }

    fn heap_index(&self) -> u32 {
        self.heap_index
    }
}

struct BlockInner<T: MemoryType> {
    memory: Option<vks::DeviceMemory>,
    phantom: PhantomData<T>,
}

struct Block<T: MemoryType> {
    inner: Arc<RwLock<BlockInner<T>>>,
}

pub const MIN_ALLOC_SIZE: vk::DeviceSize = 4096;

pub struct DeviceMemoryRange<T: MemoryType> {
    pool: MemoryPool<T>,

    block: u32,
    idx: u32,
}

impl<T: MemoryType> DeviceMemoryRange<T> {
    unsafe fn with_handle<F, O>(&self, f: F) -> O
    where
        F: FnOnce(&vks::DeviceMemory) -> O,
    {
        let pool_read = self.pool.inner.read();
        let block_read = pool_read.blocks[self.block as usize].inner.read();
        f(block_read.memory.as_ref().unwrap())
    }
}

struct MemoryPoolInner<T: MemoryType> {
    device: Device,

    ty: T,
    config: MemoryPoolConfig,
    blocks: Vec<Block<T>>,
}

impl<T: MemoryType> Drop for MemoryPoolInner<T> {
    fn drop(&mut self) {
        let device_read = self.device.read_inner();

        for block in self.blocks.drain(..) {
            let mut block_write = block.inner.write();
            if let Some(mem) = block_write.memory.take() {
                unsafe { device_read.raw.free_memory(mem) }
            };
        }
    }
}

impl<T: MemoryType> MemoryPoolInner<T> {
    /// Unconditionally attempts to allocate `num_blocks` blocks.
    ///
    /// The current number of blocks should be checked against
    /// `config.max_blocks` before calling this method.
    fn alloc_blocks(&mut self, num_blocks: u32) -> vks::VkResult<()> {
        let device_read = self.device.read_inner();

        let info = vk::MemoryAllocateInfoBuilder::new()
            .allocation_size(self.config.block_size.into())
            .memory_type_index(self.ty.type_index());

        self.blocks.reserve(num_blocks as usize);
        for _ in 0..num_blocks {
            let memory = unsafe { device_read.raw.allocate_memory(&info)? };

            let block = Block {
                inner: Arc::new(RwLock::new(BlockInner {
                    memory: Some(memory),
                    phantom: PhantomData,
                })),
            };
            self.blocks.push(block);
        }

        Ok(())
    }
}

/// Configuration values for a [`MemoryPool`].
///
/// Here, a **block** is the unit of memory that a pool allocates internally.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct MemoryPoolConfig {
    /// The size in bytes of a block.
    pub block_size: u32,

    /// The size in bytes of a chunk.
    pub alloc_size: u32,

    /// The number of blocks to allocate at pool creation.
    pub init_blocks: u32,

    /// The maximum number of blocks that this pool should ever acquire.
    ///
    /// If this value is `None`, then the limit is `u32::MAX`.
    pub max_blocks: Option<NonZeroU32>,
}

/// An allocator which yields fixed-size ranges of memory.
pub struct MemoryPool<T: MemoryType> {
    inner: Arc<RwLock<MemoryPoolInner<T>>>,
}

impl<T: MemoryType> MemoryPool<T> {
    pub unsafe fn new(device: Device, ty: T, config: MemoryPoolConfig) -> MemoryPool<T> {
        // TODO: un-panic these checks

        // Block size must be a power of two.
        assert_eq!(config.block_size.count_ones(), 1);

        // Chunk size must divide evenly into block size.
        assert_eq!(config.block_size % config.alloc_size, 0);

        // Init block count must be <= max block count.
        if let Some(max) = config.max_blocks {
            assert!(config.init_blocks <= max.get());
        }

        let mut inner = MemoryPoolInner {
            device,
            ty,
            config,
            blocks: Vec::with_capacity(config.init_blocks as usize),
        };

        if let Err(e) = inner.alloc_blocks(config.init_blocks) {
            log::error!("Failed to allocate initial memory pool blocks: {:?}", e);
            panic!("Failed to allocate initial memory pool blocks: {:?}", e);
            // TODO: free allocated blocks
        }

        MemoryPool {
            inner: Arc::new(RwLock::new(inner)),
        }
    }
}

mod private {
    pub trait Sealed {}
}
