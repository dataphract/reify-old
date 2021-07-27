use std::iter::FromIterator;

use arrayvec::ArrayVec;
use erupt::vk;

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
struct HostMemory {
    heap: u32,
    is_host_coherent: bool,
    is_host_cached: bool,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct DeviceMemory {
    heap: u32,
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
        let mut host: Option<HostMemory> = None;

        // Device memory is used for the majority of a resource's lifetime.
        // It must have the DEVICE_LOCAL property. Memory with any HOST_*
        // properties should be avoided where possible.
        let mut device: Option<DeviceMemory> = None;

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
                        let larger_heap = self.heap_size(heap_id) > self.heap_size(old.heap);

                        if better_host_cached || same_host_cached && larger_heap {
                            host = Some(HostMemory {
                                heap: memty.heap_index,
                                is_host_coherent,
                                is_host_cached,
                            });
                        }
                    }

                    None => {
                        host = Some(HostMemory {
                            heap: memty.heap_index,
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

                        let larger_heap = self.heap_size(heap_id) > self.heap_size(old.heap);

                        if better_host_visible
                            || better_host_coherent
                            || (same_host_visible && same_host_coherent && larger_heap)
                        {
                            device = Some(DeviceMemory {
                                heap: heap_id,
                                is_host_visible,
                                is_host_coherent,
                            });
                        }
                    }

                    None => {
                        device = Some(DeviceMemory {
                            heap: heap_id,
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
    host: HostMemory,
    device: DeviceMemory,
}
