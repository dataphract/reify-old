use std::convert::TryInto;

use erupt::vk;
use tinyvec::TinyVec;

use crate::vks::{self, VkObject};

/// The synchronization state of a resource during recording.
///
/// In order to read the resource, a render pass must:
/// - Record in `read_event_ids` the ID of an event which will be signaled
///   *after* the render pass has finished reading the resource.
/// - Ensure both of the following:
///   - `read_stage_mask` contains the pipeline stage flags indicating the
///      pipeline stages in which the render pass will read the resource.
///   - `read_access_mask` contains the access flags indicating the nature of
///     the render pass's access to the resource.
///
///   If either mask is missing necessary flags, a memory dependency from
///   `write_stage_mask` and `write_access_mask` to the missing flags must be
///   inserted, after which `read_stage_mask` and `read_access_mask` must be
///   updated with the new flags.
/// - Record the contents of the render pass.
/// - Record an event signal operation for the event added to `read_event_ids`.
///
/// In order to write the resource, a render pass must:
/// - Record an event wait operation for each event specified in
///   `read_event_ids` with a memory dependency from
///   `write_stage_mask | read_stage_mask` and
///   `write_access_mask | read_access_mask` to the stage and access masks of
///   the intended write.
/// - Record the contents of the render pass.
/// - Clear `read_event_ids`, `read_stage_mask` and `read_access_mask`.
/// - Set `write_stage_mask` and `write_access_mask` to the relevant values for
///   the render pass.
pub struct ResourceSyncState {
    /// The stage mask of the most recent write.
    write_stage_mask: vk::PipelineStageFlagBits,
    /// The access mask of the most recent write.
    write_access_mask: vk::AccessFlagBits,

    /// The event IDs of all render passes that have read from this resource
    /// since the most recent write.
    // TODO: Events can be per-resource-read, rather than per-pass. This would
    // prevent an event wait operation from having to wait for an entire render
    // pass to complete; it could instead only wait for the relevant set of
    // pipeline stages.
    read_event_ids: TinyVec<[u16; 8]>,
    /// The stage mask of all reads that have occurred since the most recent
    /// write.
    read_stage_mask: vk::PipelineStageFlagBits,
    /// The access mask of all reads that have occurred since the most recent
    /// write.
    read_access_mask: vk::AccessFlagBits,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct FrameEventId {
    id: u16,
}

pub struct FrameEvents {
    events: Vec<vks::Event>,
    recorded_status: Vec<vks::EventStatus>,
}

impl FrameEvents {
    pub fn new(events: Vec<vks::Event>) -> FrameEvents {
        assert!(events.len() <= u16::MAX as usize + 1);

        FrameEvents {
            recorded_status: Vec::with_capacity(events.len()),
            events,
        }
    }

    pub fn reset(&mut self, device: &vks::Device) {
        for event in self.events.iter_mut() {
            unsafe {
                device.reset_event(event).expect("event reset failed");
            }
        }

        self.recorded_status.clear();
    }

    pub fn alloc(&mut self) -> FrameEventId {
        assert!(
            self.recorded_status.len() < self.events.len(),
            "no free events"
        );

        let id: u16 = self
            .recorded_status
            .len()
            .try_into()
            .expect("event ID overflowed");
        self.recorded_status.push(vks::EventStatus::Unsignaled);
        FrameEventId { id }
    }
}

/// A set of resources used together when recording a frame.
pub struct FrameContext {
    /// Indicates whether the frame context is available for recording.
    ///
    /// The fence is reset prior to queue submission and awaited before recording a new frame.
    available: vks::Fence,

    events: FrameEvents,

    primary_cmd_pool: vks::CommandPool,
    primary_cmd_buf: vks::CommandBuffer,
}

impl FrameContext {
    /// Destroy any Vulkan resources owned by this value.
    ///
    /// # Safety
    ///
    /// The value must be dropped after this call returns.
    unsafe fn destroy_with(&mut self, device: &vks::Device) {
        unsafe {
            device
                .wait_for_fences(&[*self.available.handle()], true, None)
                .unwrap();
            device.destroy_fence(self.available.take());
            device.destroy_command_pool(self.primary_cmd_pool.take());
        }
    }

    pub fn reset(&mut self, device: &vks::Device) {
        unsafe { device.wait_for_fences(&[*self.available.handle()], true, None) }.unwrap();

        self.events.reset(device);
        unsafe {
            device.reset_command_pool(
                &mut self.primary_cmd_pool,
                vk::CommandPoolResetFlags::empty(),
            )
        }
        .unwrap();

        todo!();
    }

    pub fn alloc_event(&mut self) -> FrameEventId {
        self.events.alloc()
    }
}
