## Image classes

- Textures
  - Usage: `TRANSFER_DST | SAMPLED`
  - Format:
    - B8G8R8A8_SRGB for color data
    - R8(G8(B8(A8)))\_{S,U}NORM for non-color data
- Attachments
  - Usage: 
    - `COLOR_ATTACHMENT` for color
    - `DEPTH_STENCIL_ATTACHMENT` for depth/stencil
    - `INPUT_ATTACHMENT` on some or all attachments
    - `TRANSIENT_ATTACHMENT` on a subset for attachments between subpasses
  - Format:
    - B8G8R8A8_SRGB for color data
    - R8(G8(B8(A8)))\_{S,U}NORM for non-color data

## Synchronization of attachments

- Produce an attachment:
  a. If this output consumes a previous image, all previous reads of the image must have completed.
     This is ensured by inserting a memory dependency between each previous read and the
     `COLOR_ATTACHMENT_OUTPUT`/`DEPTH_STENCIL_ATTACHMENT_OUTPUT` stage of this output.

     The most practical option is to maintain a set of events per render pass. If a render pass A
     reads a resource R which is later consumed by render pass B, it signals its associated event
     after vkCmdEndRenderPass is recorded. Before render pass B is begun, a vkCmdWaitEvents must
     be recorded which waits for every event associated with R to be signaled.
     
     The events must be reset by the host between frames.
     
     Ownership of the consumed resource's associated physical resource is moved to this produced
     resource.
     
     As a validation step, the render graph should track locally whether events have been awaited.
     No signaled event should be un-awaited at the end of render graph compilation.

  b. If this output does not consume a previous image, a suitable image resource is acquired from
     the pool. No barrier is necessary; its initial layout is UNDEFINED.
     
- Read an attachment:
  a. If this read uses an image produced by a previous render pass, then a memory dependency must be
     inserted which makes visible the data written by that render pass.

  b. If this read uses an image produced externally, the image metadata must indicate what sort of
     memory dependency, if any, is required to synchronize access to the image.
     
  c. After the render pass is finished, the corresponding event must be signaled to indicate that
     the image is no longer being read.
     
## Resource sum types

- Multiple resources are permitted to alias, provided that they are not used simultaneously in ways
  which would cause race conditions or access to incompatible formats.
- This allows for the creation of *resource sum types*, which are unions of aliasing resources.
- These unions are tagged on the host. The tag indicates which of the aliasing resources was most
  recently used.
- As only one alias may be in use at a time, all aliases may share a synchronization state.
- The synchronization state and union tag may be used together to transfer ownership of the aliased
  memory between variants. This requires the previous resource to no longer be needed, i.e., it is
  not scheduled for consumption by any later render pass, nor do any render passes need to read it.
- To switch variants, all outstanding writes must be made visible by inserting a memory dependency.
  The resource can then be initialized for the new variant, e.g. by executing a layout transition
  from UNDEFINED for a new image variant.
