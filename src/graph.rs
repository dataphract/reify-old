//! A render graph implementation.
//!
//! The resource model of this render graph is adapted from the single static
//! assignment (SSA) form typically used in compilers. Resources are considered
//! immutable; mutation of a physical resource (like writing to an existing
//! image) is represented by consuming the original image and producing a new
//! one.

use std::{collections::VecDeque, convert::TryInto, fmt, time::Instant};

use arrayvec::ArrayVec;
use erupt::vk;
use petgraph::{
    graph::NodeIndex,
    visit::{EdgeRef, IntoEdgeReferences, IntoNodeReferences, NodeRef},
    Directed,
};
use thiserror::Error;
use tinyvec::TinyVec;

use crate::{util::SmallSet, vks};

type PassGraph = petgraph::Graph<RenderPassId, DependencyType, Directed, u16>;

#[derive(Debug)]
pub enum AccessType {
    Consume,
    Read,
    Produce,
}

impl AccessType {
    fn perfect_tense(&self) -> &'static str {
        match self {
            AccessType::Consume => "consumed",
            AccessType::Read => "read",
            AccessType::Produce => "produced",
        }
    }
}

#[derive(Error, Debug)]
pub enum RenderGraphError {
    #[error("No resource with ID {0}")]
    NoSuchResource(ResourceId),
    #[error("No render pass with ID {0}")]
    NoSuchRenderPass(RenderPassId),
    #[error("Incompatible resource type: expected {expected:?}, was {actual:?}")]
    IncompatibleResourceType {
        expected: ResourceTypeTag,
        actual: ResourceTypeTag,
    },
    #[error(
        "Render pass {pass_name:?} depends on resource {res_name:?} \
             (ID = {res_id}), which it produces."
    )]
    SelfLoop {
        pass_name: String,
        res_name: String,
        res_id: ResourceId,
    },
    #[error(
        "Resource {r_name:?} (ID = {r_id}) already {} by render pass {p_name:?}. \
         Resources may be accessed at most once by a given render pass.",
        ty.perfect_tense()
    )]
    AlreadyAccessed {
        ty: AccessType,
        r_name: String,
        r_id: ResourceId,
        p_name: String,
    },
    #[error("Final image size must be SAME_AS_SWAPCHAIN.")]
    FinalImageSize(ImageSize),
    #[error("Render graph has no swapchain image.")]
    MissingSwapchainImage,
    #[error(
        "Attempted to set swapchain image to {new_name:?}, \
         but was already set to {old_name:?} (ID = {old_id})."
    )]
    AlreadySetSwapchainImage {
        old_name: String,
        old_id: ResourceId,
        new_name: String,
    },
    #[error("No render passes write to the swapchain image.")]
    SwapchainNotWritten,
    #[error("Render pass {pass_name:?} depends on itself")]
    DependencyCycle {
        // TODO: print the cycle
        pass_name: String,
    },
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ResourceId {
    id: u16,
}

impl Default for ResourceId {
    fn default() -> Self {
        ResourceId { id: u16::MAX }
    }
}

impl fmt::Display for ResourceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.id)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct RenderPassId {
    id: u16,
}

impl fmt::Display for RenderPassId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.id)
    }
}

impl Default for RenderPassId {
    fn default() -> Self {
        RenderPassId { id: u16::MAX }
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct RelativeExtent {
    width: f32,
    height: f32,
    depth: f32,
}

impl RelativeExtent {
    pub const ONE: Self = RelativeExtent {
        width: 1.0,
        height: 1.0,
        depth: 1.0,
    };
}

impl Default for RelativeExtent {
    fn default() -> Self {
        RelativeExtent {
            width: 1.0,
            height: 1.0,
            depth: 1.0,
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub enum ImageSize {
    Absolute(vk::Extent3D),
    RelativeToSwapchain(RelativeExtent),
    RelativeToInput(RelativeExtent),
}

impl PartialEq for ImageSize {
    fn eq(&self, other: &Self) -> bool {
        use ImageSize::*;
        match (self, other) {
            (Absolute(a), Absolute(b)) => {
                a.width == b.width && a.height == b.height && a.depth == b.depth
            }
            (RelativeToSwapchain(a), RelativeToSwapchain(b)) => a == b,
            (RelativeToInput(a), RelativeToInput(b)) => a == b,
            _ => false,
        }
    }
}

impl ImageSize {
    pub const SAME_AS_SWAPCHAIN: Self = ImageSize::RelativeToSwapchain(RelativeExtent::ONE);
}

#[derive(Clone, Debug, PartialEq)]
pub struct ImageInfo {
    pub size: ImageSize,
    pub format: vk::Format,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BufferInfo {}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ResourceTypeTag {
    Image,
    Buffer,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ResourceType {
    Image(ImageInfo),
    Buffer(BufferInfo),
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct ResourceConsumed {
    pass: RenderPassId,
    successor: ResourceId,
}

struct Resource {
    ty: ResourceType,

    // The lifetime of a resource spans from  `produced_by` to `consumed_by`.
    produced_by: Option<RenderPassId>,
    read_by: TinyVec<[RenderPassId; 4]>,
    consumed_by: Option<RenderPassId>,
}

impl Resource {
    fn image(info: ImageInfo) -> Resource {
        Resource {
            ty: ResourceType::Image(info),
            produced_by: None,
            read_by: TinyVec::new(),
            consumed_by: None,
        }
    }

    fn image_info(&self) -> Result<&ImageInfo, RenderGraphError> {
        match &self.ty {
            ResourceType::Image(info) => Ok(info),
            ResourceType::Buffer(_) => Err(RenderGraphError::IncompatibleResourceType {
                expected: ResourceTypeTag::Image,
                actual: ResourceTypeTag::Buffer,
            }),
        }
    }
}

enum AttachmentType {
    Input,
    Color,
    DepthStencil,
}

/// Indicates the method used to initialize a resource.
pub enum ResourceInit<T> {
    /// The resource will be initialized either by clearing or with undefined
    /// contents.
    New(T),
    /// The resource will be initialized using the value of another resource.
    ///
    /// The other resource will be consumed.
    Consume(ResourceId),
}

// The vast majority of implementations have a limit of 8 color attachments. Add
// one depth buffer, plus a few more slots for good measure.
const MAX_PRODUCED_RESOURCES: usize = 12;

pub struct RenderPassBuilder<'a> {
    /// The name of the render pass.
    name: String,

    // Resource IDs cannot be allocated directly from the graph, as dropping
    // the builder would leave the graph in an inconsistent state. Instead, any
    // newly created resources are stored locally until the pass is finished.
    //
    /// The base resource ID of this pass.
    base_resource_id: ResourceId,
    /// The resources produced by this builder.
    produced: ArrayVec<ResourceType, MAX_PRODUCED_RESOURCES>,
    produced_names: ArrayVec<String, MAX_PRODUCED_RESOURCES>,

    graph: &'a mut RenderGraphBuilder,
    pass: RenderPassNode,
}

impl<'a> RenderPassBuilder<'a> {
    #[inline]
    fn local_id(&self, id: ResourceId) -> Option<u16> {
        id.id.checked_sub(self.base_resource_id.id)
    }

    fn check_self_loop(&self, id: ResourceId) -> Result<(), RenderGraphError> {
        if let Some(local_id) = self.local_id(id) {
            // This read would cause a self-loop.
            return Err(RenderGraphError::SelfLoop {
                pass_name: self.name.clone(),
                res_name: self.produced_names[local_id as usize].clone(),
                res_id: id,
            });
        }

        Ok(())
    }

    fn add_consume(&mut self, id: ResourceId) -> Result<(), RenderGraphError> {
        self.check_self_loop(id)?;
        let name = self.graph.resource_name(id).unwrap();

        if self.pass.reads.contains(&id) {
            return Err(RenderGraphError::AlreadyAccessed {
                ty: AccessType::Read,
                r_name: name.to_owned(),
                r_id: id,
                p_name: self.name.clone(),
            });
        }

        if !self.pass.consumes.insert(id) {
            return Err(RenderGraphError::AlreadyAccessed {
                ty: AccessType::Consume,
                r_name: name.to_owned(),
                r_id: id,
                p_name: self.name.clone(),
            });
        }

        Ok(())
    }

    fn add_read(&mut self, id: ResourceId) -> Result<(), RenderGraphError> {
        self.check_self_loop(id)?;
        let name = self.graph.resource_name(id).unwrap();

        if self.pass.consumes.contains(&id) {
            return Err(RenderGraphError::AlreadyAccessed {
                ty: AccessType::Consume,
                r_name: name.to_owned(),
                r_id: id,
                p_name: self.name.clone(),
            });
        }

        if !self.pass.reads.insert(id) {
            return Err(RenderGraphError::AlreadyAccessed {
                ty: AccessType::Read,
                r_name: name.to_owned(),
                r_id: id,
                p_name: self.name.clone(),
            });
        }

        Ok(())
    }

    fn add_produce<S: AsRef<str>>(
        &mut self,
        name: S,
        ty: ResourceType,
    ) -> Result<ResourceId, RenderGraphError> {
        let id = ResourceId {
            id: self.base_resource_id.id + self.produced.len() as u16,
        };

        // TODO: handle arrayvec panics
        self.produced.push(ty);
        self.produced_names.push(name.as_ref().into());

        // Cannot already exist, as the resource is newly created.
        self.pass.produces.insert(id);

        Ok(id)
    }

    pub fn add_input_attachment(&mut self, id: ResourceId) -> Result<(), RenderGraphError> {
        self.add_read(id)?;

        self.pass.input_attachments.push(id);

        Ok(())
    }

    /// Adds a color attachment to the render pass.
    ///
    /// If `consumes` is `Some(c)`, then `c` is the ID of an image resource
    /// whose value will be used to initialize the image resource identified by
    /// `id`. The consumed resource may not be used again.
    pub fn add_color_attachment<S: AsRef<str>>(
        &mut self,
        name: S,
        info: ImageInfo,
        consumes: Option<ResourceId>,
    ) -> Result<ResourceId, RenderGraphError> {
        let id = self.add_produce(name, ResourceType::Image(info)).unwrap();

        // TODO: sanity-check produce and consume info

        self.pass.color_attachments.push(ColorAttachment {
            consumed: consumes,
            produced: id,
        });

        if let Some(c) = consumes {
            self.add_consume(c)?;
        }

        Ok(id)
    }

    pub fn finish(mut self) -> RenderPassId {
        let id = RenderPassId {
            id: self
                .graph
                .passes
                .len()
                .try_into()
                .expect("passes overflowed"),
        };

        for (res, name) in self.produced.drain(..).zip(self.produced_names.drain(..)) {
            self.graph.add_resource(name, res);
        }

        for consume in self.pass.consumes.iter().copied() {
            self.graph.add_consume(consume, id);
        }

        for read in self.pass.reads.iter().copied() {
            self.graph.add_read(read, id);
        }

        for write in self.pass.produces.iter().copied() {
            self.graph.add_produce(write, id);
        }

        self.graph.passes.push(self.pass);
        self.graph.pass_names.push(self.name);

        id
    }
}

#[derive(Default)]
struct ColorAttachment {
    consumed: Option<ResourceId>,
    produced: ResourceId,
}

const EXPECTED_CONSUMES: usize = 4;
const EXPECTED_READS: usize = 4;
const EXPECTED_PRODUCES: usize = 4;

pub struct RenderPassNode {
    // TODO: Ideally, avoid boxing render passes.
    pass: Box<dyn RenderPass>,

    input_attachments: TinyVec<[ResourceId; 4]>,
    color_attachments: TinyVec<[ColorAttachment; 4]>,

    // Associated resources by access type.
    //
    // The number of resources used by a given pass is expected to be fairly
    // small, so linear scans over inline arrays should be significantly faster
    // than using e.g. HashSets, which require an alloc.
    consumes: SmallSet<ResourceId, EXPECTED_CONSUMES>,
    reads: SmallSet<ResourceId, EXPECTED_READS>,
    produces: SmallSet<ResourceId, EXPECTED_PRODUCES>,

    // Index of the node in the dependency graph.
    node_idx: Option<NodeIndex<u16>>,
}

/// A dependency between render passes.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum DependencyType {
    /// Render pass A produces a resource needed by render pass B.
    Produce(ResourceId),
    /// Render pass A needs a resource that will be consumed by render pass B.
    Consume(ResourceId),
}

#[derive(Default)]
pub struct RenderGraphBuilder {
    resources: Vec<Resource>,
    resource_names: Vec<String>,

    passes: Vec<RenderPassNode>,
    pass_names: Vec<String>,

    final_image: Option<ResourceId>,
}

impl RenderGraphBuilder {
    pub fn new() -> RenderGraphBuilder {
        Default::default()
    }

    fn resource(&self, id: ResourceId) -> Result<&Resource, RenderGraphError> {
        self.resources
            .get(id.id as usize)
            .ok_or(RenderGraphError::NoSuchResource(id))
    }

    fn resource_mut(&mut self, id: ResourceId) -> Result<&mut Resource, RenderGraphError> {
        self.resources
            .get_mut(id.id as usize)
            .ok_or(RenderGraphError::NoSuchResource(id))
    }

    #[inline]
    fn resource_name(&self, id: ResourceId) -> Option<&str> {
        self.resource_names.get(id.id as usize).map(String::as_str)
    }

    fn render_pass(&self, id: RenderPassId) -> Result<&RenderPassNode, RenderGraphError> {
        self.passes
            .get(id.id as usize)
            .ok_or(RenderGraphError::NoSuchRenderPass(id))
    }

    fn render_pass_mut(
        &mut self,
        id: RenderPassId,
    ) -> Result<&mut RenderPassNode, RenderGraphError> {
        self.passes
            .get_mut(id.id as usize)
            .ok_or(RenderGraphError::NoSuchRenderPass(id))
    }

    fn render_pass_name(&self, id: RenderPassId) -> Option<&str> {
        self.pass_names.get(id.id as usize).map(String::as_str)
    }

    fn add_resource<S: AsRef<str>>(&mut self, name: S, ty: ResourceType) -> ResourceId {
        let id = ResourceId {
            id: self
                .resources
                .len()
                .try_into()
                .expect("resources overflowed"),
        };

        self.resources.push(Resource {
            ty,
            produced_by: None,
            read_by: TinyVec::new(),
            consumed_by: None,
        });

        self.resource_names.push(name.as_ref().to_owned());

        id
    }

    #[inline]
    pub fn add_image<S: AsRef<str>>(&mut self, name: S, info: ImageInfo) -> ResourceId {
        self.add_resource(name, ResourceType::Image(info))
    }

    pub fn set_final_image(&mut self, id: ResourceId) -> Result<(), RenderGraphError> {
        if let Some(old_id) = self.final_image {
            return Err(RenderGraphError::AlreadySetSwapchainImage {
                old_id,
                old_name: self.resource_name(old_id).unwrap().into(),
                new_name: self.resource_name(id).unwrap().into(),
            });
        }

        self.final_image = Some(id);
        Ok(())
    }

    #[inline]
    fn add_produce(&mut self, resource: ResourceId, pass: RenderPassId) {
        let res = &mut self.resources[resource.id as usize];
        assert!(res.produced_by.is_none());
        res.produced_by.replace(pass);
    }

    #[inline]
    fn add_read(&mut self, resource: ResourceId, pass: RenderPassId) {
        let res = &mut self.resources[resource.id as usize];
        res.read_by.push(pass);
    }

    #[inline]
    fn add_consume(&mut self, resource: ResourceId, pass: RenderPassId) {
        let res = &mut self.resources[resource.id as usize];
        assert!(res.consumed_by.is_none());
        res.consumed_by.replace(pass);
    }

    #[inline]
    pub fn add_render_pass<'a, S, R>(&'a mut self, name: S, pass: R) -> RenderPassBuilder<'a>
    where
        S: AsRef<str>,
        R: RenderPass + 'static,
    {
        RenderPassBuilder {
            name: name.as_ref().to_owned(),
            base_resource_id: ResourceId {
                id: self.resources.len().try_into().unwrap(),
            },
            produced: ArrayVec::new(),
            produced_names: ArrayVec::new(),
            graph: self,
            pass: RenderPassNode {
                pass: Box::new(pass),
                input_attachments: TinyVec::new(),
                color_attachments: TinyVec::new(),
                consumes: SmallSet::new(),
                reads: SmallSet::new(),
                produces: SmallSet::new(),
                node_idx: None,
            },
        }
    }

    fn gen_dotgraph(&self, graph: &PassGraph) -> String {
        use petgraph::dot;

        fn pass_attributes<G>(builder: &RenderGraphBuilder) -> impl Fn(G, G::NodeRef) -> String + '_
        where
            G: IntoNodeReferences + IntoEdgeReferences,
            G::NodeRef: NodeRef<Weight = RenderPassId>,
        {
            move |graph: G, node: G::NodeRef| {
                let pass_id = node.weight();
                let name = builder.render_pass_name(*pass_id).unwrap();
                format!("label=\"{}\"", name)
            }
        }

        fn resource_attributes<G>(
            builder: &RenderGraphBuilder,
        ) -> impl Fn(G, G::EdgeRef) -> String + '_
        where
            G: IntoNodeReferences + IntoEdgeReferences,
            G::EdgeRef: EdgeRef<Weight = DependencyType>,
        {
            move |graph: G, edge: G::EdgeRef| match *edge.weight() {
                DependencyType::Produce(res_id) => {
                    let name = builder.resource_name(res_id).unwrap();
                    format!("label=\"Produce: {}\"", name)
                }

                DependencyType::Consume(res_id) => {
                    let name = builder.resource_name(res_id).unwrap();
                    format!("label=\"Consume: {}\"", name)
                }
            }
        }

        let get_pass_attributes = pass_attributes(self);
        let get_resource_attributes = resource_attributes(self);

        let dot = petgraph::dot::Dot::with_attr_getters(
            &graph,
            &[dot::Config::NodeNoLabel, dot::Config::EdgeNoLabel],
            &get_resource_attributes,
            &get_pass_attributes,
        );

        format!("{:?}", dot)
    }

    pub fn build(mut self) -> Result<(), RenderGraphError> {
        let final_image_id = self
            .final_image
            .clone()
            .ok_or(RenderGraphError::MissingSwapchainImage)?;
        let final_image = self.resource(final_image_id).unwrap();

        // Build a dependency graph of passes.
        //
        // - Each node is a render pass.
        // - Each edge is a resource dependency. There are two kinds of dependency:
        //   - Produce-dependencies, in which pass A produces resource R and
        //     pass B reads or consumes R. This requires both an execution
        //     barrier, to ensure A happens-before B, and a memory barrier to
        //     ensure the data written by A is made visible.
        //   - Consume-dependencies, in which pass A reads resource R and pass B
        //     consumes R. This only requires an execution barrier to ensure A
        //     happens-before B (write-after-read hazards are precluded by
        //     execution barriers in Vulkan).
        let mut graph = PassGraph::with_capacity(self.passes.len(), self.resources.len());

        // Get the pass which produces the final image.
        let final_pass_id = final_image
            .produced_by
            .ok_or(RenderGraphError::SwapchainNotWritten)?;

        // Enqueue all render passes that introduce produce-dependencies.
        // TODO: holodeque me?
        let mut next_depth: Vec<RenderPassId> = Vec::new();
        let mut cur_depth: SmallSet<RenderPassId, 16> = SmallSet::new();

        // Walk the graph in reverse, breadth-first. Process all nodes at depth N at once.
        next_depth.push(final_pass_id);

        let start_produce_insert = Instant::now();
        while !next_depth.is_empty() {
            cur_depth.clear();
            for pass_id in next_depth.drain(..) {
                if !cur_depth.contains(&pass_id) {
                    let pass = self.render_pass_mut(pass_id).unwrap();
                    if pass.node_idx.is_none() {
                        // Only consider passes that have not already been visited.
                        cur_depth.insert(pass_id);

                        pass.node_idx = Some(graph.add_node(pass_id));
                    }
                }
            }

            for pass_id in cur_depth.iter().copied() {
                let pass = self.render_pass(pass_id).unwrap();
                let pass_idx = pass.node_idx.unwrap();

                // Insert produce-dependencies. Since entire depths are inserted
                // into the graph at once, this will not miss lateral edges (i.e.,
                // edges from one depth-N node to another). Any node at this depth
                // which does not already exist in the graph is not a dependency of
                // the terminal node.
                for produce_id in pass.produces.iter().copied() {
                    let produce = self.resource(produce_id).unwrap();

                    for dependent_id in produce
                        .read_by
                        .iter()
                        .copied()
                        .chain(produce.consumed_by.iter().copied())
                    {
                        let dependent = self.render_pass(dependent_id).unwrap();
                        if let Some(dependent_idx) = dependent.node_idx {
                            // If the dependent has not been inserted, it is not
                            // a dependency.
                            graph.add_edge(
                                pass_idx,
                                dependent_idx,
                                DependencyType::Produce(produce_id),
                            );
                        }
                    }
                }

                // Enqueue next depth of the graph.
                for input_id in pass.reads.iter().chain(pass.consumes.iter()).copied() {
                    let input = self.resource(input_id).unwrap();
                    let producer = input.produced_by.unwrap();
                    next_depth.push(producer);
                }
            }
        }

        log::debug!(
            "Produce-dependencies inserted in {}μs",
            start_produce_insert.elapsed().as_micros()
        );

        // The graph now contains all necessary render passes and all
        // produce-dependencies between them.

        let start_consume_insert = Instant::now();
        // TODO: this avoids a mutable borrow error on the graph, but it
        // shouldn't be necessary -- no nodes are added or removed, only edges.
        let mut passes = next_depth;
        passes.extend(graph.node_weights().copied());

        // Insert consume-dependencies.
        // TODO: cache a list of consumers (expected to be relatively small) and
        //   insert these dependencies to any readers of the consumed resource,
        //   rather than scanning every read resource.
        for pass_id in passes.iter().copied() {
            let pass = self.render_pass(pass_id).unwrap();

            for read_id in pass.reads.iter().copied() {
                let read = self.resource(read_id).unwrap();

                if let Some(consumed_by_id) = read.consumed_by {
                    let consumed_by = self.render_pass(consumed_by_id).unwrap();
                    graph.add_edge(
                        pass.node_idx.unwrap(),
                        consumed_by.node_idx.unwrap(),
                        DependencyType::Consume(read_id),
                    );
                }
            }
        }

        log::debug!(
            "Consume-dependencies inserted in {}μs",
            start_consume_insert.elapsed().as_micros()
        );

        println!("{}", self.gen_dotgraph(&graph));

        let start_dep_resolve = Instant::now();
        let ordered = match petgraph::algo::toposort(&graph, None) {
            Ok(o) => o,
            Err(cycle) => {
                let pass_id = *graph.node_weight(cycle.node_id()).unwrap();
                return Err(RenderGraphError::DependencyCycle {
                    pass_name: self.render_pass_name(pass_id).unwrap().into(),
                });
            }
        };
        log::debug!(
            "Dependencies resolved in {}μs",
            start_dep_resolve.elapsed().as_micros()
        );

        todo!("physical resource assignment");
    }
}

// =============================================================================

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum ClearColorValue {
    Float32([f32; 4]),
    Int32([i32; 4]),
    Uint32([u32; 4]),
}

impl From<[f32; 4]> for ClearColorValue {
    fn from(f: [f32; 4]) -> Self {
        ClearColorValue::Float32(f)
    }
}

impl From<[i32; 4]> for ClearColorValue {
    fn from(i: [i32; 4]) -> Self {
        ClearColorValue::Int32(i)
    }
}

impl From<[u32; 4]> for ClearColorValue {
    fn from(u: [u32; 4]) -> Self {
        ClearColorValue::Uint32(u)
    }
}

impl From<ClearColorValue> for vk::ClearColorValue {
    fn from(val: ClearColorValue) -> Self {
        use ClearColorValue::*;

        match val {
            Float32(f) => vk::ClearColorValue { float32: f },
            Int32(i) => vk::ClearColorValue { int32: i },
            Uint32(u) => vk::ClearColorValue { uint32: u },
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ClearDepthStencilValue {
    depth: f32,
    stencil: u32,
}

impl From<ClearDepthStencilValue> for vk::ClearDepthStencilValue {
    fn from(val: ClearDepthStencilValue) -> Self {
        vk::ClearDepthStencilValue {
            depth: val.depth,
            stencil: val.stencil,
        }
    }
}

// =============================================================================

pub trait RenderPass {
    /// Returns the value used to clear color attachments.
    ///
    /// If `None`, then the initial contents of the attachment are undefined.
    fn clear_color_value(&self) -> Option<ClearColorValue> {
        None
    }

    /// Returns the value used to clear depth/stencil attachments.
    ///
    /// If `None`, then the initial contents of the attachment are undefined.
    fn clear_depth_stencil_value(&self) -> Option<ClearDepthStencilValue> {
        None
    }

    fn record(&self, device: &vks::Device, cmdbuf: &mut vks::CommandBuffer);
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DummyPass;

    impl RenderPass for DummyPass {
        fn record(&self, device: &vks::Device, cmdbuf: &mut vks::CommandBuffer) {}
    }

    const DUMMY_COLOR: ImageInfo = ImageInfo {
        size: ImageSize::SAME_AS_SWAPCHAIN,
        format: vk::Format::B8G8R8A8_SRGB,
    };

    // Invariants:
    // - Resources must be produced exactly once.
    // - Resources may be consumed at most once.
    // - Render passes may not read resources they produce.
    // - Render passes may not read the same resource multiple times.
    // - Accesses recorded in a RenderPassBuilder are recorded to the relevant
    //   resources when the builder is finished.
    // - If a RenderPassBuilder is dropped, none of its produced resources are
    //   retained by the render graph.
    // - ResourceIds produced by a dropped RenderPassBuilder cannot be used to
    //   access any resource.
    // - ResourceIds produced by a RenderPassBuilder refer to the same resource
    //   after the builder is finished.
    // - The dependency graph may not have cycles.

    #[test]
    fn add_color_attachment_sets_produced_by() {
        let mut graph = RenderGraphBuilder::new();

        let mut pass = graph.add_render_pass("main pass", DummyPass);
        let color_attachment = pass
            .add_color_attachment("color attachment", DUMMY_COLOR, None)
            .unwrap();
        let pass = pass.finish();

        let res = graph.resource(color_attachment).unwrap();
        assert_eq!(res.produced_by, Some(pass));
    }

    #[test]
    fn basic() {
        let mut graph = RenderGraphBuilder::new();

        let mut pass = graph.add_render_pass("main pass", DummyPass);
        let color_attachment = pass
            .add_color_attachment("color attachment", DUMMY_COLOR, None)
            .unwrap();
        pass.finish();

        todo!();
    }
}
