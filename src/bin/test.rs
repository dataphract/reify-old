use std::time::{Duration, Instant};

use erupt::vk;
use raw_window_handle::HasRawWindowHandle;
use reify::{
    graph::{ClearColorValue, ImageInfo, ImageSize, RenderGraphBuilder, RenderPass},
    Instance, MemoryConfig,
};
use shaderc::{Compiler, ShaderKind};
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

pub struct TrianglePass;

impl RenderPass for TrianglePass {
    fn clear_color_value(&self) -> Option<ClearColorValue> {
        Some(ClearColorValue::Float32([0.0, 0.0, 0.0, 1.0]))
    }

    fn record(&self, device: &reify::vks::Device, cmdbuf: &mut reify::vks::CommandBuffer) {
        todo!()
    }
}

fn build_graph() {
    let mut graph = RenderGraphBuilder::new();

    let mut pass_a = graph.add_render_pass("pass A", TrianglePass);
    let a_out = pass_a
        .add_color_attachment(
            "pass A color",
            ImageInfo {
                size: ImageSize::SAME_AS_SWAPCHAIN,
                format: vk::Format::B8G8R8A8_SRGB,
            },
            None,
        )
        .unwrap();
    pass_a.finish();

    let mut pass_b = graph.add_render_pass("pass B", TrianglePass);
    pass_b.add_input_attachment(a_out).unwrap();
    let b_out = pass_b
        .add_color_attachment(
            "pass B color",
            ImageInfo {
                size: ImageSize::SAME_AS_SWAPCHAIN,
                format: vk::Format::B8G8R8A8_SRGB,
            },
            None,
        )
        .unwrap();
    pass_b.finish();

    let mut pass_c = graph.add_render_pass("pass C", TrianglePass);
    pass_c.add_input_attachment(b_out).unwrap();
    let c_out = pass_c
        .add_color_attachment(
            "pass C color",
            ImageInfo {
                size: ImageSize::SAME_AS_SWAPCHAIN,
                format: vk::Format::B8G8R8A8_SRGB,
            },
            Some(a_out),
        )
        .unwrap();
    pass_c.finish();

    graph.set_final_image(c_out).unwrap();
    graph.build().unwrap();

    todo!();
}

pub fn main() {
    env_logger::init();

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new().build(&event_loop).unwrap();

    let instance = Instance::create("reify", 0);
    let _debug_messenger = instance.create_debug_messenger();
    let surface = instance.create_surface(window.raw_window_handle());
    let phys_device = match instance
        .enumerate_physical_devices(
            &surface,
            MemoryConfig {
                min_host_memory: 128 * 1024 * 1024,
                min_device_memory: 128 * 1024 * 1024,
            },
        )
        .into_iter()
        .find(|_phys| true)
    {
        Some(phys) => phys,
        None => panic!("no suitable device"),
    };

    let device = phys_device.create_device();
    let phys_size = window.inner_size();
    let mut display = unsafe {
        device.create_display(
            surface,
            vk::Extent2D {
                width: phys_size.width,
                height: phys_size.height,
            },
        )
    };

    let mem_types = phys_device.memory_types();
    println!("Memory types: {:#?}", mem_types);

    let mut compiler = Compiler::new().expect("failed to initialize shaderc");

    let vert_glsl = include_str!("../../shaders/vert.glsl");
    let frag_glsl = include_str!("../../shaders/frag.glsl");

    let vert_spv = compiler
        .compile_into_spirv(vert_glsl, ShaderKind::Vertex, "vert.glsl", "main", None)
        .unwrap();
    let frag_spv = compiler
        .compile_into_spirv(frag_glsl, ShaderKind::Fragment, "frag.glsl", "main", None)
        .unwrap();

    let pipeline =
        unsafe { device.create_pipeline(vert_spv.as_binary(), frag_spv.as_binary(), &display) };

    {
        let pipeline_read = pipeline.read_inner();
        display.rebuild_framebuffers(pipeline.read_inner().render_pass());
        display.record_command_buffers(pipeline_read.render_pass(), pipeline_read.pipeline());
    }

    let mut frames = 0;
    let mut last_frame = Instant::now();
    event_loop.run(move |event, _target, flow| {
        *flow = winit::event_loop::ControlFlow::Poll;

        match event {
            Event::MainEventsCleared => {
                if last_frame.elapsed() > Duration::from_micros(16666) {
                    last_frame = Instant::now();
                    display.draw();
                    log::debug!("frame {}: draw complete.", frames);
                    frames += 1;
                } else {
                    std::thread::sleep(Duration::from_millis(1));
                }
            }

            Event::WindowEvent {
                window_id: _,
                event,
            } => match event {
                WindowEvent::CloseRequested => *flow = ControlFlow::Exit,
                _ => (),
            },
            _ => (),
        }
    });
}
