#![deny(unsafe_op_in_unsafe_fn)]

use ash::vk;
use reify::Instance;
use winit::{
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

pub fn main() {
    env_logger::init();

    let event_loop = EventLoop::new();

    let _window = WindowBuilder::new().build(&event_loop).unwrap();

    let instance = Instance::create("reify", 0);

    let debug_messenger = instance.create_debug_messenger();

    let phys_device = match instance
        .enumerate_physical_devices()
        .into_iter()
        .find(|_phys| true)
    {
        Some(phys) => phys,
        None => panic!("no suitable device"),
    };

    let device = phys_device.create_device();

    event_loop.run(move |event, _target, flow| match event {
        winit::event::Event::WindowEvent {
            window_id: _,
            event,
        } => match event {
            winit::event::WindowEvent::CloseRequested => *flow = ControlFlow::Exit,
            _ => (),
        },
        _ => (),
    });
}
