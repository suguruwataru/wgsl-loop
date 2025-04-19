#[cfg(test)]
mod test {
    use pollster::FutureExt;

    #[test]
    fn atomic_exchange_weak() {
        test_shader("atomic_compare_exchange_weak");
    }

    fn test_shader(entry_point: &str) {
        let wgpu_instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::GL,
            ..Default::default()
        });
        let adapter = wgpu_instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .block_on()
            .unwrap();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .block_on()
            .unwrap();
        device.on_uncaptured_error(Box::new(|error| {
            println!("{}", error);
            panic!();
        }));
        let shader_module = device.create_shader_module(wgpu::include_wgsl!("loop.wgsl"));
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            module: &shader_module,
            label: None,
            layout: None,
            entry_point: Some(entry_point),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&pipeline);
            pass.dispatch_workgroups(1, 1, 1);
        }
        let idx = queue.submit([encoder.finish()]);
        device
            .poll(wgpu::PollType::WaitForSubmissionIndex(idx))
            .unwrap();
    }
}
