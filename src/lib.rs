#[cfg(test)]
mod test {
    use pollster::FutureExt;

    #[test]
    fn atomic_exchange_weak() {
        test_shader("atomic_compare_exchange_weak");
    }

    fn test_shader(entry_point: &str) {
        let wgpu_instance = wgpu::Instance::default();
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
        use wgpu::util::DeviceExt;
        let src_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: &[0u8; 8],
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE,
        });
        let dst_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 8,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[wgpu::BindGroupEntry {
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &src_buf,
                        size: Some(4.try_into().unwrap()),
                        offset: 0,
                    }),
                    binding: 0,
                }],
            });
            pass.set_bind_group(0, &bind_group, &[]);
            pass.set_pipeline(&pipeline);
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&src_buf, 0, &dst_buf, 0, 4);
        let (sender, receiver) = std::sync::mpsc::channel();
        let idx = queue.submit([encoder.finish()]);
        device
            .poll(wgpu::PollType::WaitForSubmissionIndex(idx))
            .unwrap();
        dst_buf.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            r.unwrap();
            sender.send(()).unwrap();
        });
        device.poll(wgpu::PollType::Wait).unwrap();
        receiver.recv().unwrap();
        assert_eq!(
            2,
            u32::from_ne_bytes((*dst_buf.get_mapped_range(..))[..4].try_into().unwrap())
        );
    }
}
