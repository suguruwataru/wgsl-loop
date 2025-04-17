#[cfg(test)]
mod test {
    // ONLY TO BE RUN WITH `cargo test -- --test-threads=1`
    use pollster::FutureExt;
    use std::sync::atomic::{AtomicU32, Ordering};

    static LOCK: AtomicU32 = AtomicU32::new(0);
    static mut CTR: u32 = 0;

    #[test]
    fn loop_continue() {
        unsafe { CTR = 0 }
        (0..2)
            .map(|_| {
                let t = move || loop {
                    if LOCK.fetch_or(1, Ordering::Relaxed) == 1 {
                        continue;
                    }
                    unsafe { CTR += 1 }
                    LOCK.fetch_and(0, Ordering::Relaxed);
                    break;
                };
                std::thread::spawn(t)
            })
            .collect::<Vec<_>>()
            .into_iter()
            .for_each(|h| h.join().unwrap());
        assert_eq!(unsafe { CTR }, 2);
    }

    #[test]
    fn loop_break() {
        unsafe { CTR = 0 }
        (0..2)
            .map(|_| {
                let t = move || {
                    loop {
                        if LOCK.fetch_or(1, Ordering::Relaxed) == 0 {
                            break;
                        }
                    }
                    unsafe { CTR += 1 }
                    LOCK.fetch_and(0, Ordering::Relaxed);
                };
                std::thread::spawn(t)
            })
            .collect::<Vec<_>>()
            .into_iter()
            .for_each(|h| h.join().unwrap());
        assert_eq!(unsafe { CTR }, 2);
    }

    #[test]
    fn while_other_has_lock() {
        unsafe { CTR = 0 }
        (0..2)
            .map(|_| {
                let t = move || loop {
                    while LOCK.fetch_or(1, Ordering::Relaxed) == 1 {}
                    unsafe { CTR += 1 }
                    LOCK.fetch_and(0, Ordering::Relaxed);
                    break;
                };
                std::thread::spawn(t)
            })
            .collect::<Vec<_>>()
            .into_iter()
            .for_each(|h| h.join().unwrap());
        assert_eq!(unsafe { CTR }, 2);
    }

    #[test]
    fn while_self_has_not_lock() {
        unsafe { CTR = 0 }
        (0..2)
            .map(|_| {
                let t = move || loop {
                    while LOCK.fetch_or(1, Ordering::Relaxed) != 0 {}
                    unsafe { CTR += 1 }
                    LOCK.fetch_and(0, Ordering::Relaxed);
                    break;
                };
                std::thread::spawn(t)
            })
            .collect::<Vec<_>>()
            .into_iter()
            .for_each(|h| h.join().unwrap());
        assert_eq!(unsafe { CTR }, 2);
    }

    #[ignore = "this makes your computer hang"]
    #[test]
    fn wgsl_while_other_has_lock() {
        test_shader("while_other_has_lock");
    }

    #[ignore = "this makes your computer hang"]
    #[test]
    fn wgsl_while_self_has_not_lock() {
        test_shader("while_self_has_not_lock");
    }

    #[ignore = "this makes your computer hang"]
    #[test]
    fn wgsl_loop_break() {
        test_shader("loop_break");
    }

    #[test]
    fn wgsl_loop_continue() {
        test_shader("loop_continue");
    }

    fn test_shader(entry_point: &str) {
        let wgpu_instance = wgpu::Instance::default();
        let adapter = wgpu_instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .block_on()
            .unwrap();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_features: wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES,
                ..Default::default()
            })
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
