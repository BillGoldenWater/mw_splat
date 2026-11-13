// SPDX-FileCopyrightText: 2026 Golden_Water
//
// SPDX-License-Identifier: MPL-2.0

use std::{
    marker::PhantomData,
    num::NonZero,
    ops::Deref,
    sync::{
        Arc,
        atomic::{self, AtomicU8},
    },
};

use anyhow::{Context as _, Result};
use bytemuck::Pod;
use glam::UVec2;
use image::{ImageBuffer, Pixel};
use wesl::{CompileOptions, Compiler, resolver::VirtualResolver};
use wgpu::{
    Backends, BindGroup, BindGroupDescriptor, BindGroupEntry,
    BindGroupLayout, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
    BindingResource, BindingType, Buffer, BufferAddress,
    BufferBindingType, BufferDescriptor, BufferSize, BufferUsages,
    CommandBuffer, CommandEncoder, CommandEncoderDescriptor,
    ComputePipeline, ComputePipelineDescriptor, Device, DeviceDescriptor,
    Features, MapMode, PipelineCompilationOptions, PipelineLayout,
    PipelineLayoutDescriptor, PollType, QuerySet, QuerySetDescriptor,
    Queue, QueueWriteBufferView, ShaderModule, ShaderModuleDescriptor,
    ShaderSource, ShaderStages, SubmissionIndex, Texture,
    TextureDescriptor,
};

use crate::{
    constants::{
        GRAD_COPIES, INVERSE_CDF_SIZE, TILE_SIZE, WG_SIZE, WG_SIZE_2D,
    },
    utils::copy_with_stride,
};

#[derive(Clone)]
pub struct GpuContext {
    device: Device,
    queue: Queue,
    shaders: Shaders,
}

impl GpuContext {
    pub fn new() -> Result<Self> {
        let instance = wgpu::Instance::default();
        let adapters = pollster::block_on(
            instance.enumerate_adapters(Backends::all()),
        );

        if std::env::var("WGPU_LIST_ADAPTER").is_ok_and(|it| it == "true")
        {
            for adapter in &adapters {
                tracing::info!("{:?}", adapter.get_info());
            }
        }

        let adapter = std::env::var("WGPU_BACKEND")
            .ok()
            .and_then(|name| {
                adapters
                    .into_iter()
                    .find(|it| it.get_info().backend.to_str() == name)
            })
            .map(Ok)
            .unwrap_or_else(|| {
                pollster::block_on(instance.request_adapter(
                    &wgpu::RequestAdapterOptionsBase {
                        power_preference:
                            wgpu::PowerPreference::HighPerformance,
                        force_fallback_adapter: false,
                        compatible_surface: None,
                        apply_limit_buckets: false,
                    },
                ))
                .context("instance.request_adapter")
            })?;

        let features = Features::IMMEDIATES
            | Features::BUFFER_BINDING_ARRAY
            | Features::STORAGE_RESOURCE_BINDING_ARRAY
            | Features::SHADER_FLOAT32_ATOMIC
            | if cfg!(feature = "gpu_profiling") {
                Features::TIMESTAMP_QUERY
                    | Features::TIMESTAMP_QUERY_INSIDE_PASSES
            } else {
                Default::default()
            };

        let (device, queue) = pollster::block_on(adapter.request_device(
            &DeviceDescriptor {
                label: Some("main device"),
                required_features: features,
                required_limits: adapter.limits(),
                memory_hints: wgpu::MemoryHints::Performance,
                ..Default::default()
            },
        ))
        .context("adapter.request_device")?;

        let shaders = Shaders::new(&device);

        Ok(Self {
            device,
            queue,
            shaders,
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn queue(&self) -> &Queue {
        &self.queue
    }

    pub fn create_buffer(
        &self,
        label: &str,
        size: u64,
        usage: BufferUsages,
    ) -> Buffer {
        self.device.create_buffer(&BufferDescriptor {
            label: Some(label),
            size,
            usage,
            mapped_at_creation: false,
        })
    }

    pub fn bindgroup_layout<'a>(
        &'a self,
        label: &'a str,
    ) -> BindGroupLayoutBuilder<'a> {
        BindGroupLayoutBuilder {
            device: &self.device,
            label,
            entries: Vec::with_capacity(16),
            visibility: ShaderStages::COMPUTE,
            binding: 0,
        }
    }

    pub fn bindgroup<'a>(
        &'a self,
        label: &'a str,
        layout: &'a BindGroupLayout,
    ) -> BindGroupBuilder<'a> {
        BindGroupBuilder {
            device: &self.device,
            label,
            layout,
            entries: Vec::with_capacity(16),
            binding: 0,
        }
    }

    pub fn pipeline_layout<'a>(
        &'a self,
        label: &'a str,
    ) -> PipelineLayoutBuilder<'a> {
        PipelineLayoutBuilder {
            device: &self.device,
            label,
            bind_group_layouts: Vec::with_capacity(16),
            immediate_size: 0,
        }
    }

    pub fn buffer_reader<T, U>(
        &self,
        src_buffer: Buffer,
    ) -> BufferReader<T, U> {
        BufferReader::new(self, src_buffer)
    }

    pub fn read_buffer(&self, buffer: Buffer, f: impl FnOnce(&[u8])) {
        let mut encoder = self.device.create_command_encoder(
            &CommandEncoderDescriptor {
                label: Some("read_buffer"),
            },
        );

        let mut reader = self.buffer_reader::<(), ()>(buffer);
        let initiated = reader.try_initiate_read(&mut encoder, ());
        assert!(initiated);

        let cmd = encoder.finish();

        self.queue.submit([cmd]);
        self.poll_blocking();

        reader.try_resolve_read(|view, _| f(view));
    }

    pub fn read_buffer_pod<T>(&self, buffer: Buffer) -> T
    where
        T: Pod + GpuLayout,
    {
        let mut encoder = self.device.create_command_encoder(
            &CommandEncoderDescriptor {
                label: Some("read_buffer_pod"),
            },
        );

        let mut reader = self.buffer_reader::<T, ()>(buffer);
        let initiated = reader.try_initiate_read(&mut encoder, ());
        assert!(initiated);

        let cmd = encoder.finish();

        self.queue.submit([cmd]);
        self.poll_blocking();

        reader.try_resolve_read_pod().unwrap().0
    }

    pub fn buffer_write_image<P, S, C>(
        &self,
        img: &ImageBuffer<P, C>,
        buffer: &Buffer,
        stride: usize,
        rev_y: bool,
    ) where
        P: Pixel<Subpixel = S>,
        S: Pod + GpuLayout,
        C: Deref<Target = [S]>,
    {
        let size = buffer.size().try_into().unwrap();
        let mut write_view =
            self.write_buffer_with(buffer, 0, size).unwrap();
        let raw_src: &[u8] = bytemuck::cast_slice(img.as_raw());

        let px_stride = P::CHANNEL_COUNT as usize * S::STRIDE as usize;
        let dims = UVec2::from(img.dimensions()).as_usizevec2();

        let width = dims.x * px_stride;
        let tgt_stride = stride * px_stride;
        copy_with_stride(
            width,
            dims.y,
            width,
            tgt_stride,
            rev_y,
            |src, tgt| {
                write_view.slice(tgt).copy_from_slice(&raw_src[src]);
            },
        );
    }

    pub fn poll_blocking(&self) {
        self.device.poll(PollType::wait_indefinitely()).unwrap();
    }

    delegate::delegate! {
        to self.device {
            pub fn create_texture(&self, desc: &TextureDescriptor<'_>) -> Texture;
            pub fn create_query_set(&self, desc: &QuerySetDescriptor<'_>) -> QuerySet;
            pub fn create_compute_pipeline(&self, desc: &ComputePipelineDescriptor<'_>) -> ComputePipeline;
            pub fn create_command_encoder(&self, desc: &CommandEncoderDescriptor<'_>) -> CommandEncoder;
        }

        to self.queue {
            pub fn write_buffer(&self, buffer: &Buffer, offset: BufferAddress, data: &[u8]);
            pub fn write_buffer_with(
                &self,
                buffer: &Buffer,
                offset: BufferAddress,
                size: BufferSize,
            ) -> Option<QueueWriteBufferView>;
            pub fn submit<I: IntoIterator<Item = CommandBuffer>>(
                &self,
                command_buffers: I,
            ) -> SubmissionIndex;
        }
    }

    pub fn shaders(&self) -> &Shaders {
        &self.shaders
    }
}

pub struct BindGroupLayoutBuilder<'ctx> {
    device: &'ctx Device,
    label: &'ctx str,
    entries: Vec<BindGroupLayoutEntry>,
    visibility: ShaderStages,
    binding: u32,
}

impl<'ctx> BindGroupLayoutBuilder<'ctx> {
    pub fn build(self) -> BindGroupLayout {
        self.device
            .create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some(self.label),
                entries: &self.entries,
            })
    }

    pub fn with_visibility(
        mut self,
        shader_stages: ShaderStages,
    ) -> Self {
        self.visibility = shader_stages;
        self
    }

    pub fn with_binding(mut self, binding: impl Into<u32>) -> Self {
        self.binding = binding.into();
        self
    }

    pub fn push(
        mut self,
        ty: BindingType,
        count: Option<NonZero<u32>>,
    ) -> Self {
        self.entries.push(BindGroupLayoutEntry {
            binding: self.binding,
            visibility: self.visibility,
            ty,
            count,
        });
        self.binding += 1;
        self
    }

    pub fn push_buffer(
        self,
        ty: BufferBindingType,
        count: Option<NonZero<u32>>,
    ) -> Self {
        self.push(
            BindingType::Buffer {
                ty,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count,
        )
    }

    pub fn push_buffer_storage(
        self,
        read_only: bool,
        count: Option<NonZero<u32>>,
    ) -> Self {
        self.push_buffer(BufferBindingType::Storage { read_only }, count)
    }
}

pub struct BindGroupBuilder<'ctx> {
    device: &'ctx Device,
    label: &'ctx str,
    layout: &'ctx BindGroupLayout,
    entries: Vec<BindGroupEntry<'ctx>>,
    binding: u32,
}

impl<'ctx> BindGroupBuilder<'ctx> {
    pub fn build(self) -> BindGroup {
        self.device.create_bind_group(&BindGroupDescriptor {
            label: Some(self.label),
            layout: self.layout,
            entries: &self.entries,
        })
    }

    pub fn with_binding(mut self, binding: impl Into<u32>) -> Self {
        self.binding = binding.into();
        self
    }

    pub fn push_entry(mut self, resource: BindingResource<'ctx>) -> Self {
        self.entries.push(BindGroupEntry {
            binding: self.binding,
            resource,
        });
        self.binding += 1;
        self
    }

    pub fn push_entry_buffer(self, buffer: &'ctx Buffer) -> Self {
        self.push_entry(buffer.as_entire_binding())
    }
}

pub struct PipelineLayoutBuilder<'ctx> {
    device: &'ctx Device,
    label: &'ctx str,
    bind_group_layouts: Vec<Option<&'ctx BindGroupLayout>>,
    immediate_size: u32,
}

impl<'ctx> PipelineLayoutBuilder<'ctx> {
    pub fn build(self) -> PipelineLayout {
        self.device
            .create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some(self.label),
                bind_group_layouts: &self.bind_group_layouts,
                immediate_size: self.immediate_size,
            })
    }

    pub fn build_compute_pipeline(
        self,
        shader: &ShaderModule,
        entry: &str,
    ) -> ComputePipeline {
        let label = self.label;
        let device = self.device;
        let layout = self.build();
        device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(&layout),
            module: shader,
            entry_point: Some(entry),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        })
    }

    pub fn with_immediate_size(mut self, immediate_size: u32) -> Self {
        self.immediate_size = immediate_size;
        self
    }

    pub fn push_bind_group_layout(
        mut self,
        bind_group_layout: Option<&'ctx BindGroupLayout>,
    ) -> Self {
        self.bind_group_layouts.push(bind_group_layout);
        self
    }
}

#[derive(Clone)]
pub struct Shaders {
    pub bin_count: ShaderModule,
    pub bin_alloc: ShaderModule,
    pub bin_fill: ShaderModule,
    pub bin_sort: ShaderModule,
    pub splats_cache_fill: ShaderModule,
    pub render: ShaderModule,
    pub learn_backward: ShaderModule,
    pub learn_adam: ShaderModule,

    pub render_outline: ShaderModule,

    pub redistribute_cells: ShaderModule,
    pub redistribute_select: ShaderModule,
    pub redistribute_move: ShaderModule,
    pub redistribute_sort: ShaderModule,
}

impl Shaders {
    pub fn new(device: &Device) -> Self {
        const WESL_SRC: include_dir::Dir<'static> =
            include_dir::include_dir!("src/shaders/");
        fn fill_resolver(
            dir: &include_dir::Dir<'static>,
            resolver: &mut VirtualResolver<'static>,
            parent: &str,
        ) {
            for file in dir.files() {
                let content = file.contents_utf8().unwrap();
                let name =
                    file.path().file_stem().unwrap().to_str().unwrap();
                let module_path = format!("{parent}::{name}");
                tracing::debug!("register shader module {module_path}");
                resolver.add_module(
                    module_path.parse().unwrap(),
                    content.into(),
                );
            }
            for dir in dir.dirs() {
                let name =
                    dir.path().file_name().unwrap().to_str().unwrap();
                let module_path = format!("{parent}::{name}");
                fill_resolver(dir, resolver, &module_path);
            }
        }

        let options = CompileOptions {
            lower: true,
            ..Default::default()
        };
        let mut resolver = VirtualResolver::new();

        resolver.add_module(
            "package::_constants".parse().unwrap(),
            format!(
                r#"
const WG_SIZE_2D: u32 = {WG_SIZE_2D}u;
const WG_SIZE: u32 = {WG_SIZE}u;

const TILE_SIZE: u32 = {TILE_SIZE}u;
const GRAD_COPIES: u32 = {GRAD_COPIES}u;
const INVERSE_CDF_SIZE: u32 = {INVERSE_CDF_SIZE}u;
"#
            )
            .into(),
        );

        fill_resolver(&WESL_SRC, &mut resolver, "package");
        let compiler = Compiler::new_with_resolver(options, resolver);

        macro_rules! compile {
            ($($name:ident),* $(,)?) => {
                $(
                    let $name = {
                        let name = concat!("package::", stringify!($name));
                        tracing::debug!("compile shader {name}");
                        device
                            .create_shader_module(ShaderModuleDescriptor {
                                label: Some(stringify!($name)),
                                source: ShaderSource::Wgsl(compiler
                                    .compile_module(
                                        &name
                                        .parse()
                                        .unwrap(),
                                    )
                                    .inspect_err(|err| panic!("{err}"))
                                    .unwrap()
                                    .to_string().into()),
                            })
                    };
                )*
            };
        }

        compile![
            bin_count,
            bin_alloc,
            bin_fill,
            bin_sort,
            splats_cache_fill,
            render,
            learn_backward,
            learn_adam,
            render_outline,
            redistribute_cells,
            redistribute_select,
            redistribute_move,
            redistribute_sort,
        ];

        Self {
            bin_count,
            bin_alloc,
            bin_fill,
            bin_sort,
            splats_cache_fill,
            render,
            learn_backward,
            learn_adam,

            render_outline,

            redistribute_cells,
            redistribute_select,
            redistribute_move,
            redistribute_sort,
        }
    }
}

pub const fn align_to(size: u64, align: u64) -> u64 {
    size.div_ceil(align) * align
}

pub trait GpuLayout {
    const SIZE: u64;
    const STRIDE: u64;
}

impl<T: Sized> GpuLayout for T {
    const SIZE: u64 = size_of::<T>() as u64;
    const STRIDE: u64 =
        align_to(size_of::<T>() as u64, align_of::<T>() as u64);
}

#[derive(Debug)]
#[repr(u8)]
enum AsyncReaderState {
    Idle,
    Mapping,
    Ready,
    Reading,
}

pub struct BufferReader<T, U> {
    state: Arc<AtomicU8>,
    src_buffer: Buffer,
    map_buffer: Buffer,
    user_data: Option<U>,
    _data: PhantomData<T>,
}

impl<T, U> BufferReader<T, U> {
    pub fn new(gpu: &GpuContext, src_buffer: Buffer) -> Self {
        assert!(src_buffer.size() != 0);
        assert!(src_buffer.size().is_multiple_of(4));
        assert!(src_buffer.usage().contains(BufferUsages::COPY_SRC));

        let map_buffer = gpu.create_buffer(
            "AsyncReader::new_raw_map",
            src_buffer.size(),
            BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        );

        Self {
            state: AtomicU8::new(AsyncReaderState::Idle as _).into(),
            src_buffer,
            map_buffer,
            user_data: None,
            _data: PhantomData,
        }
    }

    pub fn read_blocking(
        &mut self,
        gpu: &GpuContext,
        f: impl FnOnce(&[u8]),
    ) where
        U: Default,
    {
        // only write to state where it's not taking a &mut self
        // is the callback from map,
        // which would be Ready
        assert!(
            self.state.load(atomic::Ordering::Relaxed)
                == AsyncReaderState::Idle as u8
        );

        let mut encoder =
            gpu.create_command_encoder(&CommandEncoderDescriptor {
                label: Some("AsyncReader::read_blocking"),
            });

        let initiated =
            self.try_initiate_read(&mut encoder, U::default());
        assert!(initiated);

        let cmd = encoder.finish();
        gpu.submit([cmd]);
        gpu.poll_blocking();

        let success = self.try_resolve_read(|view, _| f(view));
        assert!(success);
    }

    pub fn try_initiate_read(
        &mut self,
        command_encoder: &mut CommandEncoder,
        additional_data: U,
    ) -> bool {
        if self
            .state
            .compare_exchange(
                AsyncReaderState::Idle as _,
                AsyncReaderState::Mapping as _,
                atomic::Ordering::Acquire,
                atomic::Ordering::Relaxed,
            )
            .is_err()
        {
            return false;
        }

        command_encoder.copy_buffer_to_buffer(
            &self.src_buffer,
            0,
            &self.map_buffer,
            0,
            self.map_buffer.size(),
        );
        let state = self.state.clone();
        command_encoder.map_buffer_on_submit(
            &self.map_buffer,
            MapMode::Read,
            ..,
            move |res| {
                res.unwrap();
                state.store(
                    AsyncReaderState::Ready as _,
                    atomic::Ordering::Release,
                );
            },
        );
        self.user_data = Some(additional_data);

        true
    }

    pub fn try_resolve_read(&mut self, f: impl FnOnce(&[u8], U)) -> bool {
        if self
            .state
            .compare_exchange(
                AsyncReaderState::Ready as _,
                AsyncReaderState::Reading as _,
                atomic::Ordering::Acquire,
                atomic::Ordering::Relaxed,
            )
            .is_err()
        {
            return false;
        }

        let view = self.map_buffer.get_mapped_range(..).unwrap();
        let user_data = self.user_data.take().unwrap();
        f(&view, user_data);
        drop(view);
        self.map_buffer.unmap();

        self.state.store(
            AsyncReaderState::Idle as _,
            atomic::Ordering::Release,
        );

        true
    }

    pub fn read_image_blocking<P, S, C>(
        &mut self,
        gpu: &GpuContext,
        dims: UVec2,
        stride: u32,
        rev_y: bool,
    ) -> ImageBuffer<P, C>
    where
        U: Default,
        P: Pixel<Subpixel = S>,
        S: Pod + GpuLayout + Default,
        C: From<Vec<S>> + Deref<Target = [S]>,
    {
        let ch = P::CHANNEL_COUNT as usize;
        let mut buffer =
            vec![S::default(); dims.element_product() as usize * ch];

        let (width, height) = dims.as_usizevec2().into();
        self.read_blocking(gpu, |view| {
            let view: &[S] = bytemuck::cast_slice(view);
            let stride = stride as usize * ch;
            assert_eq!(view.len() / stride, height);

            let width = width * 4;
            copy_with_stride(
                width,
                height,
                stride,
                width,
                rev_y,
                |src, tgt| {
                    buffer[tgt].copy_from_slice(&view[src]);
                },
            );
        });

        ImageBuffer::from_raw(dims.x, dims.y, buffer.into()).unwrap()
    }
}

impl<T: Pod + GpuLayout, U> BufferReader<T, U> {
    pub fn try_resolve_read_pod(&mut self) -> Option<(T, U)> {
        assert!(self.src_buffer.size() == T::STRIDE);

        let mut result = None;
        self.try_resolve_read(|view, user_data| {
            result = Some((*bytemuck::from_bytes::<T>(view), user_data))
        });
        result
    }

    pub fn read_blocking_pod(&mut self, gpu: &GpuContext) -> T
    where
        U: Default,
    {
        let mut result = None;
        self.read_blocking(gpu, |view| {
            result = Some(*bytemuck::from_bytes::<T>(view))
        });
        result.unwrap()
    }
}
