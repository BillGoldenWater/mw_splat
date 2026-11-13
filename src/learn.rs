// SPDX-FileCopyrightText: 2026 Golden_Water
//
// SPDX-License-Identifier: MPL-2.0

use std::{
    collections::VecDeque,
    mem::offset_of,
    time::{Duration, Instant},
};

use bytemuck::bytes_of;
use glam::{UVec2, UVec3, UVec4, Vec4};
use image::{ImageBuffer, Luma, Rgba};
use rand::{Rng, SeedableRng};
#[cfg(feature = "gpu_profiling")]
use wgpu::{
    Buffer, BufferUsages, QuerySet, QuerySetDescriptor, QueryType,
};
use wgpu::{
    ColorTargetState, ColorWrites, CommandBuffer, CommandEncoder,
    CommandEncoderDescriptor, ComputePassDescriptor, FragmentState,
    LoadOp, MultisampleState, Operations, Origin3d, PollType,
    RenderPassColorAttachment, RenderPassDescriptor,
    RenderPipelineDescriptor, SubmissionIndex, TexelCopyBufferInfo,
    TexelCopyBufferLayout, TexelCopyTextureInfo, TextureAspect,
    TextureDescriptor, TextureDimension, TextureFormat, TextureUsages,
    TextureViewDescriptor, VertexState,
};

pub use self::gpu_types::Lr;
#[cfg(feature = "gpu_profiling")]
use crate::learn::gpu_types::Timestamps;
use crate::{
    constants::{
        GPU_MAX_IN_FLIGHT, REF_HEIGHT, RFE_RAMP_CAP, TILE_SIZE, WG_SIZE,
        WG_SIZE_2D,
    },
    gpu::{BufferReader, GpuContext, GpuLayout},
    ir::{Ir, Splat},
    learn::{
        bind_groups::BindGroups,
        buffers::Buffers,
        gpu_types::{
            BinOverflow, ImmAdam, ImmBinAlloc, ImmRedistributeCells,
            ImmRedistributeMove, ImmRedistributeSelect,
            ImmRedistributeSort, STRIDE_SAMPLING_CELL,
        },
        pipelines::Pipelines,
    },
    learn_dispatch, learn_write_ts,
    utils::image_ext::ImageBufferExt as _,
};

mod bind_groups;
mod buffers;
mod gpu_types;
mod pipelines;

pub struct Learner {
    pub gpu: GpuContext,
    pub config: Config,

    pub info: gpu_types::Info,
    pub splat_count: u32,

    pub pipelines: Pipelines,

    pub buffers: Buffers,
    pub bind_groups: BindGroups,

    pub bin_overflow_reader: BufferReader<BinOverflow, ()>,

    #[cfg(feature = "gpu_profiling")]
    pub timestamp_query_set: QuerySet,
    #[cfg(feature = "gpu_profiling")]
    pub timestamp_buffer: Buffer,
    #[cfg(feature = "gpu_profiling")]
    pub timestamp_reader: BufferReader<Timestamps, u32>,
}

impl Learner {
    pub fn new(
        gpu: GpuContext,
        reference: &ImageBuffer<Rgba<f32>, Vec<f32>>,
        config: Config,
        ir: &Ir,
    ) -> Self {
        let canvas = UVec2::from(reference.dimensions());
        let splat_count = ir.splats.len() as u64;

        let info = gpu_types::Info {
            canvas: gpu_types::Canvas {
                dims: canvas,
                dimsf: canvas.as_vec2(),
                stride: canvas.x.next_multiple_of(
                    wgpu::COPY_BYTES_PER_ROW_ALIGNMENT.max(64)
                        / f32::STRIDE as u32,
                ),
                ramp_cap: RFE_RAMP_CAP * canvas.y as f32 / REF_HEIGHT,
                _pad: Default::default(),
                background: ir.background,
            },
            tile_dims: canvas.map(|it| it.div_ceil(TILE_SIZE)),
            lr: config.lr,
        };

        let pipelines = Pipelines::new(&gpu);
        let buffers = Buffers::new(&gpu, &info, splat_count, 1 << 18);
        let bind_groups = BindGroups::new(&gpu, &pipelines, &buffers);

        gpu.write_buffer(&buffers.info, 0, bytes_of(&info));

        // reference
        gpu.buffer_write_image(
            reference,
            &buffers.reference,
            info.canvas.stride as _,
            true,
        );

        // splats
        {
            let size = buffers.splats.size().try_into().unwrap();
            let mut write_view =
                gpu.write_buffer_with(&buffers.splats, 0, size).unwrap();

            let splat_stride = gpu_types::Splat::STRIDE as usize;
            let splat_size = gpu_types::Splat::SIZE as usize;
            for (idx, splat) in ir.splats.iter().enumerate() {
                let mut splat = gpu_types::Splat::from(splat);
                splat.pos *= info.canvas.dimsf;
                splat.size *= info.canvas.dimsf;
                let tgt_base = splat_stride * idx;
                let tgt_range = tgt_base..tgt_base + splat_size;
                let mut tgt = write_view.slice(tgt_range);
                tgt.copy_from_slice(bytes_of(&splat));
            }
        }

        let bin_overflow_reader =
            gpu.buffer_reader(buffers.bin_overflow.clone());

        #[cfg(feature = "gpu_profiling")]
        let timestamp_query_set =
            gpu.create_query_set(&QuerySetDescriptor {
                label: Some("learner ts query set"),
                ty: QueryType::Timestamp,
                count: Timestamps::COUNT,
            });
        #[cfg(feature = "gpu_profiling")]
        let timestamp_buffer = gpu.create_buffer(
            "learner ts",
            Timestamps::STRIDE,
            BufferUsages::QUERY_RESOLVE | BufferUsages::COPY_SRC,
        );
        #[cfg(feature = "gpu_profiling")]
        let timestamp_reader =
            gpu.buffer_reader(timestamp_buffer.clone());

        let this = Self {
            gpu,
            config,
            info,
            splat_count: u32::try_from(splat_count).unwrap(),
            pipelines,
            buffers,
            bind_groups,
            bin_overflow_reader,
            #[cfg(feature = "gpu_profiling")]
            timestamp_query_set,
            #[cfg(feature = "gpu_profiling")]
            timestamp_buffer,
            #[cfg(feature = "gpu_profiling")]
            timestamp_reader,
        };

        this.reset_overflow();

        let ones = ImageBuffer::from_pixel(
            this.info.canvas.dims.x,
            this.info.canvas.dims.y,
            Luma([1.0]),
        );
        this.set_loss_weights(&ones);

        this.gpu.queue().submit([]);
        this.gpu.poll_blocking();

        this
    }

    pub fn loss_weights_buffer(&self) -> &wgpu::Buffer {
        &self.buffers.loss_weights
    }

    pub fn set_loss_weights(
        &self,
        weights: &ImageBuffer<Luma<f32>, Vec<f32>>,
    ) {
        let dims = self.info.canvas.dims;
        assert_eq!(
            weights.dimensions(),
            (dims.x, dims.y),
            "loss weights must match the canvas size",
        );

        self.gpu.buffer_write_image(
            weights,
            &self.buffers.loss_weights,
            self.info.canvas.stride as _,
            true,
        );
    }

    pub fn reset_overflow(&self) {
        let splats_wg_num = self.splat_count.div_ceil(WG_SIZE);

        let splats_wg = self.spread_workgroup_to_x_y(splats_wg_num);
        let bin_fill_wg = self.spread_workgroup_to_x_y(self.splat_count);
        let bin_num = self.info.tile_dims.element_product();
        let bin_sort_wg = self.spread_workgroup_to_x_y(bin_num);

        let canvas_wg = self
            .info
            .canvas
            .dims
            .map(|it| it.div_ceil(WG_SIZE_2D))
            .extend(1)
            .extend(0);
        let backward_wg = self.info.tile_dims.extend(1).extend(0);

        let bin_overflow = BinOverflow {
            overflow: 0,
            iter_num: 0,
            _pad: Default::default(),
            i_bin_fill: bin_fill_wg,
            i_bin_sort: bin_sort_wg,
            i_splats_cache_fill: splats_wg,
            i_render: canvas_wg,
            i_learn_backward: backward_wg,
            i_learn_adam: splats_wg,
        };

        self.gpu.write_buffer(
            &self.buffers.bin_overflow,
            0,
            bytes_of(&bin_overflow),
        );
    }

    pub fn spread_workgroup_to_x_y(&self, num: u32) -> UVec4 {
        let limit = self
            .gpu
            .device()
            .limits()
            .max_compute_workgroups_per_dimension;
        if num > limit {
            let y = num.div_ceil(limit);
            let x = num.div_ceil(y);
            UVec4::new(x, y, 1, 0)
        } else {
            UVec4::new(num, 1, 1, 0)
        }
    }

    pub fn run(
        &mut self,
        iters: u32,
        lr_scale: f32,
        redistribute: Option<u32>,
        seed: u64,
    ) {
        assert!(iters >= 1);

        let mut iter_num = 0;
        let mut pending = VecDeque::<SubmissionIndex>::new();
        let mut avg_time = 0.;
        let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);

        loop {
            macro_rules! ensure_no_overflow {
                () => {
                    self.gpu.poll_blocking();
                    if let Some((it, _)) =
                        self.bin_overflow_reader.try_resolve_read_pod()
                        && it.overflow > 0
                    {
                        iter_num = it.iter_num;
                        self.grow_bin_data();
                        continue;
                    }
                };
            }

            if let Some((it, _)) =
                self.bin_overflow_reader.try_resolve_read_pod()
                && it.overflow > 0
            {
                iter_num = it.iter_num;
                self.grow_bin_data();
            }
            #[cfg(feature = "gpu_profiling")]
            if let Some((it, iter_num)) =
                self.timestamp_reader.try_resolve_read_pod()
            {
                it.log(&self.gpu, iter_num);
            }

            let avg_dur = Duration::from_secs_f64(avg_time);
            tracing::debug!(
                "issuing iter: {iter_num}, lr: {lr_scale}, avg_time: {avg_dur:.2?}"
            );
            let cmd = self.step(iter_num, lr_scale);
            let idx = self.gpu.submit([cmd]);
            pending.push_back(idx);

            if let Some(redistribute) = redistribute
                && redistribute > 0
                && iter_num != iters
                && iter_num.is_multiple_of(redistribute)
            {
                ensure_no_overflow!();

                let progress = iter_num as f32 / iters as f32;
                let canvas = self.info.canvas.dims;
                let splats = u64::from(self.splat_count);

                let spacing = (canvas.element_product() as f32
                    / splats.max(1) as f32)
                    .sqrt();

                let cell_size = (spacing
                    * 2f32.powf(1.0 - 3.0 * progress))
                .max(1.0) as u32;
                let cell_count = canvas
                    .map(|it| it.div_ceil(cell_size))
                    .element_product();

                let splat_size = spacing * 0.9;

                let quantile = 0.5 * 2f32.powf(-3.0 * progress);
                let move_count =
                    (quantile * cell_count as f32).max(1.0) as u32;

                let seed = rng.next_u32();

                self.run_redistribute(
                    cell_size, move_count, splat_size, seed,
                );
                pending.clear();
            }

            if pending.len() >= GPU_MAX_IN_FLIGHT {
                while pending.len() >= (GPU_MAX_IN_FLIGHT / 2) {
                    let start = Instant::now();
                    self.gpu
                        .device()
                        .poll(PollType::Wait {
                            submission_index: pending.pop_front(),
                            timeout: None,
                        })
                        .unwrap();
                    avg_time += start.elapsed().as_secs_f64();
                    avg_time /= 2.;
                }
            }

            iter_num += 1;

            if iter_num == iters {
                ensure_no_overflow!();

                break;
            }
        }
    }

    pub fn step(
        &mut self,
        iter_num: u32,
        lr_scale: f32,
    ) -> CommandBuffer {
        let mut encoder = self.gpu.device().create_command_encoder(
            &CommandEncoderDescriptor {
                label: Some("step encoders;"),
            },
        );

        self.step_render(&mut encoder, iter_num);

        let buffers_to_clear =
            [&self.buffers.gradients, &self.buffers.loss];
        for buffer in buffers_to_clear {
            encoder.clear_buffer(buffer, 0, None);
        }

        let mut pass =
            encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("step compute pass"),
                timestamp_writes: None,
            });

        learn_dispatch! {
            pass, self;
            learn_backward@i(i_learn_backward): {
                0 = base_ro;
                1 = splats_ro;
                2 = bin_heads_ro;
                3 = bin_data_ro;
                4 = canvas_ro;
                5 = learn_backward;
            },
            learn_adam@i(i_learn_adam)(&ImmAdam { lr_scale }): {
                1 = splats_adam;
                2 = learn_gradients_ro;
                3 = adam_rw;
            },
        };

        drop(pass);

        self.bin_overflow_reader.try_initiate_read(&mut encoder, ());

        #[cfg(feature = "gpu_profiling")]
        {
            encoder.resolve_query_set(
                &self.timestamp_query_set,
                0..Timestamps::COUNT,
                &self.timestamp_buffer,
                0,
            );
            self.timestamp_reader
                .try_initiate_read(&mut encoder, iter_num);
        };

        encoder.finish()
    }

    pub fn run_render(&mut self) {
        loop {
            let mut encoder = self.gpu.create_command_encoder(
                &CommandEncoderDescriptor {
                    label: Some("render to texture"),
                },
            );

            self.step_render(&mut encoder, 0);
            self.bin_overflow_reader.try_initiate_read(&mut encoder, ());

            self.gpu.submit([encoder.finish()]);

            self.gpu.poll_blocking();
            let (it, _) =
                self.bin_overflow_reader.try_resolve_read_pod().unwrap();
            if it.overflow == 0 {
                break;
            }

            self.grow_bin_data();
        }
    }

    pub fn step_render(
        &self,
        encoder: &mut CommandEncoder,
        iter_num: u32,
    ) {
        let buffers_to_clear = [
            &self.buffers.bin_counts,
            &self.buffers.bin_bump,
            &self.buffers.bin_cursor,
        ];
        for buffer in buffers_to_clear {
            encoder.clear_buffer(buffer, 0, None);
        }

        let mut pass =
            encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("step_render compute pass"),
                timestamp_writes: None,
            });

        let bin_wg_num = UVec3::new(
            self.info.tile_dims.element_product().div_ceil(WG_SIZE),
            1,
            1,
        );

        let splats_wg = self.spread_workgroup_to_x_y(self.splat_count);

        learn_write_ts!(pass, self, begin);
        learn_dispatch! {
            pass, self;
            bin_count@n(splats_wg): {
                0 = base_ro;
                1 = splats_ro;
                2 = bin_count;
            },
            bin_alloc@n(bin_wg_num)(&ImmBinAlloc { iter_num }): {
                2 = bin_alloc;
                3 = bin_data_ro;
            },
            bin_fill@i(i_bin_fill): {
                2 = bin_fill;
                3 = bin_data_rw;
            },
            bin_sort@i(i_bin_sort): {
                2 = bin_sort;
                3 = bin_data_rw;
            },
            splats_cache_fill@i(i_splats_cache_fill): {
                1 = splats_cache_fill;
            },
            render@i(i_render): {
                1 = splats_ro;
                2 = bin_heads_ro;
                3 = bin_data_ro;
                4 = canvas_rw;
            },
        };

        drop(pass);
    }

    pub fn run_redistribute(
        &mut self,
        cell_size: u32,
        move_count: u32,
        splat_size: f32,
        seed: u32,
    ) {
        self.run_render();

        let cell_count = self
            .info
            .canvas
            .dims
            .map(|it| it.div_ceil(cell_size))
            .element_product() as u64;
        self.reserve_sampling_cells(cell_count);

        let mut encoder = self.gpu.device().create_command_encoder(
            &CommandEncoderDescriptor {
                label: Some("redistribute"),
            },
        );

        self.step_redistribute(
            &mut encoder,
            cell_size,
            move_count,
            splat_size,
            seed,
        );

        self.gpu.submit([encoder.finish()]);
        self.gpu.poll_blocking();
    }

    pub fn step_redistribute(
        &mut self,
        encoder: &mut CommandEncoder,
        cell_size: u32,
        move_count: u32,
        splat_size: f32,
        seed: u32,
    ) {
        let grid_dims =
            self.info.canvas.dims.map(|it| it.div_ceil(cell_size));

        let imm_cells = ImmRedistributeCells {
            grid_dims,
            cell_size,
            _pad: Default::default(),
        };
        let imm_select = ImmRedistributeSelect {
            grid_dims,
            move_count,
            _pad: Default::default(),
        };
        let imm_move = ImmRedistributeMove {
            grid_dims,
            cell_size,
            splat_size,
            seed,
            _pad: Default::default(),
        };

        let mut pass =
            encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("step redistribute compute pass"),
                timestamp_writes: None,
            });

        let splats_wg = self
            .spread_workgroup_to_x_y(self.splat_count.div_ceil(WG_SIZE));
        let cells_wg =
            grid_dims.map(|it| it.div_ceil(WG_SIZE_2D)).extend(1);

        learn_dispatch! {
            pass, self;
            redistribute_cells@n(cells_wg)(&imm_cells): {
                0 = base_ro;
                1 = canvas_ro;
                2 = redistribute_cells;
            },
            redistribute_select@n(UVec3::ONE)(&imm_select): {
                1 = redistribute_select;
            },
            redistribute_move@n(splats_wg)(&imm_move): {
                1 = splats_redistribute;
                2 = redistribute_move;
                3 = adam_rw;
            },
        };

        if self.splat_count >= 2 {
            self.dispatch_redistribute_sort(&mut pass);
        }

        drop(pass);
    }

    fn dispatch_redistribute_sort(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
    ) {
        pass.set_pipeline(&self.pipelines.redistribute_sort);
        pass.set_bind_group(
            1,
            &self.bind_groups.splats_redistribute,
            &[],
        );
        pass.set_bind_group(2, &self.bind_groups.adam_rw, &[]);

        let n = self.splat_count.next_power_of_two();
        let sort_wg =
            self.spread_workgroup_to_x_y((n / 2).div_ceil(WG_SIZE));

        for stage_log2 in 1..=n.ilog2() {
            let stage = 1u32 << stage_log2;
            for step_log2 in (0..stage_log2).rev() {
                let step = 1u32 << step_log2;
                let imm = ImmRedistributeSort {
                    stage,
                    step,
                    step_log2,
                    step_mod_mask: step - 1,
                };
                pass.set_immediates(0, bytes_of(&imm));
                pass.dispatch_workgroups(sort_wg.x, sort_wg.y, sort_wg.z);
            }
        }

        learn_write_ts!(pass, self, redistribute_sort);
    }

    pub fn reserve_sampling_cells(&mut self, cell_count: u64) {
        let current_size =
            self.buffers.redistribute_sampling_cells.size();
        let current_cap = current_size / STRIDE_SAMPLING_CELL;

        if current_cap < cell_count {
            self.buffers.grow_sampling_cells(&self.gpu, cell_count);
            self.bind_groups.recreate_sampling_cells(
                &self.gpu,
                &self.pipelines,
                &self.buffers,
            );
        }
    }

    pub fn grow_bin_data(&mut self) {
        self.buffers.grow_bin_data(&self.gpu);
        self.bind_groups.recreate_bin_data(
            &self.gpu,
            &self.pipelines,
            &self.buffers,
        );
        self.reset_overflow();
    }

    pub fn render(&mut self) -> ImageBuffer<Rgba<f32>, Vec<f32>> {
        self.run_render();

        let mut img = self
            .gpu
            .buffer_reader::<(), ()>(self.buffers.pixels.clone())
            .read_image_blocking(
                &self.gpu,
                self.info.canvas.dims,
                self.info.canvas.stride,
                true,
            );
        img.unpremultiply();

        img
    }

    pub fn render_outlined(
        &mut self,
    ) -> ImageBuffer<Rgba<f32>, Vec<f32>> {
        let format = TextureFormat::Rgba32Float;

        let texture = self.gpu.create_texture(&TextureDescriptor {
            label: Some("render outline"),
            size: self.info.canvas_extent3d(),
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format,
            usage: TextureUsages::COPY_DST
                | TextureUsages::COPY_SRC
                | TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[format],
        });

        self.run_render();

        let info_layout = self
            .gpu
            .bindgroup_layout("render outline info")
            .with_visibility(wgpu::ShaderStages::VERTEX_FRAGMENT)
            .push_buffer_storage(true, None)
            .build();
        let splats_layout = self
            .gpu
            .bindgroup_layout("render outline splats")
            .with_visibility(wgpu::ShaderStages::VERTEX_FRAGMENT)
            .push_buffer_storage(true, None)
            .build();

        let pipeline_layout = self
            .gpu
            .pipeline_layout("render outline")
            .push_bind_group_layout(Some(&info_layout))
            .push_bind_group_layout(Some(&splats_layout))
            .build();

        let shader = &self.gpu.shaders().render_outline;
        let pipeline = self.gpu.device().create_render_pipeline(
            &RenderPipelineDescriptor {
                label: Some("render outline"),
                layout: Some(&pipeline_layout),
                vertex: VertexState {
                    module: shader,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(FragmentState {
                    module: shader,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(ColorTargetState {
                        format,
                        blend: None,
                        write_mask: ColorWrites::ALL,
                    })],
                }),
                primitive: Default::default(),
                depth_stencil: None,
                multisample: MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            },
        );

        let info_bind_group = self
            .gpu
            .bindgroup("render outline info", &info_layout)
            .push_entry_buffer(&self.buffers.info)
            .build();
        let splats_bind_group = self
            .gpu
            .bindgroup("render outline splats", &splats_layout)
            .push_entry_buffer(&self.buffers.splats)
            .build();

        let mut encoder =
            self.gpu.create_command_encoder(&CommandEncoderDescriptor {
                label: Some("render outlined"),
            });

        encoder.copy_buffer_to_texture(
            TexelCopyBufferInfo {
                buffer: &self.buffers.pixels,
                layout: TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(
                        self.info.canvas.stride * Vec4::STRIDE as u32,
                    ),
                    rows_per_image: None,
                },
            },
            TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: Origin3d { x: 0, y: 0, z: 0 },
                aspect: TextureAspect::All,
            },
            self.info.canvas_extent3d(),
        );

        let view = &texture.create_view(&TextureViewDescriptor {
            label: Some("render outlined"),
            ..Default::default()
        });

        let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("render outlined"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Load,
                    store: Default::default(),
                },
            })],
            ..Default::default()
        });

        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &info_bind_group, &[]);
        pass.set_bind_group(1, &splats_bind_group, &[]);
        pass.draw(0..6, 0..self.splat_count);

        drop(pass);

        let readback = self.gpu.create_buffer(
            "render outline readback",
            (self.info.canvas.stride * self.info.canvas.dims.y) as u64
                * Vec4::STRIDE,
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        );
        encoder.copy_texture_to_buffer(
            TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: Origin3d { x: 0, y: 0, z: 0 },
                aspect: TextureAspect::All,
            },
            TexelCopyBufferInfo {
                buffer: &readback,
                layout: TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(
                        self.info.canvas.stride * Vec4::STRIDE as u32,
                    ),
                    rows_per_image: None,
                },
            },
            self.info.canvas_extent3d(),
        );

        self.gpu.submit([encoder.finish()]);
        self.gpu.poll_blocking();

        let mut img = self
            .gpu
            .buffer_reader::<(), ()>(readback)
            .read_image_blocking(
                &self.gpu,
                self.info.canvas.dims,
                self.info.canvas.stride,
                true,
            );
        img.unpremultiply();

        img
    }

    pub fn read_splats(&self) -> Box<[Splat]> {
        let mut out =
            Vec::<Splat>::with_capacity(self.splat_count as usize);

        self.gpu.read_buffer(self.buffers.splats.clone(), |buffer| {
            let splats: &[gpu_types::Splat] =
                bytemuck::cast_slice(buffer);
            for splat in splats {
                let mut splat = *splat;
                splat.pos /= self.info.canvas.dimsf;
                splat.size /= self.info.canvas.dimsf;
                out.push(splat.into());
            }
        });

        out.into_boxed_slice()
    }

    pub fn read_loss(&self) -> f32 {
        self.gpu.read_buffer_pod(self.buffers.loss.clone())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Config {
    pub lr_scale: f32,
    pub lr: Lr,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            lr_scale: 1.,
            lr: Default::default(),
        }
    }
}

#[macro_export]
macro_rules! learn_dispatch {
    (
        $pass:expr, $self:expr;
        $(
            $pipeline:ident$(@i($indirect_field:ident))?$(@n($workgroups:expr))?$(($imm:expr))?:
            { $($bg_idx:literal = $bg_ident:ident);* $(;)? }
        ),* $(,)?
    ) => {
        $(
            $pass.set_pipeline(&$self.pipelines.$pipeline);
            $(
                $pass.set_bind_group($bg_idx, &$self.bind_groups.$bg_ident, &[]);
            )*
            $(
                $pass.set_immediates(0, bytes_of($imm));
            )?
            $(
                $pass.dispatch_workgroups_indirect(
                    &$self.buffers.bin_overflow,
                    offset_of!(BinOverflow, $indirect_field) as u64,
                );
            )?
            $(
                $pass.dispatch_workgroups(
                    $workgroups.x,
                    $workgroups.y,
                    $workgroups.z
                );
            )?
            learn_write_ts!($pass, $self, $pipeline);
        )*
    };
}

#[macro_export]
macro_rules! learn_write_ts {
    ($pass:expr, $self:expr, $name:ident) => {
        #[cfg(feature = "gpu_profiling")]
        $pass.write_timestamp(
            &$self.timestamp_query_set,
            (offset_of!(Timestamps, $name) as u64 / u64::STRIDE) as u32,
        );
    };
}
