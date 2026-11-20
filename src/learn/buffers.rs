// SPDX-FileCopyrightText: 2026 Golden_Water
//
// SPDX-License-Identifier: MPL-2.0

use glam::Vec4;
use wgpu::{Buffer, BufferUsages};

use crate::{
    constants::GRAD_COPIES,
    gpu::{GpuContext, GpuLayout},
    learn::gpu_types::{
        self, STRIDE_REDISTRIBUTE_STATE, STRIDE_SAMPLING_CELL, Splat,
    },
};

pub struct Buffers {
    pub info: Buffer,
    pub reference: Buffer,

    pub splat: SplatBuffers,

    /// clear every pass
    pub bin_counts: Buffer,
    pub bin_heads: Buffer,
    /// clear every pass
    pub bin_bump: Buffer,
    pub bin_overflow: Buffer,
    /// clear every pass
    pub bin_cursor: Buffer,

    pub bin_data: Buffer,

    pub pixels: Buffer,
    pub log_transmittance: Buffer,
    pub loss_weights: Buffer,

    pub tile_loss: Buffer,

    pub redistribute_sampling_cells: Buffer,
    pub redistribute_state: Buffer,
    pub redistribute_inverse_cdf: Buffer,
}

impl Buffers {
    pub fn new(
        gpu: &GpuContext,
        info: &gpu_types::Info,
        splat_capacity: u64,
        bin_capacity: u64,
    ) -> Self {
        let canvas_buffer_cap =
            info.canvas.stride as u64 * info.canvas.dims.y as u64;
        let bin_count = info.tile_dims.element_product() as u64;

        let info = gpu.create_buffer(
            "info",
            gpu_types::Info::STRIDE,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
        );
        let reference = gpu.create_buffer(
            "reference",
            Vec4::STRIDE * canvas_buffer_cap,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
        );

        let splat = Self::create_splat_buffers(gpu, splat_capacity);

        let bin_counts = gpu.create_buffer(
            "bin_counts",
            u32::STRIDE * bin_count,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
        );
        let bin_heads = gpu.create_buffer(
            "bin_heads",
            gpu_types::STRIDE_BIN_HEADER * bin_count,
            BufferUsages::STORAGE,
        );
        let bin_bump = gpu.create_buffer(
            "bin_bump",
            u32::STRIDE,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
        );
        let bin_overflow = gpu.create_buffer(
            "bin_overflow",
            gpu_types::BinOverflow::STRIDE,
            BufferUsages::STORAGE
                | BufferUsages::COPY_DST
                | BufferUsages::INDIRECT
                | BufferUsages::COPY_SRC,
        );
        let bin_cursor = gpu.create_buffer(
            "bin_cursor",
            u32::STRIDE * bin_count,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
        );

        let bin_data = Self::create_bin_data(gpu, bin_capacity);

        let pixels = gpu.create_buffer(
            "pixels",
            Vec4::STRIDE * canvas_buffer_cap,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        );
        let log_transmittance = gpu.create_buffer(
            "log_transmittance",
            f32::STRIDE * canvas_buffer_cap,
            BufferUsages::STORAGE,
        );
        let loss_weights = gpu.create_buffer(
            "loss_weights",
            f32::STRIDE * canvas_buffer_cap,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
        );

        let tile_loss = gpu.create_buffer(
            "tile_loss",
            f32::STRIDE * bin_count,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        );

        let redistribute_sampling_cells = gpu.create_buffer(
            "sampling_cells",
            STRIDE_SAMPLING_CELL,
            BufferUsages::STORAGE,
        );
        let redistribute_state = gpu.create_buffer(
            "redistribute_state",
            STRIDE_REDISTRIBUTE_STATE,
            BufferUsages::STORAGE,
        );
        let redistribute_inverse_cdf = gpu.create_buffer(
            "inverse_cdf",
            u32::STRIDE * crate::constants::INVERSE_CDF_SIZE as u64,
            BufferUsages::STORAGE,
        );

        Self {
            info,
            reference,
            splat,
            bin_counts,
            bin_heads,
            bin_bump,
            bin_overflow,
            bin_cursor,
            bin_data,
            pixels,
            log_transmittance,
            loss_weights,
            tile_loss,
            redistribute_sampling_cells,
            redistribute_state,
            redistribute_inverse_cdf,
        }
    }

    pub fn create_splat_buffers(
        gpu: &GpuContext,
        capacity: u64,
    ) -> SplatBuffers {
        let splats = gpu.create_buffer(
            "splats",
            Splat::STRIDE * capacity,
            BufferUsages::STORAGE
                | BufferUsages::COPY_DST
                | BufferUsages::COPY_SRC,
        );
        let splat_caches = gpu.create_buffer(
            "splat_caches",
            gpu_types::STRIDE_SPLAT_CACHE * capacity,
            BufferUsages::STORAGE,
        );
        let gradients = gpu.create_buffer(
            "gradients",
            gpu_types::STRIDE_GRADIENT * capacity * GRAD_COPIES,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
        );
        let adam_moment1 = gpu.create_buffer(
            "adam_moment1",
            Splat::STRIDE * capacity,
            BufferUsages::STORAGE
                | BufferUsages::COPY_DST
                | BufferUsages::COPY_SRC,
        );
        let adam_moment2 = gpu.create_buffer(
            "adam_moment2",
            Splat::STRIDE * capacity,
            BufferUsages::STORAGE
                | BufferUsages::COPY_DST
                | BufferUsages::COPY_SRC,
        );

        SplatBuffers {
            splats,
            splat_caches,
            gradients,
            adam_moment1,
            adam_moment2,
        }
    }

    pub fn realloc_splats(
        &mut self,
        gpu: &GpuContext,
        new_capacity: u64,
    ) {
        tracing::debug!("realloc splats to {new_capacity}");
        let splat = Self::create_splat_buffers(gpu, new_capacity);

        let mut encoder =
            gpu.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("realloc splats"),
            });

        let copy_size = self.splat.splats.size().min(splat.splats.size());
        encoder.copy_buffer_to_buffer(
            &self.splat.splats,
            0,
            &splat.splats,
            0,
            copy_size,
        );

        let copy_size = self
            .splat
            .adam_moment1
            .size()
            .min(splat.adam_moment1.size());
        encoder.copy_buffer_to_buffer(
            &self.splat.adam_moment1,
            0,
            &splat.adam_moment1,
            0,
            copy_size,
        );

        let copy_size = self
            .splat
            .adam_moment2
            .size()
            .min(splat.adam_moment2.size());
        encoder.copy_buffer_to_buffer(
            &self.splat.adam_moment2,
            0,
            &splat.adam_moment2,
            0,
            copy_size,
        );

        gpu.submit([encoder.finish()]);

        self.splat = splat;
    }

    pub fn create_bin_data(
        gpu: &GpuContext,
        bin_capacity: u64,
    ) -> Buffer {
        gpu.create_buffer(
            "bin_data",
            u32::STRIDE * bin_capacity,
            BufferUsages::STORAGE,
        )
    }

    pub fn realloc_bin_data(
        &mut self,
        gpu: &GpuContext,
        new_capacity: u64,
    ) {
        tracing::debug!("realloc bin data to {new_capacity}");
        self.bin_data = Self::create_bin_data(gpu, new_capacity);
    }

    pub fn realloc_sampling_cells(
        &mut self,
        gpu: &GpuContext,
        new_capacity: u64,
    ) {
        tracing::debug!("realloc sampling cells to {new_capacity}");
        self.redistribute_sampling_cells = gpu.create_buffer(
            "sampling_cells",
            STRIDE_SAMPLING_CELL * new_capacity,
            BufferUsages::STORAGE,
        );
    }
}

pub struct SplatBuffers {
    pub splats: Buffer,
    pub splat_caches: Buffer,

    /// clear every pass
    pub gradients: Buffer,

    pub adam_moment1: Buffer,
    pub adam_moment2: Buffer,
}
