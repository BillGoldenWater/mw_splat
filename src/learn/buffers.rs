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

    pub splats: Buffer,
    pub splat_caches: Buffer,

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
    pub transmittance: Buffer,
    pub loss_weights: Buffer,

    /// clear every pass
    pub gradients: Buffer,
    /// clear every pass
    pub loss: Buffer,

    pub adam_moment1: Buffer,
    pub adam_moment2: Buffer,

    pub redistribute_sampling_cells: Buffer,
    pub redistribute_state: Buffer,
    pub redistribute_inverse_cdf: Buffer,
}

impl Buffers {
    pub fn new(
        gpu: &GpuContext,
        info: &gpu_types::Info,
        splat_count: u64,
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

        let splats = gpu.create_buffer(
            "splats",
            gpu_types::Splat::STRIDE * splat_count,
            BufferUsages::STORAGE
                | BufferUsages::COPY_DST
                | BufferUsages::COPY_SRC,
        );
        let splat_caches = gpu.create_buffer(
            "splat_caches",
            gpu_types::STRIDE_SPLAT_CACHE * splat_count,
            BufferUsages::STORAGE,
        );

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
        let transmittance = gpu.create_buffer(
            "transmittance",
            f32::STRIDE * canvas_buffer_cap,
            BufferUsages::STORAGE,
        );
        let loss_weights = gpu.create_buffer(
            "loss_weights",
            f32::STRIDE * canvas_buffer_cap,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
        );

        let gradients = gpu.create_buffer(
            "gradients",
            gpu_types::STRIDE_GRADIENT * splat_count * GRAD_COPIES,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
        );
        let loss = gpu.create_buffer(
            "loss",
            f32::STRIDE,
            BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        );

        let adam_moment1 = gpu.create_buffer(
            "adam_moment1",
            Splat::STRIDE * splat_count,
            BufferUsages::STORAGE,
        );
        let adam_moment2 = gpu.create_buffer(
            "adam_moment1",
            Splat::STRIDE * splat_count,
            BufferUsages::STORAGE,
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
            splats,
            splat_caches,
            bin_counts,
            bin_heads,
            bin_bump,
            bin_overflow,
            bin_cursor,
            bin_data,
            pixels,
            transmittance,
            loss_weights,
            gradients,
            loss,
            adam_moment1,
            adam_moment2,
            redistribute_sampling_cells,
            redistribute_state,
            redistribute_inverse_cdf,
        }
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

    pub fn grow_bin_data(&mut self, gpu: &GpuContext) {
        let old_size = self.bin_data.size();
        let old_cap = old_size.div_ceil(u32::STRIDE);
        let new_cap = old_cap * 2;
        tracing::debug!("grow bin data from {old_cap} to {new_cap}");
        self.bin_data = Self::create_bin_data(gpu, new_cap);
    }

    pub fn grow_sampling_cells(
        &mut self,
        gpu: &GpuContext,
        cell_count: u64,
    ) {
        tracing::debug!("grow sampling cells to {cell_count}");
        self.redistribute_sampling_cells = gpu.create_buffer(
            "sampling_cells",
            STRIDE_SAMPLING_CELL * cell_count,
            BufferUsages::STORAGE,
        );
    }
}
