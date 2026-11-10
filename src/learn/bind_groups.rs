use wgpu::BindGroup;

use crate::{
    gpu::GpuContext,
    learn::{buffers::Buffers, pipelines::Pipelines},
};

pub struct BindGroups {
    pub base_ro: BindGroup,

    pub splats_ro: BindGroup,
    pub splats_adam: BindGroup,
    pub splats_redistribute: BindGroup,
    pub splats_cache_fill: BindGroup,

    pub bin_count: BindGroup,
    pub bin_alloc: BindGroup,
    pub bin_fill: BindGroup,
    pub bin_sort: BindGroup,

    pub bin_heads_ro: BindGroup,

    pub bin_data_ro: BindGroup,
    pub bin_data_rw: BindGroup,

    pub canvas_ro: BindGroup,
    pub canvas_rw: BindGroup,

    pub learn_backward: BindGroup,
    pub learn_gradients_ro: BindGroup,

    pub learn_adam: BindGroup,

    pub redistribute_cells: BindGroup,
    pub redistribute_select: BindGroup,
    pub redistribute_move: BindGroup,
}

impl BindGroups {
    pub fn new(
        gpu: &GpuContext,
        pipelines: &Pipelines,
        buffers: &Buffers,
    ) -> Self {
        let base_ro = gpu
            .bindgroup("base_ro", &pipelines.layout_base_ro)
            .push_entry_buffer(&buffers.info)
            .push_entry_buffer(&buffers.reference)
            .build();

        let splats_ro = gpu
            .bindgroup("splats_ro", &pipelines.layout_splats_ro)
            .push_entry_buffer(&buffers.splats)
            .push_entry_buffer(&buffers.splat_caches)
            .build();
        let splats_adam = gpu
            .bindgroup("splats_adam", &pipelines.layout_splats_adam)
            .push_entry_buffer(&buffers.splats)
            .build();
        let splats_redistribute = gpu
            .bindgroup(
                "splats_redistribute",
                &pipelines.layout_splats_redistribute,
            )
            .push_entry_buffer(&buffers.splats)
            .build();
        let splats_cache_fill = gpu
            .bindgroup(
                "splats_cache_fill",
                &pipelines.layout_splats_cache_fill,
            )
            .push_entry_buffer(&buffers.splats)
            .push_entry_buffer(&buffers.splat_caches)
            .build();

        let bin_count = gpu
            .bindgroup("bin_count", &pipelines.layout_bin_count)
            .push_entry_buffer(&buffers.bin_counts)
            .build();
        let bin_alloc = gpu
            .bindgroup("bin_alloc", &pipelines.layout_bin_alloc)
            .push_entry_buffer(&buffers.bin_counts)
            .push_entry_buffer(&buffers.bin_heads)
            .push_entry_buffer(&buffers.bin_bump)
            .push_entry_buffer(&buffers.bin_overflow)
            .build();
        let bin_fill = gpu
            .bindgroup("bin_fill", &pipelines.layout_bin_fill)
            .push_entry_buffer(&buffers.bin_heads)
            .push_entry_buffer(&buffers.bin_cursor)
            .build();
        let bin_sort = gpu
            .bindgroup("bin_sort", &pipelines.layout_bin_sort)
            .push_entry_buffer(&buffers.bin_heads)
            .build();

        let bin_heads_ro = gpu
            .bindgroup("bin_heads_ro", &pipelines.layout_bin_heads_ro)
            .push_entry_buffer(&buffers.bin_heads)
            .build();

        let (bin_data_ro, bin_data_rw) =
            Self::create_bin_data(gpu, pipelines, buffers);

        let canvas_ro = gpu
            .bindgroup("canvas_ro", &pipelines.layout_canvas_ro)
            .push_entry_buffer(&buffers.pixels)
            .push_entry_buffer(&buffers.transmittance)
            .build();
        let canvas_rw = gpu
            .bindgroup("canvas_rw", &pipelines.layout_canvas_rw)
            .push_entry_buffer(&buffers.pixels)
            .push_entry_buffer(&buffers.transmittance)
            .build();

        let learn_backward = gpu
            .bindgroup("learn_backward", &pipelines.layout_learn_backward)
            .push_entry_buffer(&buffers.gradients)
            .push_entry_buffer(&buffers.loss)
            .build();
        let learn_gradients_ro = gpu
            .bindgroup(
                "learn_gradients_ro",
                &pipelines.layout_learn_gradients_ro,
            )
            .push_entry_buffer(&buffers.gradients)
            .build();

        let learn_adam = gpu
            .bindgroup("learn_adam", &pipelines.layout_learn_adam)
            .push_entry_buffer(&buffers.adam_moment1)
            .push_entry_buffer(&buffers.adam_moment2)
            .build();

        let (redistribute_cells, redistribute_select, redistribute_move) =
            Self::create_sampling_cells(gpu, pipelines, buffers);

        Self {
            base_ro,
            splats_ro,
            splats_adam,
            splats_redistribute,
            splats_cache_fill,
            bin_count,
            bin_alloc,
            bin_fill,
            bin_sort,
            bin_heads_ro,
            bin_data_ro,
            bin_data_rw,
            canvas_ro,
            canvas_rw,
            learn_backward,
            learn_gradients_ro,
            learn_adam,
            redistribute_cells,
            redistribute_select,
            redistribute_move,
        }
    }

    pub fn create_bin_data(
        gpu: &GpuContext,
        pipelines: &Pipelines,
        buffers: &Buffers,
    ) -> (BindGroup, BindGroup) {
        let bin_data_ro = gpu
            .bindgroup("bin_data_ro", &pipelines.layout_bin_data_ro)
            .push_entry_buffer(&buffers.bin_data)
            .build();
        let bin_data_rw = gpu
            .bindgroup("bin_data_rw", &pipelines.layout_bin_data_rw)
            .push_entry_buffer(&buffers.bin_data)
            .build();
        (bin_data_ro, bin_data_rw)
    }

    pub fn recreate_bin_data(
        &mut self,
        gpu: &GpuContext,
        pipelines: &Pipelines,
        buffers: &Buffers,
    ) {
        let (bin_data_ro, bin_data_rw) =
            Self::create_bin_data(gpu, pipelines, buffers);
        self.bin_data_ro = bin_data_ro;
        self.bin_data_rw = bin_data_rw;
    }

    pub fn create_sampling_cells(
        gpu: &GpuContext,
        pipelines: &Pipelines,
        buffers: &Buffers,
    ) -> (BindGroup, BindGroup, BindGroup) {
        let redistribute_cells = gpu
            .bindgroup(
                "redistribute_cells",
                &pipelines.layout_redistribute_cells,
            )
            .push_entry_buffer(&buffers.redistribute_sampling_cells)
            .build();
        let redistribute_select = gpu
            .bindgroup(
                "redistribute_select",
                &pipelines.layout_redistribute_select,
            )
            .push_entry_buffer(&buffers.redistribute_sampling_cells)
            .push_entry_buffer(&buffers.redistribute_state)
            .push_entry_buffer(&buffers.redistribute_inverse_cdf)
            .build();
        let redistribute_move = gpu
            .bindgroup(
                "redistribute_move",
                &pipelines.layout_redistribute_move,
            )
            .push_entry_buffer(&buffers.redistribute_sampling_cells)
            .push_entry_buffer(&buffers.redistribute_state)
            .push_entry_buffer(&buffers.redistribute_inverse_cdf)
            .build();
        (redistribute_cells, redistribute_select, redistribute_move)
    }

    pub fn recreate_sampling_cells(
        &mut self,
        gpu: &GpuContext,
        pipelines: &Pipelines,
        buffers: &Buffers,
    ) {
        let (redistribute_cells, redistribute_select, redistribute_move) =
            Self::create_sampling_cells(gpu, pipelines, buffers);
        self.redistribute_cells = redistribute_cells;
        self.redistribute_select = redistribute_select;
        self.redistribute_move = redistribute_move;
    }
}
