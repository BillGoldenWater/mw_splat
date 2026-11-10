use wgpu::{BindGroupLayout, ComputePipeline};

use crate::{
    gpu::{GpuContext, GpuLayout},
    learn::gpu_types::{
        ImmAdam, ImmBinAlloc, ImmRedistributeCells, ImmRedistributeMove,
        ImmRedistributeSelect,
    },
};

pub struct Pipelines {
    // info, reference
    pub layout_base_ro: BindGroupLayout,

    // splats, splats cache
    pub layout_splats_ro: BindGroupLayout,
    pub layout_splats_adam: BindGroupLayout,
    pub layout_splats_redistribute: BindGroupLayout,
    pub layout_splats_cache_fill: BindGroupLayout,

    pub layout_bin_count: BindGroupLayout,
    pub layout_bin_alloc: BindGroupLayout,
    pub layout_bin_fill: BindGroupLayout,
    pub layout_bin_sort: BindGroupLayout,

    pub layout_bin_heads_ro: BindGroupLayout,

    pub layout_bin_data_ro: BindGroupLayout,
    pub layout_bin_data_rw: BindGroupLayout,

    // pixels, transmittance
    pub layout_canvas_ro: BindGroupLayout,
    pub layout_canvas_rw: BindGroupLayout,

    // gradients, loss
    pub layout_learn_backward: BindGroupLayout,
    // gradients
    pub layout_learn_gradients_ro: BindGroupLayout,

    // gradients, loss
    pub layout_learn_adam: BindGroupLayout,

    // cells
    pub layout_redistribute_cells: BindGroupLayout,
    // cells, state, inverse CDF
    pub layout_redistribute_select: BindGroupLayout,
    // cells, state, inverse CDF
    pub layout_redistribute_move: BindGroupLayout,

    pub bin_count: ComputePipeline,
    pub bin_alloc: ComputePipeline,
    // bin_fill, bin_sort, splats_cache_fill,
    // render, learn_backward, learn_adam
    // gated by bin_alloc
    pub bin_fill: ComputePipeline,
    pub bin_sort: ComputePipeline,

    pub splats_cache_fill: ComputePipeline,

    pub render: ComputePipeline,

    pub learn_backward: ComputePipeline,
    pub learn_adam: ComputePipeline,

    pub redistribute_cells: ComputePipeline,
    pub redistribute_select: ComputePipeline,
    pub redistribute_move: ComputePipeline,
}

impl Pipelines {
    pub fn new(gpu: &GpuContext) -> Self {
        let layout_base_ro = gpu
            .bindgroup_layout("base_ro")
            .push_buffer_storage(true, None)
            .push_buffer_storage(true, None)
            .build();

        let layout_splats_ro = gpu
            .bindgroup_layout("splats_ro")
            .push_buffer_storage(true, None)
            .push_buffer_storage(true, None)
            .build();
        let layout_splats_adam = gpu
            .bindgroup_layout("splats_adam")
            .push_buffer_storage(false, None)
            .build();
        let layout_splats_redistribute = gpu
            .bindgroup_layout("splats_redistribute")
            .push_buffer_storage(false, None)
            .build();
        let layout_splats_cache_fill = gpu
            .bindgroup_layout("splats_cache_fill")
            .push_buffer_storage(true, None)
            .push_buffer_storage(false, None)
            .build();

        let layout_bin_count = gpu
            .bindgroup_layout("bin_count")
            .push_buffer_storage(false, None)
            .build();
        let layout_bin_alloc = gpu
            .bindgroup_layout("bin_alloc")
            .push_buffer_storage(true, None)
            .push_buffer_storage(false, None)
            .push_buffer_storage(false, None)
            .push_buffer_storage(false, None)
            .build();
        let layout_bin_fill = gpu
            .bindgroup_layout("bin_fill")
            .push_buffer_storage(true, None)
            .push_buffer_storage(false, None)
            .build();
        let layout_bin_sort = gpu
            .bindgroup_layout("bin_sort")
            .push_buffer_storage(true, None)
            .build();

        let layout_bin_heads_ro = gpu
            .bindgroup_layout("bin_heads_ro")
            .push_buffer_storage(true, None)
            .build();

        let layout_bin_data_ro = gpu
            .bindgroup_layout("bin_data_ro")
            .push_buffer_storage(true, None)
            .build();
        let layout_bin_data_rw = gpu
            .bindgroup_layout("bin_data_rw")
            .push_buffer_storage(false, None)
            .build();

        let layout_canvas_rw = gpu
            .bindgroup_layout("canvas_rw")
            .push_buffer_storage(false, None)
            .push_buffer_storage(false, None)
            .build();
        let layout_canvas_ro = gpu
            .bindgroup_layout("canvas_ro")
            .push_buffer_storage(true, None)
            .push_buffer_storage(true, None)
            .build();

        let layout_learn_backward = gpu
            .bindgroup_layout("learn_backward")
            .push_buffer_storage(false, None)
            .push_buffer_storage(false, None)
            .build();
        let layout_learn_gradients_ro = gpu
            .bindgroup_layout("learn_gradients")
            .push_buffer_storage(true, None)
            .build();

        let layout_learn_adam = gpu
            .bindgroup_layout("learn_adam")
            .push_buffer_storage(false, None)
            .push_buffer_storage(false, None)
            .build();

        let layout_redistribute_cells = gpu
            .bindgroup_layout("redistribute_cells")
            .push_buffer_storage(false, None)
            .build();
        let layout_redistribute_select = gpu
            .bindgroup_layout("redistribute_select")
            .push_buffer_storage(true, None)
            .push_buffer_storage(false, None)
            .push_buffer_storage(false, None)
            .build();
        let layout_redistribute_move = gpu
            .bindgroup_layout("redistribute_move")
            .push_buffer_storage(true, None)
            .push_buffer_storage(true, None)
            .push_buffer_storage(true, None)
            .build();

        let bin_count = gpu
            .pipeline_layout("bin_count")
            .push_bind_group_layout(Some(&layout_base_ro))
            .push_bind_group_layout(Some(&layout_splats_ro))
            .push_bind_group_layout(Some(&layout_bin_count))
            .build_compute_pipeline(
                &gpu.shaders().bin_count,
                "bin_count",
            );
        let bin_alloc = gpu
            .pipeline_layout("bin_alloc")
            .push_bind_group_layout(None)
            .push_bind_group_layout(None)
            .push_bind_group_layout(Some(&layout_bin_alloc))
            .push_bind_group_layout(Some(&layout_bin_data_ro))
            .with_immediate_size(ImmBinAlloc::SIZE as _)
            .build_compute_pipeline(
                &gpu.shaders().bin_alloc,
                "bin_alloc",
            );
        let bin_fill = gpu
            .pipeline_layout("bin_fill")
            .push_bind_group_layout(Some(&layout_base_ro))
            .push_bind_group_layout(Some(&layout_splats_ro))
            .push_bind_group_layout(Some(&layout_bin_fill))
            .push_bind_group_layout(Some(&layout_bin_data_rw))
            .build_compute_pipeline(&gpu.shaders().bin_fill, "bin_fill");
        let bin_sort = gpu
            .pipeline_layout("bin_sort")
            .push_bind_group_layout(None)
            .push_bind_group_layout(None)
            .push_bind_group_layout(Some(&layout_bin_sort))
            .push_bind_group_layout(Some(&layout_bin_data_rw))
            .build_compute_pipeline(&gpu.shaders().bin_sort, "bin_sort");

        let splats_cache_fill = gpu
            .pipeline_layout("splats_cache_fill")
            .push_bind_group_layout(None)
            .push_bind_group_layout(Some(&layout_splats_cache_fill))
            .build_compute_pipeline(
                &gpu.shaders().splats_cache_fill,
                "splats_cache_fill",
            );

        let render = gpu
            .pipeline_layout("render")
            .push_bind_group_layout(Some(&layout_base_ro))
            .push_bind_group_layout(Some(&layout_splats_ro))
            .push_bind_group_layout(Some(&layout_bin_heads_ro))
            .push_bind_group_layout(Some(&layout_bin_data_ro))
            .push_bind_group_layout(Some(&layout_canvas_rw))
            .build_compute_pipeline(&gpu.shaders().render, "render");

        let learn_backward = gpu
            .pipeline_layout("learn_backward")
            .push_bind_group_layout(Some(&layout_base_ro))
            .push_bind_group_layout(Some(&layout_splats_ro))
            .push_bind_group_layout(Some(&layout_bin_heads_ro))
            .push_bind_group_layout(Some(&layout_bin_data_ro))
            .push_bind_group_layout(Some(&layout_canvas_ro))
            .push_bind_group_layout(Some(&layout_learn_backward))
            .build_compute_pipeline(
                &gpu.shaders().learn_backward,
                "backward",
            );
        let learn_adam = gpu
            .pipeline_layout("learn_adam")
            .push_bind_group_layout(Some(&layout_base_ro))
            .push_bind_group_layout(Some(&layout_splats_adam))
            .push_bind_group_layout(Some(&layout_learn_gradients_ro))
            .push_bind_group_layout(Some(&layout_learn_adam))
            .with_immediate_size(ImmAdam::SIZE as _)
            .build_compute_pipeline(&gpu.shaders().learn_adam, "adam");

        let redistribute_cells = gpu
            .pipeline_layout("redistribute_cells")
            .push_bind_group_layout(Some(&layout_base_ro))
            .push_bind_group_layout(Some(&layout_canvas_ro))
            .push_bind_group_layout(Some(&layout_redistribute_cells))
            .with_immediate_size(ImmRedistributeCells::SIZE as _)
            .build_compute_pipeline(
                &gpu.shaders().redistribute_cells,
                "redistribute_cells",
            );
        let redistribute_select = gpu
            .pipeline_layout("redistribute_select")
            .push_bind_group_layout(None)
            .push_bind_group_layout(Some(&layout_redistribute_select))
            .with_immediate_size(ImmRedistributeSelect::SIZE as _)
            .build_compute_pipeline(
                &gpu.shaders().redistribute_select,
                "redistribute_select",
            );
        let redistribute_move = gpu
            .pipeline_layout("redistribute_move")
            .push_bind_group_layout(Some(&layout_base_ro))
            .push_bind_group_layout(Some(&layout_splats_redistribute))
            .push_bind_group_layout(Some(&layout_redistribute_move))
            .with_immediate_size(ImmRedistributeMove::SIZE as _)
            .build_compute_pipeline(
                &gpu.shaders().redistribute_move,
                "redistribute_move",
            );

        Self {
            layout_base_ro,
            layout_splats_ro,
            layout_splats_adam,
            layout_splats_redistribute,
            layout_splats_cache_fill,
            layout_bin_count,
            layout_bin_alloc,
            layout_bin_fill,
            layout_bin_sort,
            layout_bin_heads_ro,
            layout_bin_data_ro,
            layout_bin_data_rw,
            layout_canvas_ro,
            layout_canvas_rw,
            layout_learn_backward,
            layout_learn_gradients_ro,
            layout_learn_adam,
            layout_redistribute_cells,
            layout_redistribute_select,
            layout_redistribute_move,
            bin_count,
            bin_alloc,
            bin_fill,
            bin_sort,
            splats_cache_fill,
            render,
            learn_backward,
            learn_adam,
            redistribute_cells,
            redistribute_select,
            redistribute_move,
        }
    }
}
