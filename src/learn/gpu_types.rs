use glam::{UVec2, UVec4, Vec2, Vec4};
use wgpu::Extent3d;

#[cfg(feature = "gpu_profiling")]
use crate::gpu::{GpuContext, GpuLayout};
use crate::{gpu::align_to, ir};

#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C, align(16))]
pub struct Info {
    // offset  0, size 48, align 16
    pub canvas: Canvas,
    // offset 48, size 8, align 8
    pub tile_dims: UVec2,
    // offset 56, size 24, align 4
    pub lr: Lr,
}

#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C, align(16))]
pub struct Canvas {
    // offset  0, size 8, align 8
    pub dims: UVec2,
    // offset  8, size 8, align 8
    pub dimsf: Vec2,
    // offset 16, size 4, align 4
    pub stride: u32,
    // offset 20
    pub _pad: [u8; 12],
    // offset 32, size 16, align 16
    pub background: Vec4,
}

impl Info {
    pub fn canvas_extent3d(&self) -> Extent3d {
        Extent3d {
            width: self.canvas.dims.x,
            height: self.canvas.dims.y,
            depth_or_array_layers: 1,
        }
    }
}

#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C, align(4))]
pub struct Lr {
    pub position: f32,
    pub scale: f32,
    pub rotation: f32,
    pub color: f32,
    pub alpha: f32,
    pub feather: f32,
}

impl Default for Lr {
    fn default() -> Self {
        Self {
            position: 0.95,
            scale: 0.02,
            rotation: 0.025,
            color: 0.045,
            alpha: 0.0085,
            feather: 0.0065,
        }
    }
}

#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C, align(16))]
pub struct Splat {
    // offset 0, size 8, align 8
    pub pos: Vec2,
    // offset 8, size 8, align 8
    pub size: Vec2,
    // offset 16, size 16, align 16
    pub color: Vec4,
    // offset 32, size 8, align 8
    pub fe: Vec2,
    // offset 40, size 4, align 4
    pub rot: f32,
    // offset 44
    pub _pad: [u8; 4],
}

impl From<&ir::Splat> for Splat {
    fn from(value: &ir::Splat) -> Self {
        let ir::Splat {
            pos,
            size,
            rot,
            color,
            fe,
        } = *value;

        Self {
            pos,
            size,
            color,
            fe,
            rot,
            _pad: Default::default(),
        }
    }
}

impl From<ir::Splat> for Splat {
    fn from(value: ir::Splat) -> Self {
        Self::from(&value)
    }
}

impl From<Splat> for ir::Splat {
    fn from(value: Splat) -> Self {
        let Splat {
            pos,
            size,
            color,
            fe,
            rot,
            _pad,
        } = value;

        Self {
            pos,
            size,
            rot,
            color,
            fe,
        }
    }
}

#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C, align(16))]
pub struct BinOverflow {
    // offset 0, size 4, align 4
    pub overflow: u32,
    // offset 4, size 4, align 4
    pub iter_num: u32,
    // offset 8, size 8, align 1
    pub _pad: [u8; 8],
    // offset 16, size 16, align 16
    pub i_bin_fill: UVec4,
    pub i_bin_sort: UVec4,
    pub i_splats_cache_fill: UVec4,
    pub i_render: UVec4,
    pub i_learn_backward: UVec4,
    pub i_learn_adam: UVec4,
}

#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C, align(4))]
pub struct ImmBinAlloc {
    pub iter_num: u32,
}

#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C, align(4))]
pub struct ImmAdam {
    pub lr_scale: f32,
}

#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C, align(8))]
pub struct ImmRedistributeCells {
    // offset 0, size 8, align 8
    pub grid_dims: UVec2,
    // offset 8, size 4, align 4
    pub cell_size: u32,
    // offset 12
    pub _pad: [u8; 4],
}

#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C, align(8))]
pub struct ImmRedistributeSelect {
    // offset 0, size 8, align 8
    pub grid_dims: UVec2,
    // offset 8, size 4, align 4
    pub move_count: u32,
    // offset 12
    pub _pad: [u8; 4],
}

#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C, align(8))]
pub struct ImmRedistributeMove {
    // offset 0, size 8, align 8
    pub grid_dims: UVec2,
    // offset 8, size 4, align 4
    pub cell_size: u32,
    // offset 12, size 4, align 4
    pub splat_size: f32,
    // offset 16, size 4, align 4
    pub seed: u32,
    // offset 20
    pub _pad: [u8; 4],
}

pub const STRIDE_SPLAT_CACHE: u64 = align_to(4 + 4 + 8, 8);
pub const STRIDE_BIN_HEADER: u64 = align_to(4 + 4, 4);
pub const STRIDE_GRADIENT: u64 = align_to(4 * 11, 4);
pub const STRIDE_SAMPLING_CELL: u64 = align_to(4 + 4, 4);
pub const STRIDE_REDISTRIBUTE_STATE: u64 = align_to(4 + 4 + 4, 4);

#[cfg(feature = "gpu_profiling")]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C, align(8))]
pub struct Timestamps {
    pub begin: u64,
    // end of the dispatch
    pub bin_count: u64,
    pub bin_alloc: u64,
    pub bin_fill: u64,
    pub bin_sort: u64,
    pub splats_cache_fill: u64,
    pub render: u64,
    pub learn_backward: u64,
    pub learn_adam: u64,
    pub redistribute_cells: u64,
    pub redistribute_select: u64,
    pub redistribute_move: u64,
}

#[cfg(feature = "gpu_profiling")]
impl Timestamps {
    pub const COUNT: u32 = (Self::SIZE / u64::STRIDE) as u32;

    pub fn log(&self, gpu: &GpuContext, iter_num: u32) {
        use std::{fmt::Write, time::Duration};

        let period = gpu.queue().get_timestamp_period();

        let mut out = String::with_capacity(300);
        let mut prev = self.begin;

        macro_rules! write_fmt {
            ($name:ident) => {{
                let cur = self.$name;
                let dur = cur - prev;
                prev = cur;
                let dur = Duration::from_secs_f32(
                    dur as f32 * period / 1_000_000_000.,
                );
                write!(out, " {}={dur:.2?}", stringify!($name)).unwrap();
            }};
        }
        write_fmt!(bin_count);
        write_fmt!(bin_alloc);
        write_fmt!(bin_fill);
        write_fmt!(bin_sort);
        write_fmt!(splats_cache_fill);
        write_fmt!(render);
        write_fmt!(learn_backward);
        write_fmt!(learn_adam);
        if self.redistribute_cells > self.learn_adam {
            write_fmt!(redistribute_cells);
            write_fmt!(redistribute_select);
            write_fmt!(redistribute_move);
        }
        let _ = prev;

        let total = self.learn_adam - self.begin;
        let dur = Duration::from_secs_f32(
            total as f32 * period / 1_000_000_000.,
        );

        tracing::debug!("durations({iter_num}={dur:.2?}):{out}");
    }
}
