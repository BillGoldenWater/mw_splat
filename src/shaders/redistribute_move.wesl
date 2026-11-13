/*
 * SPDX-FileCopyrightText: 2026 Golden_Water
 *
 * SPDX-License-Identifier: MPL-2.0
 */

import package::libs::types::{Info, Splat};
import package::libs::redistribute::{RedistributeState, SamplingCell, redistribute_hash};
import package::libs::constants::{INVERSE_CDF_SIZE, WG_SIZE};

@group(0) @binding(0)
var<storage, read> info: Info;
@group(0) @binding(1)
var<storage, read> reference: array<vec4f>;

@group(1) @binding(0)
var<storage, read_write> splats: array<Splat>;

@group(2) @binding(0)
var<storage, read> sampling_cells: array<SamplingCell>;
@group(2) @binding(1)
var<storage, read> state: RedistributeState;
@group(2) @binding(2)
var<storage, read> inverse_cdf: array<u32>;

@group(3) @binding(0)
var<storage, read_write> moment1: array<Splat>;
@group(3) @binding(1)
var<storage, read_write> moment2: array<Splat>;

struct Imm {
    grid_dims: vec2u,
    cell_size: u32,
    splat_size: f32,
    seed: u32,
}
var<immediate> imm: Imm;

const SPAN_MIN: f32 = 0.5;
const SPAN_MAX: f32 = 1.5;
const MIN_VISIBLE_ALPHA: f32 = 0.01;
const MIN_BACKGROUND_ALPHA: f32 = 0.05;
const MOVED_ORDER_BASE: u32 = 1u << 31u;

fn hot_pixel_pos(hot_px_idx: u32) -> vec2f {
    return vec2f(vec2u(hot_px_idx % info.canvas.stride, hot_px_idx / info.canvas.stride)) + 0.5;
}

fn coverage(splat: Splat) -> f32 {
    let sample_pos = vec2u(clamp(splat.pos, vec2f(0.5), vec2f(info.canvas.dims) - 0.5));
    let extent = sqrt(splat.size.x * splat.scale.x * splat.size.y * splat.scale.y);
    let spans_sample = smoothstep(SPAN_MIN, SPAN_MAX, extent);
    let visible = smoothstep(0.0, MIN_VISIBLE_ALPHA, splat.color.w);
    let has_background = smoothstep(
            0.0,
            MIN_BACKGROUND_ALPHA,
            reference[sample_pos.y * info.canvas.stride + sample_pos.x].a,
        );
    return spans_sample * visible * has_background;
}

@compute @workgroup_size(WG_SIZE)
fn redistribute_move(
    @builtin(global_invocation_id) id: vec3u,
    @builtin(num_workgroups) wg_num: vec3u,
) {
    let splat_idx = id.x + id.y * WG_SIZE * wg_num.x;
    if splat_idx >= arrayLength(&splats) {
        return;
    }

    let splat = &splats[splat_idx];
    let off_grid = any(splat.pos < vec2f(0.)) || any(splat.pos >= info.canvas.dimsf);

    let uncovered = 1.0 - coverage(*splat);
    var should_move = off_grid || redistribute_hash(splat_idx, 7u) < uncovered;
    if !should_move {
        let cell_coord = vec2i(splat.pos / f32(imm.cell_size));
        let cell_idx = u32(cell_coord.x) + u32(cell_coord.y) * imm.grid_dims.x;
        let cell_loss = sampling_cells[cell_idx].loss;
        let below_boundary = cell_loss < state.boundary_lo;
        let boundary_donor = cell_loss < state.boundary_hi
            && redistribute_hash(splat_idx, imm.seed) < state.boundary_frac;
        should_move = below_boundary || boundary_donor;
    }
    if !should_move {
        splat.order = splat_idx;
        return;
    }

    let cdf_rand = redistribute_hash(splat_idx, 101u);
    let cdf_slot = min(u32(cdf_rand * f32(INVERSE_CDF_SIZE - 1u)), INVERSE_CDF_SIZE - 1u);
    let target_cell = inverse_cdf[cdf_slot];
    let target_hot_px = sampling_cells[target_cell].hot_px_idx;
    var destination = hot_pixel_pos(target_hot_px);
    destination = clamp(destination, vec2f(0.5), vec2f(info.canvas.dims) - 0.5);

    let dest_px = u32(destination.y) * info.canvas.stride + u32(destination.x);
    let target_color = reference[dest_px];

    splat.pos = destination;
    splat.size = vec2f(imm.splat_size);
    splat.color = target_color;
    splat.scale = vec2f(1.0);
    splat.rot = 0.0;
    splat.order = MOVED_ORDER_BASE + splat_idx;
    moment1[splat_idx] = Splat();
    moment2[splat_idx] = Splat();
}
