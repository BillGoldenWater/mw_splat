/*
 * SPDX-FileCopyrightText: 2026 Golden_Water
 *
 * SPDX-License-Identifier: MPL-2.0
 */

// Movement half of the redistribution epoch.
//
// A splat whose cell carries less than the donor quantile is moved onto the hot
// pixel of a cell drawn from the loss distribution, then reborn with the
// reference colour so the fit does not have to rediscover it.
//
// A splat that covers no sample centre - footprint between samples, faded out,
// standing on transparent space or already gone from the canvas - draws no
// gradient, so it is free to move at the full rate.

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

struct Imm {
    grid_dims: vec2u,
    cell_size: u32,
    splat_size: f32,
    seed: u32,
}
var<immediate> imm: Imm;

fn hot_pixel_pos(hot_px_idx: u32) -> vec2f {
    return vec2f(vec2u(hot_px_idx % info.canvas.stride, hot_px_idx / info.canvas.stride)) + 0.5;
}

fn coverage(splat_idx: u32) -> f32 {
    let splat = splats[splat_idx];
    let sample_pos = vec2u(clamp(splat.pos, vec2f(0.5), vec2f(info.canvas.dims) - 1.5));
    let spans_sample = smoothstep(0.5, 1.5, min(splat.size.x, splat.size.y));
    let visible = smoothstep(0.0, 0.01, splat.color.w);
    let has_background = smoothstep(
            0.0,
            0.05,
            reference[sample_pos.y * info.canvas.stride + sample_pos.x].a,
        );
    return spans_sample * visible * has_background;
}

@compute @workgroup_size(WG_SIZE)
fn redistribute_move(@builtin(global_invocation_id) id: vec3u) {
    let splat_idx = id.x;
    if splat_idx >= arrayLength(&splats) {
        return;
    }

    let cell_coord = vec2i(splats[splat_idx].pos / f32(imm.cell_size));
    let off_grid = any(cell_coord < vec2i(0)) || any(cell_coord >= vec2i(imm.grid_dims));

    let uncovered = 1.0 - coverage(splat_idx);
    var should_move = off_grid || redistribute_hash(splat_idx, 7u) < uncovered;
    if !should_move && !off_grid {
        let cell_idx = u32(cell_coord.x) + u32(cell_coord.y) * imm.grid_dims.x;
        let cell_loss = sampling_cells[cell_idx].loss;
        let below_boundary = cell_loss < state.boundary_lo;
        let boundary_donor = cell_loss < state.boundary_hi
            && redistribute_hash(splat_idx, imm.seed) < state.boundary_frac;
        should_move = below_boundary || boundary_donor;
    }
    if !should_move {
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

    splats[splat_idx].pos = destination;
    splats[splat_idx].size = vec2f(imm.splat_size);
    splats[splat_idx].color = target_color;
    splats[splat_idx].fe = vec2f(0.5);
    splats[splat_idx].rot = 0.0;
}
