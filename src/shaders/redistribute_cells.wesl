/*
 * SPDX-FileCopyrightText: 2026 Golden_Water
 *
 * SPDX-License-Identifier: MPL-2.0
 */

import package::libs::types::Info;
import package::libs::redistribute::SamplingCell;
import package::libs::constants::WG_SIZE_2D;

@group(0) @binding(0)
var<storage, read> info: Info;
@group(0) @binding(1)
var<storage, read> reference: array<vec4f>;

@group(1) @binding(0)
var<storage, read> pixels: array<vec4f>;
@group(1) @binding(2)
var<storage, read> loss_weights: array<f32>;

@group(2) @binding(0)
var<storage, read_write> sampling_cells: array<SamplingCell>;

struct Imm {
    grid_dims: vec2u,
    cell_size: u32,
}
var<immediate> imm: Imm;

@compute @workgroup_size(WG_SIZE_2D, WG_SIZE_2D)
fn redistribute_cells(@builtin(global_invocation_id) id: vec3u) {
    if any(id.xy >= imm.grid_dims) {
        return;
    }

    let px_base = id.xy * imm.cell_size;
    let px_max = min(px_base + vec2u(imm.cell_size), info.canvas.dims);
    var loss_sum = 0.0;
    var hot_px_idx = 0u;
    var peak = -1.0;
    for(var y = px_base.y; y < px_max.y; y++) {
        for(var x = px_base.x; x < px_max.x; x++) {
            let px_idx = x + y * info.canvas.stride;
            let weight = loss_weights[px_idx];
            let px = pixels[px_idx];
            let tgt = reference[px_idx];
            let black = px.rgb - tgt.rgb;
            let white = black + vec3f(tgt.a - px.a);
            let loss = (dot(black, black) + dot(white, white)) / 6.0 * weight;
            loss_sum += loss;
            if loss > peak {
                peak = loss;
                hot_px_idx = px_idx;
            }
        }
    }
    sampling_cells[id.x + id.y * imm.grid_dims.x] = SamplingCell(loss_sum, hot_px_idx);
}
