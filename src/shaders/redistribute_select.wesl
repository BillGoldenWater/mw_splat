/*
 * SPDX-FileCopyrightText: 2026 Golden_Water
 *
 * SPDX-License-Identifier: MPL-2.0
 */

// Selection half of the redistribution epoch: histogram the cell peaks for the
// donor threshold, then build the destination inverse CDF.

import package::libs::redistribute::{RedistributeState, SamplingCell};
import package::libs::constants::INVERSE_CDF_SIZE;

@group(1) @binding(0)
var<storage, read> sampling_cells: array<SamplingCell>;
@group(1) @binding(1)
var<storage, read_write> state: RedistributeState;
@group(1) @binding(2)
var<storage, read_write> inverse_cdf: array<u32>;

struct Imm {
    grid_dims: vec2u,
    move_count: u32,
}
var<immediate> imm: Imm;

const BINS: u32 = 512u;

var<workgroup> hist: array<u32, BINS>;

@compute @workgroup_size(1)
fn redistribute_select() {
    let cell_count = imm.grid_dims.x * imm.grid_dims.y;
    var total_peak = 0.0;
    var max_peak = 0.0;
    for(var cell = 0u; cell < cell_count; cell++) {
        total_peak += sampling_cells[cell].peak;
        max_peak = max(max_peak, sampling_cells[cell].peak);
    }

    let bin_scale = select(0.0, f32(BINS - 1u) / max_peak, max_peak > 0.0);
    for(var cell = 0u; cell < cell_count; cell++) {
        let bin_idx = min(u32(sampling_cells[cell].peak * bin_scale), BINS - 1u);
        hist[bin_idx] += 1u;
    }

    var cells_below = 0u;
    var bin = 0u;
    while bin + 1u < BINS {
        let bin_cells = hist[bin];
        if cells_below + bin_cells > imm.move_count {
            break;
        }
        cells_below += bin_cells;
        bin += 1u;
    }

    let boundary_bin_cells = max(hist[bin], 1u);
    let boundary_frac = clamp(
            f32(imm.move_count - cells_below) / f32(boundary_bin_cells),
            0.0,
            1.0,
        );

    state.boundary_lo = f32(bin) / max(bin_scale, 1e-20);
    state.boundary_hi = f32(bin + 1u) / max(bin_scale, 1e-20);
    state.boundary_frac = boundary_frac;

    var cum_peak = 0.0;
    var cell_idx = 0u;
    for(var slot = 0u; slot < INVERSE_CDF_SIZE; slot++) {
        let target_peak = total_peak * f32(slot) / f32(INVERSE_CDF_SIZE - 1u);
        while cell_idx + 1u < cell_count {
            let cell_peak = sampling_cells[cell_idx].peak;
            if cum_peak + cell_peak >= target_peak {
                break;
            }
            cum_peak += cell_peak;
            cell_idx += 1u;
        }
        inverse_cdf[slot] = cell_idx;
    }
}
