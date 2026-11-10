/*
 * SPDX-FileCopyrightText: 2026 Golden_Water
 *
 * SPDX-License-Identifier: MPL-2.0
 */

// Shared by the redistribution epoch passes.
//
// A sampling cell is a square pixel block; `redistribute_cells` writes one
// `SamplingCell` per cell (its summed loss and hottest pixel), and a mover lands
// on the hottest pixel of its cell.

fn redistribute_hash(a: u32, b: u32) -> f32 {
    var h = (a * 747796405u) ^ ((b + 1u) * 2891336453u);
    h = (h ^ (h >> 16u)) * 2246822519u;
    h = (h ^ (h >> 13u)) * 3266489917u;
    return f32((h ^ (h >> 16u)) & 0xffffffu) / 16777216.0;
}

struct SamplingCell {
    loss: f32,
    hot_px_idx: u32,
}

// Boundary-bin loss bounds and donation fraction: a cell below `boundary_lo`
// donates, and one below `boundary_hi` donates with probability `boundary_frac`.
struct RedistributeState {
    boundary_lo: f32,
    boundary_hi: f32,
    boundary_frac: f32,
}
