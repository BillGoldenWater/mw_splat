/*
 * SPDX-FileCopyrightText: 2026 Golden_Water
 *
 * SPDX-License-Identifier: MPL-2.0
 */

import package::libs::types::{BinHeader, Info, Splat, SplatCache};
import package::libs::constants::{TILE_SIZE, WG_SIZE_2D};
import package::libs::kernel::sample_kernel;

@group(0) @binding(0)
var<storage, read> info: Info;

@group(1) @binding(0)
var<storage, read> splats: array<Splat>;
@group(1) @binding(1)
var<storage, read> splat_caches: array<SplatCache>;

@group(2) @binding(0)
var<storage, read> bin_heads: array<BinHeader>;

@group(3) @binding(0)
var<storage, read> bin_data: array<u32>;

@group(4) @binding(0)
var<storage, read_write> pixels: array<vec4f>;
@group(4) @binding(1)
var<storage, read_write> transmittance: array<f32>;

@compute @workgroup_size(WG_SIZE_2D, WG_SIZE_2D)
fn render(@builtin(global_invocation_id) id: vec3u) {
    if any(id.xy >= info.canvas.dims) {
        return;
    }
    let idx = id.x + id.y * info.canvas.stride;
    let bin_idx = id.x / TILE_SIZE + id.y / TILE_SIZE * info.tile_dims.x;
    let bin = bin_heads[bin_idx];

    let px = vec2f(id.xy) + 0.5;

    var composite = info.canvas.background;
    var trans = 1.0;
    for(var i = bin.base; i < bin.base + bin.len; i += 1u) {
        let splat_index = bin_data[i];
        let splat = splats[splat_index];
        let sample = sample_kernel(splat, splat_caches[splat_index], px);
        let inv = 1.0 - sample * splat.color.a;
        composite *= inv;
        composite += splat.color * sample;
        trans *= inv;
    }
    pixels[idx] = composite;
    transmittance[idx] = trans;
}
