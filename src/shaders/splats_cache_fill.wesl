/*
 * SPDX-FileCopyrightText: 2026 Golden_Water
 *
 * SPDX-License-Identifier: MPL-2.0
 */

import package::libs::types::{Info, Splat, SplatCache};
import package::libs::constants::WG_SIZE;

@group(0) @binding(0)
var<storage, read> info: Info;

@group(1) @binding(0)
var<storage, read> splats: array<Splat>;
@group(1) @binding(1)
var<storage, read_write> splat_caches: array<SplatCache>;

@compute @workgroup_size(WG_SIZE)
fn splats_cache_fill(
    @builtin(global_invocation_id) id: vec3u,
    @builtin(num_workgroups) wg_num: vec3u,
) {
    let idx = id.x + id.y * WG_SIZE * wg_num.x;
    if idx >= arrayLength(&splats) {
        return;
    }

    let splat = splats[idx];
    let raw_half = splat.size * 0.5;
    let ramp_len = min(raw_half, vec2f(info.canvas.ramp_cap));
    let ramp = ramp_len * splat.scale;
    splat_caches[idx] = SplatCache(
            sin(splat.rot),
            cos(splat.rot),
            raw_half * splat.scale - ramp,
            ramp,
        );
}
