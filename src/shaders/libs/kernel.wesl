/*
 * SPDX-FileCopyrightText: 2026 Golden_Water
 *
 * SPDX-License-Identifier: MPL-2.0
 */

import package::libs::types::{Splat, SplatCache, splat_rot, splat_rot_inv};

struct Sample {
    kernel: f32,
    d_pos: vec2f,
    d_size: vec2f,
    d_scale: vec2f,
    d_rot: f32,
}

fn profile(t: f32) -> f32 {
    let u = clamp(t, 0.0, 1.0);
    return u * (0.5 + u * (1.5 - u));
}

fn d_profile(t: f32) -> f32 {
    let u = clamp(t, 0.0, 1.0);
    return 0.5 + 3.0 * u * (1.0 - u);
}

const TEXTURE_OFFSET: vec2f = vec2f(1.0 / 256.0, -1.0 / 256.0);

fn sample_kernel(splat: Splat, cache: SplatCache, px: vec2f) -> f32 {
    let uv = splat_rot_inv(cache) * (px - splat.pos);
    if !all(abs(uv) <= cache.core + cache.ramp) {
        return 0.0;
    }
    let excess = max(vec2f(0.0), abs(uv) - cache.core);
    let kernel_s_vec = sign(uv) * excess / max(cache.ramp, vec2f(1e-6)) - 2.0 * TEXTURE_OFFSET;
    return profile(1.0 - length(kernel_s_vec));
}

fn sample_full(splat: Splat, cache: SplatCache, px: vec2f, cap: f32) -> Sample {
    let transform = splat_rot(cache);
    let uv = transpose(transform) * (px - splat.pos);

    let ramp_safe = max(cache.ramp, vec2f(1e-6));
    let excess = max(vec2f(0.0), abs(uv) - cache.core);
    let in_ramp = excess > vec2f(0.0);
    let kernel_s_vec = sign(uv) * excess / ramp_safe - 2.0 * TEXTURE_OFFSET;
    let ramp_s = length(kernel_s_vec);
    let in_rect = all(abs(uv) <= cache.core + cache.ramp);

    var out: Sample;
    out.kernel = select(0.0, profile(1.0 - ramp_s), in_rect);
    out.d_pos = vec2f(0.0);
    out.d_size = vec2f(0.0);
    out.d_scale = vec2f(0.0);
    out.d_rot = 0.0;
    if in_rect && any(in_ramp) && ramp_s > 1e-6 && ramp_s < 1.0 {
        let slope = d_profile(1.0 - ramp_s);
        let dk_ds_vec = -slope * kernel_s_vec / ramp_s;
        let dk_duv = select(dk_ds_vec / ramp_safe, vec2f(0.0), !in_ramp);
        let dk_dhalf = -sign(uv) * dk_duv;
        let dk_dramp = sign(uv) * dk_duv * (1.0 - excess / ramp_safe);

        let raw_half = splat.size * 0.5;
        let ramp_len = min(raw_half, vec2f(cap));
        let size_limited = select(vec2f(0.0), vec2f(1.0), raw_half < vec2f(cap));

        out.d_pos = -(transform * dk_duv);
        out.d_rot = dot(dk_duv, vec2f(uv.y, -uv.x));
        out.d_size = 0.5 * splat.scale * (dk_dhalf + dk_dramp * size_limited);
        out.d_scale = dk_dhalf * raw_half + dk_dramp * ramp_len;
    }
    return out;
}
