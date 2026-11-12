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

fn sample_kernel(splat: Splat, cache: SplatCache, px: vec2f) -> f32 {
    let uv = splat_rot_inv(cache) * (px - splat.pos);
    let excess = max(vec2f(0.0), abs(uv) - cache.core);
    return profile(1.0 - length(excess / max(cache.ramp, vec2f(1e-6))));
}

fn sample_full(splat: Splat, cache: SplatCache, px: vec2f, cap: f32) -> Sample {
    let transform = splat_rot(cache);
    let uv = transpose(transform) * (px - splat.pos);

    let excess = max(vec2f(0.0), abs(uv) - cache.core);
    let ramp_safe = max(cache.ramp, vec2f(1e-6));
    let s = length(excess / ramp_safe);

    var out: Sample;
    out.kernel = profile(1.0 - s);
    out.d_pos = vec2f(0.0);
    out.d_size = vec2f(0.0);
    out.d_scale = vec2f(0.0);
    out.d_rot = 0.0;
    if s > 1e-6 && s < 1.0 {
        let raw_half = splat.size * 0.5;
        let ramp_len = min(raw_half, vec2f(cap));
        let size_limited = select(vec2f(0.0), vec2f(1.0), raw_half < vec2f(cap));

        let slope = d_profile(1.0 - s);
        let dk_dq = -slope * excess / (ramp_safe * ramp_safe * s);
        let dk_duv = dk_dq * sign(uv);
        let dk_dcore = -dk_dq;
        let dk_dramp = slope * excess * excess / (s * ramp_safe * ramp_safe * ramp_safe) - dk_dcore;

        out.d_pos = -(transform * dk_duv);
        out.d_rot = dk_duv.x * uv.y - dk_duv.y * uv.x;
        out.d_size = 0.5 * splat.scale * (dk_dcore + dk_dramp * size_limited);
        out.d_scale = dk_dcore * raw_half + dk_dramp * ramp_len;
    }
    return out;
}
