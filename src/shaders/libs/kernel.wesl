/*
 * SPDX-FileCopyrightText: 2026 Golden_Water
 *
 * SPDX-License-Identifier: MPL-2.0
 */

import package::libs::constants::EXTENDED_REACH;
import package::libs::types::{Canvas, Splat, SplatCache, splat_rot, splat_rot_inv};

struct Sample {
    kernel: f32,
    affects: bool,
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

fn normal_pdf(x: f32) -> f32 {
    return 0.3989422804 * exp(-0.5 * x * x);
}

fn normal_cdf(x: f32) -> f32 {
    let z = abs(x) * 0.7071067812;
    let t = 1.0 / (1.0 + 0.3275911 * z);
    let poly = t
        * (0.254829592
            + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    let erf = 1.0 - poly * exp(-z * z);
    return 0.5 * (1.0 + select(-erf, erf, x >= 0.0));
}

fn tail_slope(s: f32, sigma: f32) -> f32 {
    if sigma <= 0.0 {
        return 0.0;
    }
    if sigma > 6.0 {
        return normal_pdf((s - 0.5) / sigma) / sigma;
    }

    let a = -s / sigma;
    let b = (1.0 - s) / sigma;
    let mass = normal_cdf(b) - normal_cdf(a);
    let pdf_a = normal_pdf(a);
    let pdf_b = normal_pdf(b);
    let body = 0.5 + 3.0 * s * (1.0 - s);
    let skew = 3.0 * (1.0 - 2.0 * s);

    return body * mass + skew * sigma * (pdf_a - pdf_b)
        - 3.0 * sigma * sigma * (mass - b * pdf_b + a * pdf_a);
}

fn extent_along(axis_weight: vec2f, half_extent: vec2f) -> f32 {
    return 1.0 / max(dot(axis_weight, 1.0 / max(half_extent, vec2f(1e-6))), 1e-6);
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

fn sample_full(splat: Splat, cache: SplatCache, px: vec2f, canvas: Canvas) -> Sample {
    let transform = splat_rot(cache);
    let uv = transpose(transform) * (px - splat.pos);

    let ramp_safe = max(cache.ramp, vec2f(1e-6));
    let excess = max(vec2f(0.0), abs(uv) - cache.core);
    let in_ramp = excess > vec2f(0.0);
    let kernel_s_vec = sign(uv) * excess / ramp_safe - 2.0 * TEXTURE_OFFSET;
    let ramp_s = length(kernel_s_vec);
    let in_rect = all(abs(uv) <= cache.core + cache.ramp);

    let axis_weight = kernel_s_vec * kernel_s_vec / max(ramp_s * ramp_s, 1e-6);
    let reach = EXTENDED_REACH * extent_along(axis_weight, cache.core);
    let sigma = reach / (3.0 * extent_along(axis_weight, ramp_safe));

    var dk_duv = vec2f(0.0);
    var dk_doutward = vec2f(0.0);
    let band = any(in_ramp) && ramp_s > 1e-6 && ramp_s < 1.0 + 3.0 * sigma;
    if band {
        let own_slope = d_profile(1.0 - ramp_s);
        let slope = select(tail_slope(ramp_s, sigma), own_slope, ramp_s < 1.0);
        let dk_ds_vec = -slope * kernel_s_vec / ramp_s;
        dk_duv = select(dk_ds_vec / ramp_safe, vec2f(0.0), !in_ramp);
        dk_doutward = sign(uv) * dk_duv;
    }

    let dk_dcore = -dk_doutward;
    let dk_dramp = dk_doutward * (1.0 - excess / ramp_safe);
    let raw_half = splat.size * 0.5;
    let ramp_len = min(raw_half, vec2f(canvas.ramp_cap));
    let size_limited = select(vec2f(0.0), vec2f(1.0), raw_half < vec2f(canvas.ramp_cap));

    var out: Sample;
    out.kernel = select(0.0, profile(1.0 - ramp_s), in_rect);
    out.affects = in_rect || band;
    out.d_pos = -(transform * dk_duv);
    out.d_rot = dot(dk_duv, vec2f(uv.y, -uv.x));
    out.d_size = 0.5 * splat.scale * (dk_dcore + dk_dramp * size_limited);
    out.d_scale = dk_dcore * raw_half + dk_dramp * ramp_len;
    return out;
}
