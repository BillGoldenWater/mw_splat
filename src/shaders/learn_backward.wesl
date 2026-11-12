/*
 * SPDX-FileCopyrightText: 2026 Golden_Water
 *
 * SPDX-License-Identifier: MPL-2.0
 */

import package::libs::types::{BinHeader, Info, Splat, SplatCache};
import package::libs::constants::{
    BWD_BLOCK_LEN,
    BWD_BLOCK_SIZE,
    BWD_WG_LEN,
    BWD_WG_SIZE,
    GRAD_COPIES,
    TILE_SIZE
};
import package::libs::kernel::sample_full;
import package::libs::learn::{Gradient, GradientAtomic};

@group(0) @binding(0)
var<storage, read> info: Info;
@group(0) @binding(1)
var<storage, read> reference: array<vec4f>;

// engine space
@group(1) @binding(0)
var<storage, read> splats: array<Splat>;
@group(1) @binding(1)
var<storage, read> splat_caches: array<SplatCache>;

@group(2) @binding(0)
var<storage, read> bin_heads: array<BinHeader>;

// will overwritten
@group(3) @binding(0)
var<storage, read> bin_data: array<u32>;

@group(4) @binding(0)
var<storage, read> pixels: array<vec4f>;
@group(4) @binding(1)
var<storage, read> transmittance: array<f32>;
@group(4) @binding(2)
var<storage, read> loss_weights: array<f32>;

@group(5) @binding(0)
var<storage, read_write> grads: array<GradientAtomic>;
@group(5) @binding(1)
var<storage, read_write> loss: atomic<f32>;

struct Px {
    composite: vec4f,
    adjoint: vec4f,
    center: vec2f,
    t_before: f32,
    transmittance: f32,
    on: bool,
}

var<workgroup> workgroup_loss_buf: array<f32, BWD_WG_LEN>;

// NOTE: one tile per workgroup,
// so inside a workgroup is uniform control flow for tile
@compute @workgroup_size(BWD_WG_SIZE, BWD_WG_SIZE)
fn backward(
    @builtin(workgroup_id) w_id: vec3u,
    @builtin(num_workgroups) w_num: vec3u,
    @builtin(local_invocation_id) l_id: vec3u,
    @builtin(local_invocation_index) l_ii: u32,
) {
    let tile_idx = w_id.x + w_id.y * w_num.x;
    // NOTE: is uniform inside workgroup
    if tile_idx >= arrayLength(&bin_heads) {
        return;
    }
    let tile_pos = w_id.xy;
    let block_pos = tile_pos * TILE_SIZE + l_id.xy * BWD_BLOCK_SIZE;

    let loss_norm = 1.0 / (f32(info.canvas.dims.x * info.canvas.dims.y) * 6.0);
    var block_pixels: array<Px, BWD_BLOCK_LEN>;
    var block_loss = 0.;
    for(var block_idx = 0u; block_idx < BWD_BLOCK_LEN; block_idx += 1) {
        let px_pos = block_pos + vec2u(block_idx % BWD_BLOCK_SIZE, block_idx / BWD_BLOCK_SIZE);

        let in_canvas = all(px_pos < info.canvas.dims);
        block_pixels[block_idx].on = in_canvas;
        if !in_canvas {
            continue;
        }

        block_pixels[block_idx].center = vec2f(px_pos) + 0.5;
        block_pixels[block_idx].composite = info.canvas.background;
        block_pixels[block_idx].t_before = 1.;

        let px_idx = px_pos.x + px_pos.y * info.canvas.stride;
        let weight = loss_weights[px_idx];
        let px = pixels[px_idx];
        let tgt = reference[px_idx];
        let black = px.rgb - tgt.rgb;
        let white = black + vec3f(tgt.a - px.a);
        block_loss += (dot(black, black) + dot(white, white)) * loss_norm * weight;

        block_pixels[block_idx].adjoint = vec4f(
                (black + white) * (2.0 * loss_norm * weight),
                -2.0 * loss_norm * weight * (white.x + white.y + white.z),
            );
        block_pixels[block_idx].transmittance = transmittance[px_idx];
    }

    workgroup_loss_buf[l_ii] = block_loss;
    workgroupBarrier();
    for(var step = BWD_WG_LEN / 2u; step > 0u; step >>= 1u) {
        if l_ii < step {
            workgroup_loss_buf[l_ii] += workgroup_loss_buf[l_ii + step];
        }
        workgroupBarrier();
    }
    if l_ii == 0u {
        atomicAdd(&loss, workgroup_loss_buf[0]);
    }

    let head = bin_heads[tile_idx];
    // NOTE: is uniform inside workgroup
    if head.len == 0 {
        return;
    }

    let grad_base = arrayLength(&splats) * (tile_idx % GRAD_COPIES);
    let bg_transparency = 1.0 - info.canvas.background.a;
    // NOTE: is uniform inside workgroup
    for(var bin_idx = head.base; bin_idx < head.base + head.len; bin_idx += 1u) {
        let splat_idx = bin_data[bin_idx];
        let splat = splats[splat_idx];
        let splat_cache = splat_caches[splat_idx];

        var grads_local: Gradient;
        var touched = false;

        for(var block_idx = 0u; block_idx < BWD_BLOCK_LEN; block_idx += 1) {
            if !block_pixels[block_idx].on {
                continue;
            }

            let b = &block_pixels[block_idx];

            let smp = sample_full(splat, splat_cache, b.center, info.canvas.ramp_cap);
            let inv = 1. - (smp.kernel * splat.color.a);
            let denom = b.t_before * inv;
            if smp.kernel > 0.0 {
                var t_after = 0.0;
                if denom > 1e-30 {
                    t_after = b.transmittance / denom;
                }
                let adjoint4 = b.adjoint * t_after;
                let d_cov = adjoint4.a * bg_transparency * b.t_before
                    - dot(b.composite.rgb, adjoint4.rgb);
                let dk = vec4f(adjoint4.rgb, d_cov);
                let d_kernel = dot(dk, splat.color);
                grads_local.x += smp.d_pos.x * d_kernel;
                grads_local.y += smp.d_pos.y * d_kernel;
                grads_local.w += smp.d_size.x * d_kernel;
                grads_local.h += smp.d_size.y * d_kernel;
                grads_local.rot += smp.d_rot * d_kernel;
                grads_local.r += dk.x * smp.kernel;
                grads_local.g += dk.y * smp.kernel;
                grads_local.b += dk.z * smp.kernel;
                grads_local.a += dk.w * smp.kernel;
                grads_local.scale_x += smp.d_scale.x * d_kernel;
                grads_local.scale_y += smp.d_scale.y * d_kernel;
                touched = true;
            }
            b.composite *= inv;
            b.composite += splat.color * smp.kernel;
            b.t_before = denom;
        }

        if touched {
            let grad = &grads[grad_base + splat_idx];
            atomicAdd(&grad.x, grads_local.x);
            atomicAdd(&grad.y, grads_local.y);
            atomicAdd(&grad.w, grads_local.w);
            atomicAdd(&grad.h, grads_local.h);
            atomicAdd(&grad.rot, grads_local.rot);
            atomicAdd(&grad.r, grads_local.r);
            atomicAdd(&grad.g, grads_local.g);
            atomicAdd(&grad.b, grads_local.b);
            atomicAdd(&grad.a, grads_local.a);
            atomicAdd(&grad.scale_x, grads_local.scale_x);
            atomicAdd(&grad.scale_y, grads_local.scale_y);
        }
    }
}
