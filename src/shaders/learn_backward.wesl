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
import package::libs::learn::{Gradient, GradientAtomic, grad_scales};

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
var<storage, read> log_transmittance: array<f32>;
@group(4) @binding(2)
var<storage, read> loss_weights: array<f32>;

@group(5) @binding(0)
var<storage, read_write> grads: array<GradientAtomic>;
@group(5) @binding(1)
var<storage, read_write> tile_loss: array<f32>;

struct Px {
    below: vec4f,
    pixel_grad: vec4f,
    center: vec2f,
    log_below: f32,
    log_total: f32,
    valid: bool,
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
        let b = &block_pixels[block_idx];

        b.valid = all(px_pos < info.canvas.dims);
        if !b.valid {
            continue;
        }

        b.center = vec2f(px_pos) + 0.5;
        b.below = info.canvas.background;
        b.log_below = 0.;

        let px_idx = px_pos.x + px_pos.y * info.canvas.stride;
        let loss_weight = loss_weights[px_idx];
        let actual = pixels[px_idx];
        let expected = reference[px_idx];
        let black_err = actual.rgb - expected.rgb;
        let white_err = black_err + vec3f(expected.a - actual.a);
        block_loss += (dot(black_err, black_err) + dot(white_err, white_err)) * loss_norm
            * loss_weight;

        b.pixel_grad = vec4f(
                (black_err + white_err) * (2.0 * loss_norm * loss_weight),
                -2.0 * loss_norm * loss_weight * (white_err.x + white_err.y + white_err.z),
            );
        b.log_total = log_transmittance[px_idx];
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
        tile_loss[tile_idx] = workgroup_loss_buf[0];
    }

    let head = bin_heads[tile_idx];
    // NOTE: is uniform inside workgroup
    if head.len == 0 {
        return;
    }

    let grad_base = info.splats_length * (tile_idx % GRAD_COPIES);
    // NOTE: is uniform inside workgroup
    for(var bin_idx = head.base; bin_idx < head.base + head.len; bin_idx += 1u) {
        let splat_idx = bin_data[bin_idx];
        let splat = splats[splat_idx];
        let splat_cache = splat_caches[splat_idx];

        var block_grads: Gradient;
        var has_grad = false;

        for(var block_idx = 0u; block_idx < BWD_BLOCK_LEN; block_idx += 1) {
            if !block_pixels[block_idx].valid {
                continue;
            }

            let b = &block_pixels[block_idx];

            let sample = sample_full(splat, splat_cache, b.center, info.canvas);
            let inv = 1. - (sample.kernel * splat.color.a);
            let log_through = b.log_below + log(inv);
            if sample.affects {
                let splat_grad = b.pixel_grad * exp(b.log_total - log_through);
                let d_loss_d_inv = dot(splat_grad, b.below);
                let d_color = splat_grad - vec4f(0., 0., 0., d_loss_d_inv);
                let d_kernel = dot(d_color, splat.color);
                block_grads.x += sample.d_pos.x * d_kernel;
                block_grads.y += sample.d_pos.y * d_kernel;
                block_grads.w += sample.d_size.x * d_kernel;
                block_grads.h += sample.d_size.y * d_kernel;
                block_grads.rot += sample.d_rot * d_kernel;
                block_grads.r += d_color.r * sample.kernel;
                block_grads.g += d_color.g * sample.kernel;
                block_grads.b += d_color.b * sample.kernel;
                block_grads.a += d_color.a * sample.kernel;
                block_grads.scale_x += sample.d_scale.x * d_kernel;
                block_grads.scale_y += sample.d_scale.y * d_kernel;
                has_grad = true;
            }
            b.below = b.below * inv + splat.color * sample.kernel;
            b.log_below = log_through;
        }

        if has_grad {
            let grad = &grads[grad_base + splat_idx];
            let scales = grad_scales(splat, info.canvas.dims);
            atomicAdd(&grad.x, i32(round(block_grads.x * scales.lin.x)));
            atomicAdd(&grad.y, i32(round(block_grads.y * scales.lin.y)));
            atomicAdd(&grad.w, i32(round(block_grads.w * scales.lin.x)));
            atomicAdd(&grad.h, i32(round(block_grads.h * scales.lin.y)));
            atomicAdd(&grad.rot, i32(round(block_grads.rot * scales.quad)));
            atomicAdd(&grad.r, i32(round(block_grads.r * scales.quad)));
            atomicAdd(&grad.g, i32(round(block_grads.g * scales.quad)));
            atomicAdd(&grad.b, i32(round(block_grads.b * scales.quad)));
            atomicAdd(&grad.a, i32(round(block_grads.a * scales.quad)));
            atomicAdd(&grad.scale_x, i32(round(block_grads.scale_x * scales.scale_norm.x)));
            atomicAdd(&grad.scale_y, i32(round(block_grads.scale_y * scales.scale_norm.y)));
        }
    }
}
