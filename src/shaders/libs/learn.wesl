/*
 * SPDX-FileCopyrightText: 2026 Golden_Water
 *
 * SPDX-License-Identifier: MPL-2.0
 */

import package::libs::types::Splat;
import package::libs::constants::GRAD_SCALE;

struct Gradient {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    rot: f32,
    r: f32,
    g: f32,
    b: f32,
    a: f32,
    scale_x: f32,
    scale_y: f32,
}

struct GradientBuf {
    x: i32,
    y: i32,
    w: i32,
    h: i32,
    rot: i32,
    r: i32,
    g: i32,
    b: i32,
    a: i32,
    scale_x: i32,
    scale_y: i32,
}

struct GradientAtomic {
    x: atomic<i32>,
    y: atomic<i32>,
    w: atomic<i32>,
    h: atomic<i32>,
    rot: atomic<i32>,
    r: atomic<i32>,
    g: atomic<i32>,
    b: atomic<i32>,
    a: atomic<i32>,
    scale_x: atomic<i32>,
    scale_y: atomic<i32>,
}

struct GradScales {
    lin: vec2f,
    quad: f32,
    scale_norm: vec2f,
}

fn grad_scales(splat: Splat, canvas_dims: vec2u) -> GradScales {
    let canvas_area = f32(canvas_dims.x * canvas_dims.y);
    let extent = splat.size * splat.scale;
    var s: GradScales;
    s.lin = GRAD_SCALE * canvas_area / max(extent, vec2f(1e-6));
    s.quad = GRAD_SCALE * canvas_area / max(extent.x * extent.y, 1e-6);
    s.scale_norm = GRAD_SCALE * canvas_area / max(
            vec2f(
                splat.size.x * splat.size.y * splat.scale.y,
                splat.size.x * splat.size.y * splat.scale.x,
            ),
            vec2f(1e-6),
        );
    return s;
}
