/*
 * SPDX-FileCopyrightText: 2026 Golden_Water
 *
 * SPDX-License-Identifier: MPL-2.0
 */

import package::libs::types::Splat;
import package::libs::constants::FP_SCALE;

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

@if(atomic_f32)
alias GradElem = f32;
@else
alias GradElem = i32;

struct GradientBuf {
    x: GradElem,
    y: GradElem,
    w: GradElem,
    h: GradElem,
    rot: GradElem,
    r: GradElem,
    g: GradElem,
    b: GradElem,
    a: GradElem,
    scale_x: GradElem,
    scale_y: GradElem,
}

struct GradientAtomic {
    x: atomic<GradElem>,
    y: atomic<GradElem>,
    w: atomic<GradElem>,
    h: atomic<GradElem>,
    rot: atomic<GradElem>,
    r: atomic<GradElem>,
    g: atomic<GradElem>,
    b: atomic<GradElem>,
    a: atomic<GradElem>,
    scale_x: atomic<GradElem>,
    scale_y: atomic<GradElem>,
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
    s.lin = FP_SCALE * canvas_area / max(extent, vec2f(1e-6));
    s.quad = FP_SCALE * canvas_area / max(extent.x * extent.y, 1e-6);
    s.scale_norm = FP_SCALE * canvas_area / max(
            vec2f(
                splat.size.x * splat.size.y * splat.scale.y,
                splat.size.x * splat.size.y * splat.scale.x,
            ),
            vec2f(1e-6),
        );
    return s;
}
