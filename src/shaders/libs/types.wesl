/*
 * SPDX-FileCopyrightText: 2026 Golden_Water
 *
 * SPDX-License-Identifier: MPL-2.0
 */

struct Info {
    canvas: Canvas,
    tile_dims: vec2u,
    lr: Lr,
}

struct Canvas {
    dims: vec2u,
    dimsf: vec2f,
    stride: u32,
    ramp_cap: f32,
    background: vec4f,
}

struct Lr {
    position: f32,
    size: f32,
    rotation: f32,
    color: f32,
    alpha: f32,
    scale: f32,
}

struct Splat {
    pos: vec2f,
    size: vec2f,
    color: vec4f,
    scale: vec2f,
    rot: f32,
}

struct SplatCache {
    sin: f32,
    cos: f32,
    core: vec2f,
    ramp: vec2f,
}

fn splat_rot(cache: SplatCache) -> mat2x2f {
    return mat2x2f(
            cache.cos, cache.sin,
            -cache.sin, cache.cos,
        );
}

fn splat_rot_inv(cache: SplatCache) -> mat2x2f {
    return mat2x2f(
            cache.cos, -cache.sin,
            cache.sin, cache.cos,
        );
}

struct BinHeader {
    base: u32,
    len: u32,
}

struct BinOverflow {
    // bool like
    // CPU: check >0 asyncly and resize
    overflow: atomic<u32>,
    // the iter_num when the overflow happens
    iter_num: u32,
    // set to 0 when overflow in shader
    // reset to actual count when reallocation
    indirect_args: array<vec3u>,
}
