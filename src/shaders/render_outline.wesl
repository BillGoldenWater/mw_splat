/*
 * SPDX-FileCopyrightText: 2026 Golden_Water
 *
 * SPDX-License-Identifier: MPL-2.0
 */

import package::libs::types::{Info, Splat};

@group(0) @binding(0)
var<storage, read> info: Info;
@group(1) @binding(0)
var<storage, read> splats: array<Splat>;

struct Out {
    @builtin(position) pos: vec4f,
    @location(0) local: vec2f,
    @location(1) half: vec2f,
    @location(2) color: vec4f,
}

const vertices = array<vec2f, 6>(
        vec2f(-1.0, -1.0),
        vec2f(1.0, -1.0),
        vec2f(1.0, 1.0),
        vec2f(-1.0, -1.0),
        vec2f(1.0, 1.0),
        vec2f(-1.0, 1.0),
    );

@vertex
fn vs_main(@builtin(vertex_index) v_i: u32, @builtin(instance_index) i_i: u32) -> Out {
    let splat = splats[i_i];
    let half = splat.size * 0.5;

    let local = vertices[v_i] * half;
    let rot_cos = cos(splat.rot);
    let rot_sin = sin(splat.rot);
    let rot = mat2x2f(
            rot_cos, rot_sin,
            -rot_sin, rot_cos,
        );
    let world = splat.pos + rot * local;

    let pos = (world / info.canvas.dimsf * 2. - 1.) * vec2f(1., -1.);

    let depth = f32(i_i) / f32(arrayLength(&splats));
    let hsv = vec3f(depth * 0.8, 1., 1.);
    let color = vec4f(hsv2rgb(hsv), 1.);

    var out: Out;
    out.pos = vec4f(pos, 0.0, 1.0);
    out.local = local;
    out.half = half;
    out.color = color;
    return out;
}

const thickness = 1.;

@fragment
fn fs_main(in: Out) -> @location(0) vec4f {
    let border = max(in.half - thickness, vec2f(thickness));
    if all(abs(in.local) < border) {
        discard;
    }
    return vec4f(in.color.rgb, 1.0);
}

fn hsv2rgb(hsv: vec3f) -> vec3f {
    let hue = fract(vec3f(hsv.x) + vec3f(0., 2. / 3., 1. / 3.));
    let saturated = clamp(abs(hue * 6. - 3.) - 1., vec3f(0.), vec3f(1.));
    return hsv.z * mix(vec3f(1.), saturated, vec3f(hsv.y));
}
