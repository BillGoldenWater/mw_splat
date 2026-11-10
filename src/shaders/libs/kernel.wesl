import package::libs::types::{Splat, SplatCache, splat_rot, splat_rot_inv};

struct Sample {
    kernel: f32,
    d_pos: vec2f,
    d_size: vec2f,
    d_fe: vec2f,
    d_rot: f32,
}

fn d_smoothstep(t: vec2f) -> vec2f {
    let c = clamp(t, vec2f(0.0), vec2f(1.0));
    return 6. * c * (vec2f(1.) - c);
}

fn sample_kernel(splat: Splat, cache: SplatCache, px: vec2f) -> f32 {
    let uv = splat_rot_inv(cache) * (px - splat.pos);
    let tuv = (splat.size * 0.5 - abs(uv)) / cache.ramp;
    let value = smoothstep(vec2f(0), vec2f(1), tuv);
    return value.x * value.y;
}

fn sample_full(splat: Splat, cache: SplatCache, px: vec2f) -> Sample {
    let ramp = cache.ramp;
    let size = splat.size;
    let transform = splat_rot(cache);

    let uv = transpose(transform) * (px - splat.pos);
    let tuv = (size * 0.5 - abs(uv)) / ramp;

    let value = smoothstep(vec2f(0), vec2f(1), tuv);

    let tuv_c = clamp(tuv, vec2f(0.0), vec2f(1.0));
    let d_value = 6. * tuv_c * (vec2f(1.) - tuv_c);

    let crossed = d_value * value.yx;
    let d_kernel = crossed * (-sign(uv) / ramp);

    var out: Sample;
    out.kernel = value.x * value.y;
    out.d_pos = -(transform * d_kernel);
    out.d_rot = d_kernel.x * uv.y - d_kernel.y * uv.x;
    out.d_size = crossed * (vec2f(0.5) / ramp - tuv * splat.fe / ramp);
    out.d_fe = crossed * (-tuv * size / ramp);
    return out;
}
