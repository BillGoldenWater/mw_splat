import package::libs::types::{Info, Splat};
import package::libs::constants::{GRAD_COPIES, LOGIT_EPS, WG_SIZE};
import package::libs::learn::Gradient;

@group(0) @binding(0)
var<storage, read> info: Info;

// engine space
@group(1) @binding(0)
var<storage, read_write> splats: array<Splat>;

@group(2) @binding(0)
var<storage, read> grads: array<Gradient>;

@group(3) @binding(0)
var<storage, read_write> moment1: array<Splat>;
@group(3) @binding(1)
var<storage, read_write> moment2: array<Splat>;

struct Imm {
    lr_scale: f32,
}
var<immediate> imm: Imm;

const BETA1: f32 = 0.9;
const BETA2: f32 = 0.999;
const ADAM_EPS: f32 = 1e-8;
const LN_SIZE_MAX: f32 = log(20000.);

@compute @workgroup_size(WG_SIZE)
fn adam(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) wg_num: vec3u) {
    let idx = id.x + id.y * WG_SIZE * wg_num.x;
    if idx >= arrayLength(&splats) {
        return;
    }

    let g = sum_grad(idx, arrayLength(&splats));
    var grad: Splat;
    grad.pos = vec2f(g.x, g.y);
    grad.size = vec2f(g.w, g.h);
    grad.rot = g.rot;
    grad.color = vec4f(g.r, g.g, g.b, g.a);
    grad.fe = vec2f(g.fe_x, g.fe_y);

    let splat = splats[idx];
    // straight rgb
    var rgb: vec3f;
    if splat.color.a > LOGIT_EPS {
        rgb = splat.color.rgb / splat.color.a;
    } else {
        rgb = vec3f(0.5);
    }

    var raw_grad: Splat;
    raw_grad.pos = grad.pos;
    raw_grad.size = grad.size * splat.size;
    raw_grad.rot = grad.rot;
    raw_grad.color = vec4f(
            grad.color.rgb * splat.color.a * rgb * (1. - rgb),
            (grad.color.a + dot(grad.color.rgb, rgb)) * splat.color.a * (1. - splat.color.a),
        );
    raw_grad.fe = grad.fe * splat.fe * (1. - splat.fe);

    var m = moment1[idx];
    var v = moment2[idx];

    m.pos = m.pos * BETA1 + raw_grad.pos * (1.0 - BETA1);
    v.pos = v.pos * BETA2 + raw_grad.pos * raw_grad.pos * (1.0 - BETA2);
    m.size = m.size * BETA1 + raw_grad.size * (1.0 - BETA1);
    v.size = v.size * BETA2 + raw_grad.size * raw_grad.size * (1.0 - BETA2);
    m.rot = m.rot * BETA1 + raw_grad.rot * (1.0 - BETA1);
    v.rot = v.rot * BETA2 + raw_grad.rot * raw_grad.rot * (1.0 - BETA2);
    m.color = m.color * BETA1 + raw_grad.color * (1.0 - BETA1);
    v.color = v.color * BETA2 + raw_grad.color * raw_grad.color * (1.0 - BETA2);
    m.fe = m.fe * BETA1 + raw_grad.fe * (1.0 - BETA1);
    v.fe = v.fe * BETA2 + raw_grad.fe * raw_grad.fe * (1.0 - BETA2);

    let rate = imm.lr_scale;
    let lr = info.lr;
    let lr_col = vec4f(vec3f(lr.color), lr.alpha);

    var raw = raw_from_splat(splat);
    raw.pos -= rate * (lr.position * m.pos) / (sqrt(v.pos) + vec2f(ADAM_EPS));
    raw.size -= rate * (lr.scale * m.size) / (sqrt(v.size) + vec2f(ADAM_EPS));
    raw.rot -= rate * lr.rotation * m.rot / (sqrt(v.rot) + ADAM_EPS);
    raw.color -= rate * (lr_col * m.color) / (sqrt(v.color) + vec4f(ADAM_EPS));
    raw.fe -= rate * (lr.feather * m.fe) / (sqrt(v.fe) + vec2f(ADAM_EPS));

    raw.size = min(raw.size, vec2f(LN_SIZE_MAX));
    raw.color = clamp(raw.color, vec4f(-10.0), vec4f(10.0));
    raw.fe = clamp(raw.fe, vec2f(-10.0), vec2f(10.0));

    splats[idx] = splat_from_raw(raw);
    moment1[idx] = m;
    moment2[idx] = v;
}

fn sigmoid_vec2(x: vec2<f32>) -> vec2<f32> {
    return 1.0 / (1.0 + exp(-x));
}

fn logit_vec2(x: vec2<f32>) -> vec2<f32> {
    let value = clamp(x, vec2(LOGIT_EPS), vec2(1. - LOGIT_EPS));
    return log(value / (1. - value));
}

fn sigmoid_vec4(x: vec4<f32>) -> vec4<f32> {
    return 1.0 / (1.0 + exp(-x));
}

fn logit_vec4(x: vec4<f32>) -> vec4<f32> {
    let value = clamp(x, vec4(LOGIT_EPS), vec4(1. - LOGIT_EPS));
    return log(value / (1. - value));
}

fn raw_from_splat(splat: Splat) -> Splat {
    let color = splat.color;
    var rgb: vec3f;
    if color.a > LOGIT_EPS {
        rgb = color.rgb / color.a;
    } else {
        rgb = vec3(0.5);
    }

    return Splat(
            splat.pos,
            log(max(splat.size, vec2(LOGIT_EPS))),
            logit_vec4(vec4(rgb, color.a)),
            logit_vec2(splat.fe),
            splat.rot,
        );
}

fn splat_from_raw(raw: Splat) -> Splat {
    let size = exp(raw.size);
    let fe = sigmoid_vec2(raw.fe);

    var color = sigmoid_vec4(raw.color);
    color.r = color.r * color.a;
    color.g = color.g * color.a;
    color.b = color.b * color.a;

    return Splat(raw.pos, size, color, fe, raw.rot);
}

fn sum_grad(idx: u32, splat_count: u32) -> Gradient {
    var total: Gradient;
    for(var copy_idx = 0u; copy_idx < GRAD_COPIES; copy_idx += 1u) {
        let g = grads[splat_count * copy_idx + idx];
        total.x += g.x;
        total.y += g.y;
        total.w += g.w;
        total.h += g.h;
        total.rot += g.rot;
        total.r += g.r;
        total.g += g.g;
        total.b += g.b;
        total.a += g.a;
        total.fe_x += g.fe_x;
        total.fe_y += g.fe_y;
    }
    return total;
}
