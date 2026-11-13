// SPDX-FileCopyrightText: 2026 Golden_Water
//
// SPDX-License-Identifier: MPL-2.0

use glam::{UVec2, Vec2, Vec4};
use image::{ImageBuffer, Rgba};
use petgraph::unionfind::UnionFind;
use rand::{RngExt, SeedableRng, rngs::StdRng};

use crate::ir::{Ir, Splat};

const MS_BANDWIDTH: f32 = 0.1;
const MS_EPS: f32 = 0.01;
const MS_MAX_ITERS: usize = 64;
const MS_MAX_SAMPLES: usize = 4096;

const FEATHER: f32 = 0.5;
const CANDIDATES: usize = 24;
const NO_GEN_ALPHA: f32 = 0.01;

pub fn init(
    input: &ImageBuffer<Rgba<f32>, Vec<f32>>,
    n: usize,
    seed: u64,
) -> Ir {
    let dims = UVec2::from(input.dimensions());
    let canvasf = dims.as_vec2();
    let size = (canvasf.x * canvasf.y / n.max(1) as f32).sqrt().max(1.0);

    tracing::debug!("background");
    let background = background_color(input);

    tracing::debug!("blue noise");
    let no_gen = no_gen_mask(input);
    let points = blue_noise(dims, &no_gen, n, seed);
    tracing::debug!(count = points.len(), "sampled splats");

    let splats = points
        .iter()
        .map(|&p| {
            let color =
                feather_mean(input, p.x, p.y, size * 0.5, FEATHER);
            Splat {
                pos: Vec2::new(p.x, canvasf.y - p.y) / canvasf,
                size: Vec2::splat(size) / canvasf,
                rot: 0.0,
                color,
                scale: Vec2::ONE,
            }
        })
        .collect();

    Ir {
        aspect_ratio: canvasf.x / canvasf.y,
        background,
        splats,
    }
}

fn background_color(image: &ImageBuffer<Rgba<f32>, Vec<f32>>) -> Vec4 {
    if image.pixels().any(|p| p.0[3] < 1.0) {
        return Vec4::ZERO;
    }
    let mean = mean_shift_background(image);
    Vec4::new(mean[0], mean[1], mean[2], 1.0)
}

fn mean_shift_background(
    image: &ImageBuffer<Rgba<f32>, Vec<f32>>,
) -> [f32; 3] {
    let samples = sample_colors(image, MS_MAX_SAMPLES);
    let modes = mean_shift(&samples);
    largest_cluster_mean(&samples, &modes)
}

fn sample_colors(
    image: &ImageBuffer<Rgba<f32>, Vec<f32>>,
    max: usize,
) -> Vec<[f32; 3]> {
    let (w, h) = image.dimensions();
    let stride =
        (((w * h) as f32 / max as f32).sqrt().ceil() as u32).max(1);
    let mut out = Vec::new();
    for y in (0..h).step_by(stride as usize) {
        for x in (0..w).step_by(stride as usize) {
            let p = image.get_pixel(x, y).0;
            out.push([p[0], p[1], p[2]]);
        }
    }
    out
}

fn mean_shift(samples: &[[f32; 3]]) -> Vec<[f32; 3]> {
    let inv = 1.0 / MS_BANDWIDTH;
    let g = (inv.ceil() as usize).max(1) + 1;
    let mut grid: Vec<Vec<usize>> = vec![Vec::new(); g * g * g];
    for (i, &c) in samples.iter().enumerate() {
        let (a, b, c) = cell(c, inv, g);
        grid[(a * g + b) * g + c].push(i);
    }

    let h2 = MS_BANDWIDTH * MS_BANDWIDTH;
    samples
        .iter()
        .map(|&start| {
            let mut x = start;
            for _ in 0..MS_MAX_ITERS {
                let (mut sx, mut sy, mut sz) = (0.0f64, 0.0f64, 0.0f64);
                let mut count = 0usize;
                let (a, b, c) = cell(x, inv, g);
                for da in a.saturating_sub(1)..=(a + 1).min(g - 1) {
                    for db in b.saturating_sub(1)..=(b + 1).min(g - 1) {
                        for dc in c.saturating_sub(1)..=(c + 1).min(g - 1)
                        {
                            for &j in &grid[(da * g + db) * g + dc] {
                                let q = samples[j];
                                if dist2(x, q) <= h2 {
                                    sx += q[0] as f64;
                                    sy += q[1] as f64;
                                    sz += q[2] as f64;
                                    count += 1;
                                }
                            }
                        }
                    }
                }
                if count == 0 {
                    break;
                }
                let m = [
                    (sx / count as f64) as f32,
                    (sy / count as f64) as f32,
                    (sz / count as f64) as f32,
                ];
                if dist2(x, m) <= MS_EPS * MS_EPS {
                    return m;
                }
                x = m;
            }
            x
        })
        .collect()
}

fn largest_cluster_mean(
    samples: &[[f32; 3]],
    modes: &[[f32; 3]],
) -> [f32; 3] {
    let n = modes.len();
    let merge = MS_BANDWIDTH * 0.5;
    let inv = 1.0 / merge.max(1e-4);
    let g = (inv.ceil() as usize).max(1) + 1;
    let mut grid: Vec<Vec<usize>> = vec![Vec::new(); g * g * g];
    for (i, &m) in modes.iter().enumerate() {
        let (a, b, c) = cell(m, inv, g);
        grid[(a * g + b) * g + c].push(i);
    }

    let mut dsu = UnionFind::new(n);
    let r2 = merge * merge;
    for (i, &m) in modes.iter().enumerate() {
        let (a, b, c) = cell(m, inv, g);
        for da in a.saturating_sub(1)..=(a + 1).min(g - 1) {
            for db in b.saturating_sub(1)..=(b + 1).min(g - 1) {
                for dc in c.saturating_sub(1)..=(c + 1).min(g - 1) {
                    for &j in &grid[(da * g + db) * g + dc] {
                        if j > i && dist2(m, modes[j]) <= r2 {
                            dsu.union(i, j);
                        }
                    }
                }
            }
        }
    }

    let roots: Vec<usize> = (0..n).map(|i| dsu.find(i)).collect();
    let mut size = vec![0usize; n];
    for &r in &roots {
        size[r] += 1;
    }
    let root = (0..n).max_by_key(|&r| size[r]).unwrap_or(0);
    let mut acc = [0.0f64; 3];
    let mut count = 0usize;
    for i in 0..n {
        if roots[i] == root {
            for c in 0..3 {
                acc[c] += samples[i][c] as f64;
            }
            count += 1;
        }
    }
    let inv = 1.0 / count.max(1) as f64;
    [
        (acc[0] * inv) as f32,
        (acc[1] * inv) as f32,
        (acc[2] * inv) as f32,
    ]
}

fn cell(c: [f32; 3], inv: f32, g: usize) -> (usize, usize, usize) {
    let a =
        ((c[0] * inv).floor() as isize).clamp(0, g as isize - 1) as usize;
    let b =
        ((c[1] * inv).floor() as isize).clamp(0, g as isize - 1) as usize;
    let c =
        ((c[2] * inv).floor() as isize).clamp(0, g as isize - 1) as usize;
    (a, b, c)
}

fn dist2(a: [f32; 3], b: [f32; 3]) -> f32 {
    let d0 = a[0] - b[0];
    let d1 = a[1] - b[1];
    let d2 = a[2] - b[2];
    d0 * d0 + d1 * d1 + d2 * d2
}

fn no_gen_mask(image: &ImageBuffer<Rgba<f32>, Vec<f32>>) -> Vec<bool> {
    let mut mask: Vec<bool> =
        image.pixels().map(|p| p.0[3] >= NO_GEN_ALPHA).collect();
    if mask.iter().all(|&a| !a) {
        mask.fill(true);
    }
    mask
}

fn blue_noise(
    dims: UVec2,
    no_gen: &[bool],
    n: usize,
    seed: u64,
) -> Vec<Vec2> {
    let w = dims.x as usize;
    let (wf, hf) = (dims.x as f32, dims.y as f32);
    let allowed: Vec<u32> = (0..no_gen.len() as u32)
        .filter(|&i| no_gen[i as usize])
        .collect();
    if allowed.is_empty() {
        return Vec::new();
    }
    let cell = (wf * hf / n.max(1) as f32).sqrt().max(1.0) * 0.5;
    let gw = (wf / cell).ceil() as i32 + 1;
    let gh = (hf / cell).ceil() as i32 + 1;
    let mut grid: Vec<Vec<Vec2>> = vec![Vec::new(); (gw * gh) as usize];
    let mut rng = StdRng::seed_from_u64(seed);
    let mut points = Vec::with_capacity(n);
    for _ in 0..n {
        let mut best = Vec2::ZERO;
        let mut best_dist = -1.0f32;
        for _ in 0..CANDIDATES {
            let idx =
                allowed[rng.random_range(0..allowed.len())] as usize;
            let p = Vec2::new(
                (idx % w) as f32 + rng.random::<f32>(),
                (idx / w) as f32 + rng.random::<f32>(),
            );
            let cx = (p.x / cell).floor() as i32;
            let cy = (p.y / cell).floor() as i32;
            let mut dist = f32::INFINITY;
            for dy in -2..=2 {
                for dx in -2..=2 {
                    let (gx, gy) = (cx + dx, cy + dy);
                    if gx < 0 || gy < 0 || gx >= gw || gy >= gh {
                        continue;
                    }
                    for q in &grid[(gx + gy * gw) as usize] {
                        dist = dist.min((*q - p).length_squared());
                    }
                }
            }
            if dist > best_dist {
                best_dist = dist;
                best = p;
            }
        }
        let idx = (best.x / cell).floor() as i32
            + (best.y / cell).floor() as i32 * gw;
        grid[idx as usize].push(best);
        points.push(best);
    }
    points
}

fn feather_mean(
    image: &ImageBuffer<Rgba<f32>, Vec<f32>>,
    cx: f32,
    cy: f32,
    hs: f32,
    feather: f32,
) -> Vec4 {
    let ramp = (2.0 * feather * hs).max(1e-9);
    let rad = hs.ceil() as isize;
    let (w, h) = (image.width() as isize, image.height() as isize);
    let (cx_i, cy_i) = (cx.round() as isize, cy.round() as isize);
    let x0 = (cx_i - rad).max(0) as usize;
    let x1 = (cx_i + rad + 1).min(w) as usize;
    let y0 = (cy_i - rad).max(0) as usize;
    let y1 = (cy_i + rad + 1).min(h) as usize;
    let mut acc = [0.0f64; 4];
    let mut wsum = 0.0f64;
    for py in y0..y1 {
        for px in x0..x1 {
            let dx = px as f32 + 0.5 - cx;
            let dy = py as f32 + 0.5 - cy;
            let tent = smoothstep((hs - dx.abs()) / ramp)
                * smoothstep((hs - dy.abs()) / ramp);
            if tent <= 0.0 {
                continue;
            }
            let p = image.get_pixel(px as u32, py as u32).0;
            for k in 0..4 {
                acc[k] += tent as f64 * p[k] as f64;
            }
            wsum += tent as f64;
        }
    }
    if wsum <= 1e-12 {
        return Vec4::ZERO;
    }
    let inv = 1.0 / wsum;
    Vec4::new(
        (acc[0] * inv) as f32,
        (acc[1] * inv) as f32,
        (acc[2] * inv) as f32,
        (acc[3] * inv) as f32,
    )
}

fn smoothstep(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}
