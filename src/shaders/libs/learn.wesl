/*
 * SPDX-FileCopyrightText: 2026 Golden_Water
 *
 * SPDX-License-Identifier: MPL-2.0
 */

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

struct GradientAtomic {
    x: atomic<f32>,
    y: atomic<f32>,
    w: atomic<f32>,
    h: atomic<f32>,
    rot: atomic<f32>,
    r: atomic<f32>,
    g: atomic<f32>,
    b: atomic<f32>,
    a: atomic<f32>,
    scale_x: atomic<f32>,
    scale_y: atomic<f32>,
}
