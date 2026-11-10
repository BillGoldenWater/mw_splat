// SPDX-FileCopyrightText: 2026 Golden_Water
//
// SPDX-License-Identifier: MPL-2.0

use std::path::Path;

use anyhow::{Context as _, Result};
use image::{ImageBuffer, Rgba};
use tracing_subscriber::EnvFilter;

pub mod image_ext;

pub fn init_env() {
    dotenvy::dotenv().ok();
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();
}

pub fn load_premultiplied(
    path: &Path,
) -> Result<ImageBuffer<Rgba<f32>, Vec<f32>>> {
    let mut image = image::open(path)
        .with_context(|| format!("failed to load {}", path.display()))?
        .to_rgba32f();
    image.pixels_mut().for_each(|p| {
        p.0[0] *= p.0[3];
        p.0[1] *= p.0[3];
        p.0[2] *= p.0[3];
    });
    Ok(image)
}
