// SPDX-FileCopyrightText: 2026 Golden_Water
//
// SPDX-License-Identifier: MPL-2.0

use std::{ops::Range, path::Path};

use anyhow::{Context as _, Result};
use image::{ImageBuffer, Luma, Rgba};
use tracing_subscriber::EnvFilter;

use crate::constants::LOSS_WEIGHT_GAMMA;

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

pub fn load_loss_weights(
    path: &Path,
) -> Result<ImageBuffer<Luma<f32>, Vec<f32>>> {
    let mut image = image::open(path)
        .with_context(|| format!("failed to load {}", path.display()))?
        .to_luma32f();
    image
        .pixels_mut()
        .for_each(|p| p.0[0] = p.0[0].powf(LOSS_WEIGHT_GAMMA));
    Ok(image)
}

pub fn copy_with_stride(
    width: usize,
    height: usize,
    src_stride: usize,
    tgt_stride: usize,
    rev_y: bool,
    mut copy: impl FnMut(Range<usize>, Range<usize>),
) {
    let y_range = 0..height;
    if rev_y {
        for (src_y, tgt_y) in (y_range.clone()).zip(y_range.rev()) {
            let src_base = src_y * src_stride;
            let tgt_base = tgt_y * tgt_stride;

            let src_range = src_base..src_base + width;
            let tgt_range = tgt_base..tgt_base + width;
            copy(src_range, tgt_range);
        }
    } else {
        for (src_y, tgt_y) in (y_range.clone()).zip(y_range) {
            let src_base = src_y * src_stride;
            let tgt_base = tgt_y * tgt_stride;

            let src_range = src_base..src_base + width;
            let tgt_range = tgt_base..tgt_base + width;
            copy(src_range, tgt_range);
        }
    }
}
