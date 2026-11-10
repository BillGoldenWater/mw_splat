// SPDX-FileCopyrightText: 2026 Golden_Water
//
// SPDX-License-Identifier: MPL-2.0

use glam::{Vec3, Vec4, Vec4Swizzles as _};
use image::{ImageBuffer, Rgba};

pub trait ImageBufferExt {
    fn unpremultiply(&mut self);
}

impl ImageBufferExt for ImageBuffer<Rgba<f32>, Vec<f32>> {
    fn unpremultiply(&mut self) {
        for px in self.pixels_mut() {
            let rgba = Vec4::from(px.0);
            let mut rgb = rgba.xyz();
            rgb /= rgba.w;
            rgb = rgb.clamp(Vec3::ZERO, Vec3::ONE);
            px.0[..3].copy_from_slice(&rgb.to_array());
        }
    }
}
