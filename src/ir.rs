// SPDX-FileCopyrightText: 2026 Golden_Water
//
// SPDX-License-Identifier: MPL-2.0

use std::{fs, path::Path};

use anyhow::Context;
use glam::{Vec2, Vec4};
use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct Ir {
    /// width / height
    pub aspect_ratio: f32,
    /// premultiplied
    pub background: Vec4,
    pub splats: Box<[Splat]>,
}

#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct Splat {
    /// in uv space
    /// y up, 0..=1
    pub pos: Vec2,
    /// in uv space
    pub size: Vec2,
    /// radians, +CCW
    pub rot: f32,
    /// premultiplied
    pub color: Vec4,
    pub scale: Vec2,
}

impl Ir {
    pub fn load_json(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let reader = fs::OpenOptions::new()
            .read(true)
            .open(path)
            .context("open file")?;
        serde_json::from_reader(reader).context("serde_json::from_reader")
    }

    pub fn save_json(
        &self,
        path: impl AsRef<Path>,
    ) -> anyhow::Result<()> {
        let writer = fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(path)
            .context("open file")?;
        serde_json::to_writer_pretty(writer, self)
            .context("serde_json::to_writer_pretty")?;
        Ok(())
    }
}
