// SPDX-FileCopyrightText: 2026 Golden_Water
//
// SPDX-License-Identifier: MPL-2.0

use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use image::DynamicImage;
use mw_splat::{
    gpu::GpuContext,
    init::init,
    learn::Learner,
    utils::{init_env, load_premultiplied},
};

#[derive(Parser)]
struct Cli {
    /// Image to load.
    #[arg(short = 'i', long)]
    input: PathBuf,

    /// Splats to emit.
    #[arg(short = 'n', long, default_value_t = 8000)]
    splats: usize,

    /// Iterations to fit for.
    #[arg(short = 'I', long, default_value_t = 3000)]
    iters: u32,

    /// Redistribute every this many iterations.
    #[arg(short = 'r', long, default_value_t = 200)]
    redistribute: u32,

    /// Global learning-rate scale.
    #[arg(long, default_value_t = 1.0)]
    lr: f32,

    /// Seed for intentionally random stuffs.
    #[arg(short = 'e', long, default_value_t = 0)]
    seed: u64,

    /// Output directory.
    #[arg(short = 'o', long, default_value = "./out")]
    output: PathBuf,
}

fn main() -> Result<()> {
    init_env();

    let cli = Cli::parse();
    let input = load_premultiplied(&cli.input)?;
    let gpu = GpuContext::new()?;

    let mut ir = init(&input, cli.splats, cli.seed);
    let mut learner = Learner::new(gpu, &input, Default::default(), &ir);
    learner.run(cli.iters, cli.lr, Some(cli.redistribute), cli.seed);
    ir.splats = learner.read_splats();

    std::fs::create_dir_all(&cli.output)?;
    ir.save_json(cli.output.join("splats.json"))?;

    DynamicImage::ImageRgba32F(learner.render())
        .into_rgba8()
        .save(cli.output.join("render.png"))?;

    DynamicImage::ImageRgba32F(learner.render_outlined())
        .into_rgba8()
        .save(cli.output.join("render_outlined.png"))?;

    Ok(())
}
