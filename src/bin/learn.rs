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
    ir::Ir,
    learn::{Learner, RunConfig},
    utils::{init_env, load_loss_weights, load_premultiplied},
};

#[derive(Parser)]
struct Cli {
    /// Image to load.
    #[arg(short = 'i', long)]
    input: PathBuf,

    /// Optional per-pixel loss weight map (grayscale).
    #[arg(short = 'w', long)]
    weights: Option<PathBuf>,

    /// Optional initial splats JSON to load instead of running init.
    #[arg(short = 'R', long)]
    reuse: Option<PathBuf>,

    /// Splats to emit.
    #[arg(short = 'n', long, default_value_t = 8000)]
    splats: usize,

    /// Iterations to fit for.
    #[arg(short = 'I', long, default_value_t = 2000)]
    iters: u32,

    /// Redistribute every this many iterations.
    #[arg(short = 'r', long, default_value_t = 50)]
    redistribute: u32,

    /// Global learning-rate scale.
    #[arg(long, default_value_t = 1.0)]
    lr: f32,

    /// Additional refinement iterations after the main fit (0 to skip).
    #[arg(long, default_value_t = 2000)]
    refine: u32,

    /// End learning-rate for the refinement phase.
    #[arg(long, default_value_t = 0.1)]
    lr_refine_to: f32,

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

    let mut ir = match &cli.reuse {
        Some(path) => Ir::load_json(path)?,
        None => init(&input, cli.splats, cli.seed),
    };
    let mut learner = Learner::new(gpu, &input, Default::default(), &ir);
    if let Some(weights) = &cli.weights {
        learner.set_loss_weights(&load_loss_weights(weights)?);
    }
    learner.run(RunConfig {
        iters: cli.iters,
        lr_scale: cli.lr,
        redistribute: Some(cli.redistribute),
        seed: cli.seed,
        ..Default::default()
    });
    if cli.refine != 0 {
        learner.run(RunConfig {
            iters: cli.refine,
            lr_scale: cli.lr,
            lr_anneal_to: Some(cli.lr_refine_to),
            seed: cli.seed,
            ..Default::default()
        });
    }
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
