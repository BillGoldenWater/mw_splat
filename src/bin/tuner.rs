// SPDX-FileCopyrightText: 2026 Golden_Water
//
// SPDX-License-Identifier: MPL-2.0

use std::path::PathBuf;

use anyhow::{Result, bail};
use clap::Parser;
use image::{ImageBuffer, Rgba};
use mw_splat::{
    gpu::GpuContext,
    init::init,
    learn::{Config, Learner, Lr},
    utils::{init_env, load_premultiplied},
};

const N_PARAMS: usize = 7;
const DEFAULT_REDISTRIBUTE: f32 = 30.0;
const EXPLODE_PENALTY: f32 = 1.0;
const H: f32 = 0.25;
const GD_STEP: f32 = 0.25;

#[derive(Parser)]
struct Cli {
    #[arg(long, num_args = 1.., required = true)]
    image: Vec<PathBuf>,

    #[arg(long, num_args = 1.., required = true)]
    splats: Vec<usize>,

    #[arg(long, default_value_t = 1000)]
    iters: u32,

    #[arg(long, default_value_t = 1)]
    runs: usize,

    #[arg(long, default_value_t = 20)]
    steps: usize,

    #[arg(long)]
    no_redistribute: bool,
}

fn main() -> Result<()> {
    init_env();
    let cli = Cli::parse();
    let iters = cli.iters.max(1);
    let tune_redistribute = !cli.no_redistribute;
    let n_params = if tune_redistribute {
        N_PARAMS
    } else {
        N_PARAMS - 1
    };

    let images = cli
        .image
        .iter()
        .map(|p| load_premultiplied(p))
        .collect::<Result<Vec<_>>>()?;

    let splats = if cli.splats.len() == 1 {
        vec![cli.splats[0]; images.len()]
    } else if cli.splats.len() == images.len() {
        cli.splats.clone()
    } else {
        bail!("--splats must be a single count or one per image");
    };

    let gpu = GpuContext::new()?;

    let mut theta = default_theta();
    let mut best = (f32::INFINITY, theta);

    for step in 0..cli.steps {
        let f0 = evaluate(
            &gpu,
            &images,
            &splats,
            theta,
            iters,
            cli.runs,
            tune_redistribute,
        );
        if f0 < best.0 {
            best = (f0, theta);
        }

        let mut grad = [0.0f32; N_PARAMS];
        for i in 0..n_params {
            let mut tp = theta;
            tp[i] += H;
            grad[i] = (evaluate(
                &gpu,
                &images,
                &splats,
                tp,
                iters,
                cli.runs,
                tune_redistribute,
            ) - f0)
                / H;
        }

        let norm =
            grad[..n_params].iter().map(|g| g * g).sum::<f32>().sqrt();
        if norm > 0.0 {
            for i in 0..n_params {
                theta[i] -= GD_STEP * grad[i] / norm;
            }
        }
        theta = clamp_theta(theta, iters, tune_redistribute);

        let (lr, redistribute) =
            theta_to_params(theta, iters, tune_redistribute);
        tracing::info!(
            "step {step}: loss {f0:.6e} (best {:.6e}) redistribute {redistribute:?} lr {lr:?}",
            best.0
        );
    }

    let f_final = evaluate(
        &gpu,
        &images,
        &splats,
        theta,
        iters,
        cli.runs,
        tune_redistribute,
    );
    if f_final < best.0 {
        best = (f_final, theta);
    }

    let (lr, redistribute) =
        theta_to_params(best.1, iters, tune_redistribute);
    tracing::info!(
        "best loss {:.6e}, redistribute {redistribute:?}, lr {lr:?}",
        best.0
    );

    Ok(())
}

fn default_theta() -> [f32; N_PARAMS] {
    let lr = Lr::default();
    [
        lr.position.log2(),
        lr.size.log2(),
        lr.rotation.log2(),
        lr.color.log2(),
        lr.alpha.log2(),
        lr.scale.log2(),
        DEFAULT_REDISTRIBUTE.log2(),
    ]
}

fn clamp_theta(
    theta: [f32; N_PARAMS],
    iters: u32,
    tune_redistribute: bool,
) -> [f32; N_PARAMS] {
    let mut t = theta;
    for v in &mut t[..6] {
        *v = v.clamp(-20.0, 2.0);
    }
    if tune_redistribute {
        t[6] = t[6].clamp(0.0, (iters as f32).log2());
    }
    t
}

fn theta_to_params(
    theta: [f32; N_PARAMS],
    iters: u32,
    tune_redistribute: bool,
) -> (Lr, Option<u32>) {
    let lr = Lr {
        position: 2f32.powf(theta[0]),
        size: 2f32.powf(theta[1]),
        rotation: 2f32.powf(theta[2]),
        color: 2f32.powf(theta[3]),
        alpha: 2f32.powf(theta[4]),
        scale: 2f32.powf(theta[5]),
    };
    let redistribute = if tune_redistribute {
        Some((2f32.powf(theta[6]).round() as u32).clamp(1, iters))
    } else {
        None
    };
    (lr, redistribute)
}

fn evaluate(
    gpu: &GpuContext,
    images: &[ImageBuffer<Rgba<f32>, Vec<f32>>],
    splats: &[usize],
    theta: [f32; N_PARAMS],
    iters: u32,
    runs: usize,
    tune_redistribute: bool,
) -> f32 {
    let (lr, redistribute) =
        theta_to_params(theta, iters, tune_redistribute);
    tracing::debug!("{lr:?}");
    let config = Config { lr_scale: 1.0, lr };

    let mut sum = 0.0f32;
    let mut count = 0usize;
    for (image, &n) in images.iter().zip(splats) {
        for run in 0..runs {
            let ir = init(image, n, run as u64);
            let mut learner =
                Learner::new(gpu.clone(), image, config, &ir);

            learner.run(1, 1.0, None, run as u64);
            let first = learner.read_loss();

            if iters > 1 {
                learner.run(iters - 1, 1.0, redistribute, run as u64);
            }
            let loss = learner.read_loss();

            sum += if loss.is_finite() && loss <= first {
                loss
            } else {
                EXPLODE_PENALTY
            };
            count += 1;
        }
    }
    sum / count.max(1) as f32
}
