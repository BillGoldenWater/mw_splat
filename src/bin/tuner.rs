// SPDX-FileCopyrightText: 2026 Golden_Water
//
// SPDX-License-Identifier: MPL-2.0

use std::{collections::HashMap, path::PathBuf};

use anyhow::{Result, bail};
use clap::Parser;
use image::{ImageBuffer, Rgba};
use mw_splat::{
    gpu::GpuContext,
    init::init,
    learn::{Config, Learner, Lr, scheduler},
    utils::{init_env, load_premultiplied},
};

const NAMES: [&str; 6] = [
    "position", "size", "rotation", "color", "alpha", "scale",
];
const N_PARAMS: usize = NAMES.len();
const REFINE_TO: f32 = 0.1;

#[derive(Parser)]
struct Cli {
    #[arg(long, num_args = 1.., required = true)]
    image: Vec<PathBuf>,

    #[arg(long, num_args = 1.., required = true)]
    splats: Vec<usize>,

    /// Fit iterations per evaluation, before the refinement pass.
    /// Must match how the learning rates are used, a short budget tunes for
    /// a transient that the real fits never sit in.
    #[arg(long, default_value_t = 2000)]
    iters: u32,

    /// Refinement iterations per evaluation, 0 to skip.
    #[arg(long, default_value_t = 2000)]
    refine: u32,

    /// Redistribute every this many iterations, as the fits are run.
    #[arg(long, default_value_t = 50)]
    redistribute: u32,

    /// Seeds averaged per evaluation.
    #[arg(long, default_value_t = 2)]
    runs: usize,

    /// Coordinate passes after the coarse grid.
    #[arg(long, default_value_t = 3)]
    passes: usize,

    /// Relative loss improvement a coordinate must show to be accepted.
    /// Below this the objective is flat, and following it just walks the rates
    /// into the plateau where they stop doing anything.
    #[arg(long, default_value_t = 2e-3)]
    min_gain: f32,

    /// Coordinates the objective cannot resolve, skipped by default: the peak
    /// alpha rate changes the loss by ~3% across four orders of magnitude and
    /// non-monotonically, well inside the seed spread, so searching it only
    /// walks it into the plateau where it stops doing anything.
    #[arg(long, default_value = "alpha")]
    skip: String,

    /// Print the objective over one coordinate instead of searching.
    #[arg(long)]
    sweep: Option<String>,
}

fn main() -> Result<()> {
    init_env();
    let cli = Cli::parse();
    let images = cli
        .image
        .iter()
        .map(|p| load_premultiplied(p))
        .collect::<Result<Vec<_>>>()?;
    let splats = match cli.splats.len() {
        1 => vec![cli.splats[0]; images.len()],
        n if n == images.len() => cli.splats.clone(),
        _ => bail!("--splats must be a single count or one per image"),
    };

    let gpu = GpuContext::new()?;
    let mut eval = Eval {
        gpu: &gpu,
        images: &images,
        splats: &splats,
        iters: cli.iters.max(1),
        refine: cli.refine,
        redistribute: cli.redistribute,
        runs: cli.runs.max(1),
        cache: HashMap::new(),
        min_gain: cli.min_gain,
    };

    let skip: Vec<&str> = cli.skip.split(',').filter(|s| !s.is_empty()).collect();
    let search: Vec<usize> = (0..N_PARAMS)
        .filter(|i| !skip.contains(&NAMES[*i]))
        .collect();

    let mut theta = default_theta();
    if let Some(name) = &cli.sweep {
        let index = NAMES
            .iter()
            .position(|n| n == name)
            .ok_or_else(|| anyhow::anyhow!("unknown parameter {name}"))?;
        let mut line = theta;
        for half in (-16..=4).step_by(3) {
            line[index] = half as f32;
            let loss = eval.loss(line);
            tracing::info!(
                "{name} {:>10.5} ({:>5.1} log2) loss {loss:.6e} psnr {:.2} dB",
                2f32.powf(half as f32),
                half,
                loss.log10() * -10.
            );
        }
        return Ok(());
    }

    // coarse log2 grid per coordinate, then pattern passes.
    // the grid walks outwards from the seed: a rate far from it only explodes,
    // and those evaluations are slow (bin growth) as well as useless.
    let mut best = eval.loss(theta);
    tracing::info!("start loss {best:.6e} {theta:?}");
    for &i in &search {
        let mut pick = theta;
        let mut lo = theta[i] - 9.0;
        let mut hi = theta[i] + 3.0;
        loop {
            for half in [lo, lo + 3.0, hi - 3.0, hi] {
                let mut t = pick;
                t[i] = half;
                let loss = eval.loss(t);
                if eval.better(loss, best) {
                    best = loss;
                    pick = t;
                }
            }
            let at_edge = pick[i] <= lo + 1e-3 || pick[i] >= hi - 1e-3;
            if !at_edge || hi - lo > 24.0 {
                break;
            }
            lo -= 3.0;
            hi += 3.0;
        }
        theta = pick;
        tracing::info!("grid {:<8} loss {best:.6e} {theta:?}", NAMES[i]);
    }

    for pass in 0..cli.passes {
        let step = 0.5f32.powi(pass as i32);
        let mut improved = false;
        for &i in &search {
            let mut pick = theta;
            let mut local = best;
            for delta in [-2.0, -1.0, 1.0, 2.0] {
                let mut t = theta;
                t[i] = (t[i] + delta * step).clamp(-20.0, 2.0);
                let loss = eval.loss(t);
                if eval.better(loss, local) {
                    local = loss;
                    pick = t;
                }
            }
            if local < best {
                best = local;
                theta = pick;
                improved = true;
            }
        }
        tracing::info!("pass {pass} (step {step}) loss {best:.6e} {theta:?}");
        if !improved && pass > 0 {
            break;
        }
    }

    let lr = theta_to_lr(theta);
    tracing::info!("best loss {best:.6e}, PSNR {:.2} dB", best.log10() * -10.);
    tracing::info!("{lr:#?}");
    Ok(())
}

struct Eval<'a> {
    min_gain: f32,
    gpu: &'a GpuContext,
    images: &'a [ImageBuffer<Rgba<f32>, Vec<f32>>],
    splats: &'a [usize],
    iters: u32,
    refine: u32,
    redistribute: u32,
    runs: usize,
    cache: HashMap<[u32; N_PARAMS], f32>,
}

impl Eval<'_> {
    /// strict improvement by the margin, so a flat objective cannot move a rate
    fn better(&self, loss: f32, best: f32) -> bool {
        loss < best * (1. - self.min_gain)
    }

    fn loss(&mut self, theta: [f32; N_PARAMS]) -> f32 {
        let key = theta.map(f32::to_bits);
        if let Some(&loss) = self.cache.get(&key) {
            return loss;
        }

        let lr = theta_to_lr(theta);
        let config = Config { lr };
        let mut sum = 0.0f32;
        let mut count = 0usize;
        for (image, &n) in self.images.iter().zip(self.splats) {
            for run in 0..self.runs {
                let seed = run as u64;
                let ir = init(image, n, seed);
                let mut learner =
                    Learner::new(self.gpu.clone(), image, config, &ir);
                learner.run(scheduler::default(
                    self.iters,
                    1.0,
                    None,
                    Some(self.redistribute),
                    seed,
                ));
                if self.refine > 0 {
                    learner.run(scheduler::default(
                        self.refine,
                        1.0,
                        Some(REFINE_TO),
                        None,
                        seed,
                    ));
                }
                let loss = learner.read_loss();
                sum += if loss.is_finite() { loss } else { 1. };
                count += 1;
            }
        }

        let loss = sum / count.max(1) as f32;
        self.cache.insert(key, loss);
        loss
    }
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
    ]
}

fn theta_to_lr(theta: [f32; N_PARAMS]) -> Lr {
    Lr {
        position: 2f32.powf(theta[0]),
        size: 2f32.powf(theta[1]),
        rotation: 2f32.powf(theta[2]),
        color: 2f32.powf(theta[3]),
        alpha: 2f32.powf(theta[4]),
        scale: 2f32.powf(theta[5]),
    }
}
