// SPDX-FileCopyrightText: 2026 Golden_Water
//
// SPDX-License-Identifier: MPL-2.0

use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use mw_splat::utils::{init_env, load_premultiplied};

#[derive(Parser)]
struct Cli {
    /// Image to load.
    #[arg(long)]
    input: PathBuf,

    /// Splats to emit.
    #[arg(long, default_value_t = 100)]
    n: usize,

    /// Blue-noise seed.
    #[arg(long, default_value_t = 0)]
    seed: u64,
}

fn main() -> Result<()> {
    init_env();

    let cli = Cli::parse();
    let input = load_premultiplied(&cli.input)?;
    let ir = mw_splat::init::init(&input, cli.n, cli.seed);
    std::fs::create_dir_all("./out")?;
    ir.save_json("./out/splats.json")?;

    Ok(())
}
