// SPDX-FileCopyrightText: 2026 Golden_Water
//
// SPDX-License-Identifier: MPL-2.0

use std::{fs, path::Path};

use image::{DynamicImage, ImageBuffer, Rgba};
use mw_splat::{
    gpu::GpuContext,
    ir::Ir,
    learn::Learner,
    utils::{init_env, load_premultiplied},
};

fn main() -> anyhow::Result<()> {
    init_env();

    let gpu = GpuContext::new()?;

    let image = load_premultiplied(Path::new("./inputs/input.png"))?;
    // let image = DynamicImage::ImageRgba32F(image).resize_exact(
    //     1920,
    //     1080,
    //     image::imageops::FilterType::Lanczos3,
    // );
    // let image = image.into_rgba32f();

    let out = Path::new("./out");

    let prefix = out.join("0");
    let mut ir = mw_splat::init::init(&image, 8000, 0);
    ir = run(gpu.clone(), &ir, &image, 30_000, Some(2000), 1.);
    save(&gpu, &ir, &prefix, &image);

    // ir = run(gpu.clone(), &ir, &image, 3000, None, 0.25);
    // let prefix = out.join("final");
    // save(&gpu, &ir, &prefix, &image);

    // for i in 1..=7 {
    //     let next =
    //         ((doc.splats.len() as f64 * 2.).ceil() as usize).min(8000);
    //     doc = prune(doc, 0.1);
    //     doc = run(&gpu, &doc, &image, 100);
    //
    //     let last_out = out.join(format!("{}_pruned/", i - 1));
    //     save(&doc, &last_out, dims);
    //     let bg =
    //         load_premultiplied(&last_out.join("rendered.png")).unwrap();
    //     let prefix = out.join(i.to_string());
    //     fs::create_dir_all(&prefix).unwrap();
    //     let append = mw_splat::init::init(
    //         &image,
    //         Some(&bg),
    //         dbg!(next - doc.splats.len()),
    //         Some(dbg!(next)),
    //         Some(&prefix),
    //     )
    //     .unwrap();
    //     doc.splats.extend(append.splats);
    //     doc.meta.splats = dbg!(doc.splats.len());
    //     doc = run(&gpu, &doc, &image, 500);
    //     save(&doc, &prefix, dims);
    // }
    //
    // doc = run(&gpu, &doc, &image, 3000);
    // let prefix = out.join("final");
    // save(&doc, &prefix, dims);

    Ok(())
}

fn save(
    gpu: &GpuContext,
    ir: &Ir,
    prefix: &Path,
    reference: &ImageBuffer<Rgba<f32>, Vec<f32>>,
) {
    fs::create_dir_all(prefix).unwrap();
    ir.save_json(prefix.join("splats.json")).unwrap();
    let mut learner =
        Learner::new(gpu.clone(), reference, Default::default(), ir);

    let render = learner.render();
    DynamicImage::ImageRgba32F(render)
        .into_rgba8()
        .save(prefix.join("rendered.png"))
        .unwrap();

    let render_outlined = learner.render_outlined();
    DynamicImage::ImageRgba32F(render_outlined)
        .into_rgba8()
        .save(prefix.join("rendered_outline.png"))
        .unwrap();
}

fn run(
    gpu: GpuContext,
    ir: &Ir,
    reference: &ImageBuffer<Rgba<f32>, Vec<f32>>,
    iters: u32,
    redistribute: Option<u32>,
    lr: f32,
) -> Ir {
    let mut learner =
        Learner::new(gpu, reference, Default::default(), ir);
    learner.run(iters, lr, redistribute, 1);
    let mut ir = ir.clone();
    ir.splats = learner.read_splats();
    ir
}
