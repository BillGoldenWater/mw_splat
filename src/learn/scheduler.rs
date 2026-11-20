use crate::learn::Learner;

pub trait Scheduler {
    /// return None if no more iteration,
    /// this should be pure
    fn step(
        &self,
        learner: &Learner,
        iter_num: u32,
    ) -> Option<ScheduleStep>;
}

impl<F> Scheduler for F
where
    F: Fn(&Learner, u32) -> Option<ScheduleStep>,
{
    fn step(
        &self,
        learner: &Learner,
        iter_num: u32,
    ) -> Option<ScheduleStep> {
        self(learner, iter_num)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ScheduleStep {
    pub lr_scale: f32,
    pub redistribute: Option<RedistributeStep>,
}

#[derive(Debug, Clone, Copy)]
pub struct RedistributeStep {
    pub cell_size: u32,
    pub move_count: u32,
    pub splat_size: f32,
    pub seed: u32,
}

pub fn default(
    iters: u32,
    lr_scale: f32,
    lr_anneal_to: Option<f32>,
    redistribute: Option<u32>,
    seed: u64,
) -> impl Scheduler {
    fn hash_seed(base: u64, iter_num: u32) -> u32 {
        let mut x = base
            ^ u64::from(iter_num).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        (x ^ (x >> 31)) as u32
    }

    move |learner: &Learner, iter_num: u32| {
        if iter_num >= iters {
            return None;
        }

        let lr_scale = match lr_anneal_to {
            Some(min) => {
                let t = iter_num as f32 / iters as f32;
                let c = 0.5 * (1.0 + (std::f32::consts::PI * t).cos());
                lr_scale * (min + (1.0 - min) * c)
            }
            None => lr_scale,
        };

        let redistribute = redistribute
            .filter(|every| *every > 0)
            .filter(|every| {
                iter_num > 0 && iter_num.is_multiple_of(*every)
            })
            .map(|_| {
                let progress = iter_num as f32 / iters as f32;
                let canvas = learner.info.canvas.dims;
                let splats = u64::from(learner.splat_count);

                let spacing = (canvas.element_product() as f32
                    / splats.max(1) as f32)
                    .sqrt();

                let cell_size = (spacing
                    * 2f32.powf(1.0 - 3.0 * progress))
                .max(1.0) as u32;
                let cell_count = canvas
                    .map(|it| it.div_ceil(cell_size))
                    .element_product();

                let quantile = 0.25 * 2f32.powf(-3.0 * progress);
                let move_count =
                    (quantile * cell_count as f32).max(1.0) as u32;

                RedistributeStep {
                    cell_size,
                    move_count,
                    splat_size: spacing * 0.9,
                    seed: hash_seed(seed, iter_num),
                }
            });

        Some(ScheduleStep {
            lr_scale,
            redistribute,
        })
    }
}
