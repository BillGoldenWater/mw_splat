pub const GPU_MAX_IN_FLIGHT: usize = 4;

pub const TILE_SIZE: u32 = 16;
pub const WG_SIZE_2D: u32 = 16;
pub const WG_SIZE: u32 = WG_SIZE_2D * WG_SIZE_2D;

// backward pass
pub const GRAD_COPIES: u64 = 8;

// redistribute pass
/// inverse CDF slot count
pub const INVERSE_CDF_SIZE: u32 = 1 << 14;
