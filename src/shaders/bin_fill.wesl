import package::libs::types::{BinHeader, Info, Splat};
import package::libs::constants::WG_SIZE;
import package::libs::binning::{splat_tile_range, splat_tile_x_range};

@group(0) @binding(0)
var<storage, read> info: Info;

@group(1) @binding(0)
var<storage, read> splats: array<Splat>;

// will overwritten
@group(2) @binding(0)
var<storage, read> bin_heads: array<BinHeader>;
// CPU: need clear every pass
@group(2) @binding(1)
var<storage, read_write> bin_cursor: array<atomic<u32>>;

// will overwritten
@group(3) @binding(0)
var<storage, read_write> bin_data: array<u32>;

@compute @workgroup_size(WG_SIZE)
fn bin_fill(
    @builtin(workgroup_id) wg_id: vec3u,
    @builtin(num_workgroups) wg_num: vec3u,
    @builtin(local_invocation_index) l_ii: u32,
) {
    let idx = wg_id.x + wg_id.y * wg_num.x;
    if idx >= arrayLength(&splats) {
        return;
    }

    let range = splat_tile_range(splats[idx], info);
    if !range.valid {
        return;
    }
    let t_min = range.t_min;
    let t_max = range.t_max;

    for(var tile_y = t_min.y + l_ii; tile_y <= t_max.y; tile_y += WG_SIZE) {
        let x_range = splat_tile_x_range(splats[idx], info, tile_y);
        if !x_range.valid {
            continue;
        }
        for(var tile_x = x_range.t_min; tile_x <= x_range.t_max; tile_x += 1) {
            let tile_idx = tile_y * info.tile_dims.x + tile_x;
            let bin_head = bin_heads[tile_idx];
            let bin_cursor = atomicAdd(&bin_cursor[tile_idx], 1);
            if bin_cursor >= bin_head.len {
                continue;
            }
            bin_data[bin_head.base + bin_cursor] = idx;
        }
    }
}
