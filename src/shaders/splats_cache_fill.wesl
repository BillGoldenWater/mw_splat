import package::libs::types::{Splat, SplatCache};
import package::libs::constants::WG_SIZE;

@group(1) @binding(0)
var<storage, read> splats: array<Splat>;
@group(1) @binding(1)
var<storage, read_write> splat_caches: array<SplatCache>;

@compute @workgroup_size(WG_SIZE)
fn splats_cache_fill(
    @builtin(global_invocation_id) id: vec3u,
    @builtin(num_workgroups) wg_num: vec3u,
) {
    let idx = id.x + id.y * WG_SIZE * wg_num.x;
    if idx >= arrayLength(&splats) {
        return;
    }

    let splat = splats[idx];
    splat_caches[idx] = SplatCache(
            sin(splat.rot),
            cos(splat.rot),
            max(splat.size * splat.fe, vec2f(0.4)),
        );
}
