import package::libs::types::BinHeader;
import package::libs::constants::WG_SIZE;

// will overwritten
@group(2) @binding(0)
var<storage, read> bin_heads: array<BinHeader>;

@group(3) @binding(0)
var<storage, read_write> bin_data: array<u32>;

@compute @workgroup_size(WG_SIZE)
fn bin_sort(
    @builtin(workgroup_id) wg_id: vec3u,
    @builtin(num_workgroups) wg_num: vec3u,
    @builtin(local_invocation_index) lii: u32,
) {
    let bin_idx = wg_id.x + wg_id.y * wg_num.x;
    if bin_idx >= arrayLength(&bin_heads) {
        return;
    }
    let head = bin_heads[bin_idx];
    if head.len < 2u {
        return;
    }

    let base = head.base;
    let len = head.len;
    let n = next_pow2(len);

    for(var stage = 2u; stage <= n; stage <<= 1u) {
        let first = stage >> 1u;
        for(var step = first; step > 0u; step >>= 1u) {
            for(var op = lii; op < (n >> 1u); op += WG_SIZE) {
                let r = op & (step - 1u);
                let offset = (op - r) << 1u;
                let left = offset + r;
                let right = select(left + step, offset + stage - 1u - r, step == first);
                if right >= len {
                    continue;
                }
                let a = bin_data[base + left];
                let b = bin_data[base + right];
                if a > b {
                    bin_data[base + left] = b;
                    bin_data[base + right] = a;
                }
            }
            storageBarrier();
        }
    }
}

fn next_pow2(n: u32) -> u32 {
    if n <= 1u {
        return 1;
    }

    if n > 1 << 31 {
        return 0;
    }

    return 1u << (firstLeadingBit(n - 1) + 1);
}
