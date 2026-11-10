import package::libs::types::{BinHeader, BinOverflow};
import package::libs::constants::WG_SIZE;

// CPU: need clear every pass
@group(2) @binding(0)
var<storage, read> bin_counts: array<u32>;
// will overwritten
@group(2) @binding(1)
var<storage, read_write> bin_heads: array<BinHeader>;
// CPU: need reset every pass
@group(2) @binding(2)
var<storage, read_write> bin_bump: atomic<u32>;
@group(2) @binding(3)
var<storage, read_write> bin_overflow: BinOverflow;

// will overwritten
@group(3) @binding(0)
var<storage, read> bin_data: array<u32>;

struct Imm {
    iter_num: u32,
}
var<immediate> imm: Imm;

var<workgroup> wg_bin_buf: array<u32, WG_SIZE>;
var<workgroup> wg_bin_base: u32;
@compute @workgroup_size(WG_SIZE)
fn bin_alloc(@builtin(workgroup_id) wg_id: vec3u, @builtin(local_invocation_index) lii: u32) {
    let bin_idx = wg_id.x * WG_SIZE + lii;
    var bin_count = 0u;
    if bin_idx < arrayLength(&bin_counts) {
        bin_count = bin_counts[bin_idx];
    }

    wg_bin_buf[lii] = bin_count;
    workgroupBarrier();
    for(var depth = 1u; depth < WG_SIZE; depth <<= 1) {
        var value = 0u;
        if lii >= depth {
            value = wg_bin_buf[lii - depth];
        }
        workgroupBarrier();
        if lii >= depth {
            wg_bin_buf[lii] += value;
        }
        workgroupBarrier();
    }

    let wg_total = wg_bin_buf[WG_SIZE - 1];
    if lii == 0 {
        wg_bin_base = atomicAdd(&bin_bump, wg_total);
        let overflow = u32((wg_bin_base + wg_total) > arrayLength(&bin_data));
        let prev = atomicOr(&bin_overflow.overflow, overflow);
        if prev == 0 && overflow == 1 {
            for(var i = 0u; i < arrayLength(&bin_overflow.indirect_args); i += 1) {
                bin_overflow.indirect_args[i] = vec3u(0);
            }
            bin_overflow.iter_num = imm.iter_num;
        }
    }
    workgroupBarrier();

    if bin_idx < arrayLength(&bin_counts) {
        let base = wg_bin_base + wg_bin_buf[lii] - bin_count;
        let cap = arrayLength(&bin_data);
        let len = select(0, min(bin_count, cap - base), base < cap);
        bin_heads[bin_idx] = BinHeader(base, len);
    }
}
