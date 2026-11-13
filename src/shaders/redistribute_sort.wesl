/*
 * SPDX-FileCopyrightText: 2026 Golden_Water
 *
 * SPDX-License-Identifier: MPL-2.0
 */

import package::libs::types::Splat;
import package::libs::constants::WG_SIZE;

@group(1) @binding(0)
var<storage, read_write> splats: array<Splat>;

@group(2) @binding(0)
var<storage, read_write> moment1: array<Splat>;
@group(2) @binding(1)
var<storage, read_write> moment2: array<Splat>;

struct Imm {
    stage: u32,
    step: u32,
    step_log2: u32,
    step_mod_mask: u32,
}
var<immediate> imm: Imm;

@compute @workgroup_size(WG_SIZE)
fn redistribute_sort(
    @builtin(global_invocation_id) id: vec3u,
    @builtin(num_workgroups) wg_num: vec3u,
) {
    let op_id = id.x + id.y * WG_SIZE * wg_num.x;

    let offset = (op_id >> imm.step_log2) << (imm.step_log2 + 1u);
    let rem = op_id & imm.step_mod_mask;
    let left = offset + rem;
    let right = select(left + imm.step, offset + imm.stage - 1u - rem, imm.stage >> 1u == imm.step);
    if right >= arrayLength(&splats) {
        return;
    }
    if splats[left].order > splats[right].order {
        swap(left, right);
    }
}

fn swap(left: u32, right: u32) {
    let s = splats[left];
    splats[left] = splats[right];
    splats[right] = s;

    let m = moment1[left];
    moment1[left] = moment1[right];
    moment1[right] = m;

    let v = moment2[left];
    moment2[left] = moment2[right];
    moment2[right] = v;
}
