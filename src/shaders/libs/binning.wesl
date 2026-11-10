import package::libs::types::{Info, Splat};
import package::libs::constants::{TILE_SIZE};

struct BinSplatTileRange {
    valid: bool,
    t_min: vec2u,
    t_max: vec2u,
}

struct BinSplatTileXRange {
    valid: bool,
    t_min: u32,
    t_max: u32,
}

fn splat_tile_range(splat: Splat, info: Info) -> BinSplatTileRange {
    let rot = splat.rot;
    let rot_sin = sin(rot);
    let rot_cos = cos(rot);
    let rot_mat = mat2x2f(
            rot_cos, rot_sin,
            -rot_sin, rot_cos,
        );
    let rot_mat_abs = mat2x2f(
            abs(rot_mat[0]),
            abs(rot_mat[1]),
        );

    // AABB
    let half_size = rot_mat_abs * splat.size * 0.5;
    let pos = splat.pos;

    let b_min = vec2u(clamp(floor(pos - half_size), vec2f(0.), info.canvas.dimsf));
    let b_max = vec2u(clamp(ceil(pos + half_size), vec2f(0.), info.canvas.dimsf));
    if b_min.x >= b_max.x || b_min.y >= b_max.y {
        return BinSplatTileRange(false, vec2u(0), vec2u(0));
    }

    // clamp for NaN safe
    let t_min = clamp(b_min / TILE_SIZE, vec2u(0), info.tile_dims - 1);
    let t_max = clamp((b_max - 1) / TILE_SIZE, vec2u(0), info.tile_dims - 1);

    return BinSplatTileRange(true, t_min, t_max);
}

fn splat_tile_x_range(splat: Splat, info: Info, tile_y: u32) -> BinSplatTileXRange {
    let rot = splat.rot;
    let rot_sin = sin(rot);
    let rot_cos = cos(rot);
    let rot_mat = mat2x2f(
            rot_cos, rot_sin,
            -rot_sin, rot_cos,
        );
    let half_size = splat.size * 0.5;
    let pos = splat.pos;
    let y0 = f32(tile_y * TILE_SIZE);
    let y1 = y0 + f32(TILE_SIZE);

    var corners: array<vec2f, 4>;
    corners[0] = pos + rot_mat * vec2f(-half_size.x, -half_size.y);
    corners[1] = pos + rot_mat * vec2f(half_size.x, -half_size.y);
    corners[2] = pos + rot_mat * vec2f(half_size.x, half_size.y);
    corners[3] = pos + rot_mat * vec2f(-half_size.x, half_size.y);

    var x_min = 1e30;
    var x_max = -1e30;
    var found = false;

    for(var i = 0u; i < 4u; i = i + 1u) {
        let p = corners[i];
        if p.y >= y0 && p.y <= y1 {
            x_min = min(x_min, p.x);
            x_max = max(x_max, p.x);
            found = true;
        }

        let q = corners[(i + 1u) % 4u];
        for(var j = 0u; j < 2u; j = j + 1u) {
            let yb = select(y0, y1, j == 1u);
            let lo = min(p.y, q.y);
            let hi = max(p.y, q.y);
            if yb > lo && yb < hi {
                let t = (yb - p.y) / (q.y - p.y);
                let x = p.x + (q.x - p.x) * t;
                x_min = min(x_min, x);
                x_max = max(x_max, x);
                found = true;
            }
        }
    }

    if !found {
        return BinSplatTileXRange(false, 0u, 0u);
    }

    let b_min = clamp(floor(x_min), 0.0, info.canvas.dimsf.x);
    let b_max = clamp(ceil(x_max), 0.0, info.canvas.dimsf.x);
    if b_min >= b_max {
        return BinSplatTileXRange(false, 0u, 0u);
    }

    let t_min = clamp(u32(b_min / f32(TILE_SIZE)), 0u, info.tile_dims.x - 1u);
    let t_max = clamp(u32((b_max - 1.0) / f32(TILE_SIZE)), 0u, info.tile_dims.x - 1u);
    return BinSplatTileXRange(true, t_min, t_max);
}
