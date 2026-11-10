/*
 * SPDX-FileCopyrightText: 2026 Golden_Water
 *
 * SPDX-License-Identifier: MPL-2.0
 */

const LOGIT_EPS: f32 = 1e-4;
const WG_SIZE_2D: u32 = package::_constants::WG_SIZE_2D;
const WG_SIZE: u32 = package::_constants::WG_SIZE;

const TILE_SIZE: u32 = package::_constants::TILE_SIZE;

// backward pass
const BWD_BLOCK_SIZE = 2u;
const BWD_BLOCK_LEN = BWD_BLOCK_SIZE * BWD_BLOCK_SIZE;
const BWD_WG_SIZE = TILE_SIZE / BWD_BLOCK_SIZE;
const BWD_WG_LEN = BWD_WG_SIZE * BWD_WG_SIZE;
const GRAD_COPIES: u32 = package::_constants::GRAD_COPIES;

// redistribute pass
const INVERSE_CDF_SIZE: u32 = package::_constants::INVERSE_CDF_SIZE;
