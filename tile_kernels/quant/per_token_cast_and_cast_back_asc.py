"""VMI per-token quantize + dequantize (in-place cast-and-cast-back).

Structural mirror of the upstream ASC source of truth
third_party/TileKernels-Nightly/tile_kernels/quant/per_token_cast_and_cast_back_asc.py.

Structure layer (line-for-line with ASC):
  * _get_block_shape with get_max_ub_per_vector_core(use_simt=False);
  * num_stages = 3 and quant_max / round_carry / quant_max_exp;
  * num_groups = align(block_k // num_per_channels, 8) and
    num_sf = align(block_m * num_groups, 64), use_bf16_sf / sf_inv_dtype,
    use_bf16_dequant / sf_dtype;
  * the ASC macro decomposition: get_sf_from_inv / load_sf_dintlv / load_sf /
    collect_amax_tile / clamp_e4m3 / cast_and_cast_back_f32_pair /
    cast_and_cast_back / cast_and_cast_back_bf16_no_round /
    cast_and_cast_back_bf16 / collect_amax / cast_and_cast_back_per16 /
    cast_and_cast_back_per32 / compute_sf, plus the VMI-only physical packed-FP4
    helpers load_sf_fp4 / cast_and_cast_back_fp4_pair /
    cast_and_cast_back_fp4_quad.  The per-stage tile helpers
    cast_and_cast_back_per32_tile / cast_and_cast_back_per16_fp4_tile are
    inlined into their VF bodies, so the logical rewrite stays textually inside
    with T.SimdVF() and the VF bodies mirror the ASC per-tile orchestration;
  * the same five T.alloc_shared buffers in the same order (sf_ub carries
    sf_dtype, bfloat16 for the per-16 E4M3-SF bf16-dequant path) and the same
    T.annotate_buffer_versions({...: num_stages});
  * with T.Kernel(num_vec_cores) as core_id: plus the same T.Persistent
    call and body order (T.copy -> collect_amax -> compute_sf ->
    cast_and_cast_back_per16/per32 -> T.copy);
  * the same four with T.SimdVF(): regions (amax -> SF -> per16 / per32).

Only the SIMD compute inside T.SimdVF() is rewritten to logical T.vmi.*
vectors.  ASC's physical register-layout repairs are folded away:
vintlv/vdintlv broadcast trees, E2B_* / DINTLV_B16 loads, PART_* conversion
splits, PK_B32 / NORM_B* / ONEPT_B32 store dists and pset pattern masks all
become plain logical vectors, grouped broadcast loads, vcvt/vinterpret_cast
and create_mask.  The packed FP4 (e2m1) round trip is the one place where
packing is real, so it keeps vdintlv/vor/vmins/vintlv/vinterpret_cast.

The kernel is compiled through the VMI path only: the public entry is the
ASC-shaped factory itself, decorated with
tilelang.jit(target='pto', pass_configs=PTO_FAST_MATH), and returns the
prim_func for TileLang to compile.
"""

import os

import tilelang
from tilelang.ascend import language as T

from tile_kernels.config import get_max_ub_per_vector_core
from tile_kernels.quant.common import CastOutputConfig
from tile_kernels.quant.vmi_common import (
    FAST_MATH,
    MX_GROUP,
    PTO_FAST_MATH,
    VL,
    vdiv_with_fast_math,
)
from tile_kernels.utils import align

# BF16 bit-field constants for power-of-two round_sf (ASC values).
_ABS_MASK_BF16 = 0x7FFF
_BF16_EXP_BIAS = 0x7F00
_EXP_MASK_F32 = 0x7F800000
_F32_EXP_BIAS = 0x7F000000

# VF latency profiles (issue #1337).
#
# The value handed to T.SimdVF(latency=...) is a *scheduling* constant: it does
# not estimate the region's real cycle cost, it shapes the pipeline the Z3
# pipeliner emits.  Two frozen profiles are kept and selectable:
#
#   'asc'  -- the calibrated constants that make the emitted VF pipeline
#             per-config equal to ASC's (all 6 task stages + solve II; 10/12
#             e4m3 rows class A).  This is the ASC-alignment profile.  Values
#             from experiments/issue1337-vf-pipeline-align/.
#   'vmi'  -- the frozen per-region costs of the optimised VMI SimdVF body
#             itself.  The emitted pipeline differs from ASC's (VF stages
#             (2,2,3) instead of (1,1,2)) but it is the profile that clears the
#             owner acceptance: 12/12 e4m3 configs asc/vmi >= 1.0 on the
#             rebased base (min 1.0028), every config bit-exact.
#
# Both profiles are hard-coded on purpose.  Do NOT leave the latency unfilled
# (`with T.SimdVF():`): tilelang would then derive it from a static TIR-node
# cost model (src/ascend/transform/estimate_latency.cc) whose numbers drift with
# the compiler base -- the same kernel measured 12/12 before the rebase and two
# failing configs after it, purely because that model's op counts changed.
#
# The default profile is resolved ONCE at module import (from
# `TILE_KERNELS_VMI_VF_LATENCY=asc|vmi`, default 'asc') into
# _DEFAULT_VF_LATENCY_PROFILE below, so it is a bound parameter value and thus
# part of the @tilelang.jit cache key for no-argument calls.  Switch per call
# with `vf_latency_profile='asc'|'vmi'`; the env var is not re-read after import.
_VF_LATENCY_ASC = {
    ('fp32', True, 2048, 'amax'): 8,
    ('fp32', True, 2048, 'scale'): 0,
    ('fp32', True, 2048, 'cast'): 8,
    ('fp32', True, 7168, 'amax'): 8,
    ('fp32', True, 7168, 'scale'): 0,
    ('fp32', True, 7168, 'cast'): 80,
    ('fp32', False, 2048, 'amax'): 600,
    ('fp32', False, 2048, 'scale'): 4,
    ('fp32', False, 2048, 'cast'): 0,
    ('fp32', False, 7168, 'amax'): 400,
    ('fp32', False, 7168, 'scale'): 8,
    ('fp32', False, 7168, 'cast'): 12,
    ('bf16', True, 2048, 'amax'): 8,
    ('bf16', True, 2048, 'scale'): 0,
    ('bf16', True, 2048, 'cast'): 8,
    ('bf16', True, 7168, 'amax'): 8,
    ('bf16', True, 7168, 'scale'): 0,
    ('bf16', True, 7168, 'cast'): 4,
    ('bf16', False, 2048, 'amax'): 0,
    ('bf16', False, 2048, 'scale'): 600,
    ('bf16', False, 2048, 'cast'): 96,
    ('bf16', False, 7168, 'amax'): 400,
    ('bf16', False, 7168, 'scale'): 8,
    ('bf16', False, 7168, 'cast'): 12,
}

_VF_LATENCY_VMI = {
    ('fp32', True, 2048, 'amax'): 17,
    ('fp32', True, 2048, 'scale'): 16,
    ('fp32', True, 2048, 'cast'): 28,
    ('fp32', True, 7168, 'amax'): 13,
    ('fp32', True, 7168, 'scale'): 16,
    ('fp32', True, 7168, 'cast'): 22,
    ('fp32', False, 2048, 'amax'): 17,
    ('fp32', False, 2048, 'scale'): 147,
    ('fp32', False, 2048, 'cast'): 28,
    ('fp32', False, 7168, 'amax'): 13,
    ('fp32', False, 7168, 'scale'): 147,
    ('fp32', False, 7168, 'cast'): 20,
    ('bf16', True, 2048, 'amax'): 27,
    ('bf16', True, 2048, 'scale'): 17,
    ('bf16', True, 2048, 'cast'): 37,
    ('bf16', True, 7168, 'amax'): 27,
    ('bf16', True, 7168, 'scale'): 17,
    ('bf16', True, 7168, 'cast'): 37,
    ('bf16', False, 2048, 'amax'): 27,
    ('bf16', False, 2048, 'scale'): 147,
    ('bf16', False, 2048, 'cast'): 38,
    ('bf16', False, 7168, 'amax'): 27,
    ('bf16', False, 7168, 'scale'): 147,
    ('bf16', False, 7168, 'cast'): 38,
}

_VF_LATENCY_PROFILES = {'asc': _VF_LATENCY_ASC, 'vmi': _VF_LATENCY_VMI}
_VF_LATENCY_PROFILE_ENV = 'TILE_KERNELS_VMI_VF_LATENCY'

def resolve_vf_latency_profile(profile: str | None = None) -> str:
    """Resolve a profile name, defaulting to the env var and then 'asc'."""
    name = profile if profile is not None else os.environ.get(_VF_LATENCY_PROFILE_ENV, 'asc')
    name = name.strip().lower()
    if name not in _VF_LATENCY_PROFILES:
        raise ValueError(
            f'unknown VF latency profile {name!r}; expected one of '
            f'{sorted(_VF_LATENCY_PROFILES)}'
        )
    return name

def eval_vf_latency(dtype_key: str, round_sf: bool, hidden: int, region: str,
                    profile: str | None = None) -> int:
    table = _VF_LATENCY_PROFILES[resolve_vf_latency_profile(profile)]
    return table.get((dtype_key, bool(round_sf), int(hidden), region), 0)

# Default profile for no-argument calls, resolved once at import time so it is a
# bound default and therefore part of the jit cache key.  An explicit
# vf_latency_profile argument still overrides it (and is validated per call).
_DEFAULT_VF_LATENCY_PROFILE = resolve_vf_latency_profile()

def _get_block_shape(hidden: int, dtype: T.dtype, num_stages: int, num_per_channels: int) -> tuple[int, int]:
    """Choose the widest valid block_k and largest UB-fitting block_m (ASC)."""
    stage_budget = get_max_ub_per_vector_core(use_simt=False) // num_stages

    alignment = 256 if hidden % 256 == 0 else 128
    for block_k in range(hidden, alignment - 1, -alignment):
        if hidden % block_k != 0:
            continue
        for block_m in range(32, 0, -1):
            num_sf = align(block_m * align(block_k // num_per_channels, 8), 64)
            ub_usage = block_m * block_k * dtype.bits // 4 + num_sf * 12
            if ub_usage <= stage_budget:
                return block_m, block_k

@tilelang.jit(target='pto', pass_configs=PTO_FAST_MATH)
def get_per_token_cast_and_cast_back_kernel_vmi(
    hidden: int,
    token_stride: int,
    dtype: T.dtype,
    out_config: CastOutputConfig,
    num_vec_cores: int,
    vf_latency_profile: str = _DEFAULT_VF_LATENCY_PROFILE,
):
    """ASC-shaped VMI entry; mirrors get_per_token_cast_and_cast_back_kernel_asc.

    The only structural differences from the ASC source of truth are the jit
    decorator's target/pass_configs, this latency-profile selector (the
    sanctioned _VF_LATENCY exemption), and the logical T.vmi.* bodies inside
    the four T.SimdVF regions.

    vf_latency_profile defaults to _DEFAULT_VF_LATENCY_PROFILE, resolved from
    TILE_KERNELS_VMI_VF_LATENCY at import time, so the default is part of the
    @tilelang.jit cache key and a no-argument call is stable for the process
    lifetime.  Pass an explicit 'asc'/'vmi' to switch per call.
    """
    vf_latency_profile = resolve_vf_latency_profile(vf_latency_profile)
    num_per_channels = out_config.sf_block[1]
    is_bf16_in = dtype == T.bfloat16
    assert dtype in (T.bfloat16, T.float32), 'Ascend per-token cast and cast-back only supports bf16 or fp32 input'
    assert (num_per_channels, out_config.use_e4m3_sf) in ((16, True), (32, False)), 'Ascend supports per-16 with E4M3 SF or per-32 without E4M3 SF'
    assert hidden % 128 == 0, 'Ascend per-token cast and cast-back hidden must be a multiple of 128'

    is_fp4_out = out_config.dtype == T.float4_e2m1fn
    round_sf = out_config.round_sf
    quant_max = 6.0 if is_fp4_out else 448.0
    round_carry = 0x003FFFFF if is_fp4_out else 0x001FFFFF
    quant_max_exp = 0x01000000 if is_fp4_out else 0x04000000

    num_stages = 3

    block_m, block_k = _get_block_shape(hidden, dtype, num_stages, num_per_channels)
    # Pad only the final four scales when a row has a 128-element tail.
    num_groups = align(block_k // num_per_channels, 8)

    num_sf = align(block_m * num_groups, 64)
    use_bf16_sf = is_bf16_in and round_sf
    sf_inv_dtype = T.bfloat16 if use_bf16_sf else T.float32
    use_bf16_dequant = is_bf16_in and out_config.use_e4m3_sf
    sf_dtype = T.bfloat16 if use_bf16_dequant else T.float32

    num_tokens = T.dynamic('num_tokens')

    @T.macro
    def get_sf_from_inv(sf_inv, sf_inv_bias):
        """Recover power-of-two scales from inverse scales using exponent bits."""
        sf_bits = T.vmi.vsub(sf_inv_bias, T.vmi.vinterpret_cast(sf_inv, 'uint16' if use_bf16_sf else 'uint32'))
        return T.vmi.vinterpret_cast(sf_bits, 'bfloat16' if use_bf16_sf else 'float32')

    @T.macro
    def load_sf_dintlv(sf_ub, sf_offset, lanes=VL):
        """Load eight FP32 scales and broadcast each to 16 lanes."""
        return T.vmi.vload(
            sf_ub[sf_offset], size=lanes, stride=1, dist_mode='brc',
            group=lanes // MX_GROUP,
        )

    @T.macro
    def load_sf(sf_ub, sf_offset, lanes=VL):
        """Load eight FP32 scales and broadcast each to 32 lanes."""
        return T.vmi.vload(
            sf_ub[sf_offset],
            size=lanes,
            stride=1,
            dist_mode='brc',
            group=lanes // MX_GROUP,
        )

    @T.macro
    def collect_amax_tile(x_ub, amax_ub, row, col, tile_k):
        """Compute and store the maximum absolute value of each scale group in one tile."""
        sf_offset = row * num_groups + col // num_per_channels
        groups = tile_k // num_per_channels
        if is_bf16_in:
            abs_mask_u16 = T.vmi.vbrc(T.uint16(_ABS_MASK_BF16), size=tile_k)
            values = T.vmi.vload(x_ub[row, col], size=tile_k)
            abs_u16 = T.vmi.vand(
                T.vmi.vinterpret_cast(values, 'uint16'), abs_mask_u16
            )
            amax_u16 = T.vmi.vcmax(abs_u16, group=groups)
            amax_u32 = T.vmi.vcvt(amax_u16, 'uint32')
            amax = T.vmi.vinterpret_cast(
                T.vmi.vshls(amax_u32, 16), 'float32'
            )
            T.vmi.vstore(amax, amax_ub[sf_offset], stride=1, group=groups)
        elif num_per_channels == 16 or tile_k == VL:
            values = T.vmi.vload(x_ub[row, col], size=tile_k)
            amax = T.vmi.vcmax(T.vmi.vabs(values), group=groups)
            T.vmi.vstore(amax, amax_ub[sf_offset], stride=1, group=groups)
        else:
            values = T.vmi.vload(x_ub[row, col], size=tile_k)
            amax = T.vmi.vcmax(T.vmi.vabs(values), group=groups)
            T.vmi.vstore(amax, amax_ub[sf_offset], stride=1, group=groups)

    @T.macro
    def load_sf_fp4(sf_ub, sf_offset):
        """FP4-only 64-lane SF pair loader.

        The packed-FP4 round trip is physically 2x64-lane and VMI has no
        512-lane view, so a 256-lane tile cannot be reinterpreted to u16.  This
        loader therefore stays at 64 lanes and deliberately avoids the
        unaligned dist_mode='brc', group=2 form.
        """
        mask = T.vmi.create_mask(64, size=64)
        hi = T.vmi.vcmps(
            T.vmi.vci(T.uint32(0), size=64), T.uint32(31), mask, 'gt'
        )
        lo0 = T.vmi.vload(sf_ub[sf_offset + 0], size=64, dist_mode='brc')
        up0 = T.vmi.vload(sf_ub[sf_offset + 1], size=64, dist_mode='brc')
        lo1 = T.vmi.vload(sf_ub[sf_offset + 2], size=64, dist_mode='brc')
        up1 = T.vmi.vload(sf_ub[sf_offset + 3], size=64, dist_mode='brc')
        lo2 = T.vmi.vload(sf_ub[sf_offset + 4], size=64, dist_mode='brc')
        up2 = T.vmi.vload(sf_ub[sf_offset + 5], size=64, dist_mode='brc')
        lo3 = T.vmi.vload(sf_ub[sf_offset + 6], size=64, dist_mode='brc')
        up3 = T.vmi.vload(sf_ub[sf_offset + 7], size=64, dist_mode='brc')
        return (
            T.vmi.vsel(hi, up0, lo0),
            T.vmi.vsel(hi, up1, lo1),
            T.vmi.vsel(hi, up2, lo2),
            T.vmi.vsel(hi, up3, lo3),
        )

    @T.macro
    def cast_and_cast_back_fp4_pair(scaled0, scaled1, sf0, sf1):
        """Packed FP4 round trip for two 64-lane chunks (fp4-only path).

        ASC combines the two 64-lane chunks into one 128-lane bf16 vector with
        a vdintlv sticky-round pack, quantizes to packed e2m1, widens back with
        S.vcvt(..., part=0/1) and finally S.vintlv's the two 64-lane halves
        back onto scaled0/scaled1.  The vdintlv here is the same packing, so
        the recovered bf16 vector is the concatenation
        [roundtrip(chunk0) | roundtrip(chunk1)].

        VMI's vcvt has no part selector and keeps the lane count, so the ASC
        widen+interleave is modelled directly in the 16-bit domain: interleaving
        the recovered bf16 bits with zero in the even 16-bit slots, then
        reinterpreting as f32, places each bf16 pattern in the f32 high half --
        exactly bf16->f32 widening -- and the two 128-lane vintlv outputs are
        the two 64-lane chunks that scaled0/scaled1 came from.

        When use_bf16_dequant is set (per-16 E4M3-SF bf16 input) ASC returns the
        recovered bf16 vector rescaled by the bf16 SF directly, over all 128
        lanes, so sf0 carries a 128-lane bf16 broadcast and sf1 is ignored.
        """
        mask = T.vmi.create_mask(64, size=64)
        mask128 = T.vmi.create_mask(128, size=128)
        low, high = T.vmi.vdintlv(
            T.vmi.vinterpret_cast(scaled0, 'uint16'),
            T.vmi.vinterpret_cast(scaled1, 'uint16'),
            mask128,
        )
        scaled_bf16 = T.vmi.vinterpret_cast(
            T.vmi.vor(high, T.vmi.vmins(low, T.uint16(1), mask128), mask128),
            'bfloat16',
        )
        quantized = T.vmi.vcvt(scaled_bf16, 'float4_e2m1fn', rounding='R')
        recovered = T.vmi.vcvt(quantized, 'bfloat16')
        if use_bf16_dequant:
            return T.vmi.vmul(recovered, sf0, mask128)
        # ASC: cast_back0, cast_back1 = S.vintlv(S.vcvt(recovered, f32, part=0),
        #                                        S.vcvt(recovered, f32, part=1)).
        zero_u16 = T.vmi.vbrc(T.uint16(0), size=2 * 64)
        chunk0_u16, chunk1_u16 = T.vmi.vintlv(
            zero_u16, T.vmi.vinterpret_cast(recovered, 'uint16'), mask128
        )
        cast_back0 = T.vmi.vinterpret_cast(chunk0_u16, 'float32')
        cast_back1 = T.vmi.vinterpret_cast(chunk1_u16, 'float32')
        return T.vmi.vmul(cast_back0, sf0, mask), T.vmi.vmul(cast_back1, sf1, mask)

    @T.macro
    def cast_and_cast_back_fp4_quad(x_ub, out_ub, row, base, sf_inv0, sf_inv1, sf0, sf1):
        """One 128-element fp4 half (two 64-lane chunks); blocked path."""
        mask = T.vmi.create_mask(64, size=64)
        if is_bf16_in:
            values0 = T.vmi.vcvt(T.vmi.vload(x_ub[row, base], size=64), 'float32')
            values1 = T.vmi.vcvt(
                T.vmi.vload(x_ub[row, base + 64], size=64), 'float32'
            )
        else:
            values0 = T.vmi.vload(x_ub[row, base], size=64)
            values1 = T.vmi.vload(x_ub[row, base + 64], size=64)
        output0, output1 = cast_and_cast_back_fp4_pair(
            T.vmi.vmul(values0, sf_inv0, mask),
            T.vmi.vmul(values1, sf_inv1, mask),
            sf0,
            sf1,
        )
        # ASC (cast_and_cast_back) narrows to bf16 only for bf16 input; the
        # fp32 arms write the f32 result straight back into the f32 out_ub.
        if is_bf16_in:
            T.vmi.vstore(T.vmi.vcvt(output0, 'bfloat16'), out_ub[row, base], mask)
            T.vmi.vstore(T.vmi.vcvt(output1, 'bfloat16'), out_ub[row, base + 64], mask)
        else:
            T.vmi.vstore(output0, out_ub[row, base], mask)
            T.vmi.vstore(output1, out_ub[row, base + 64], mask)

    @T.macro
    def clamp_e4m3(values, mask):
        """Clamp normalized values to the finite E4M3 range.

        Rounding the SF down can make normalized values exceed E4M3's finite
        range; ASC: S.vmins(S.vmaxs(values, -448.0), 448.0).
        """
        return T.vmi.vmins(
            T.vmi.vmaxs(values, T.float32(-448.0), mask), T.float32(448.0), mask
        )

    @T.macro
    def cast_and_cast_back_f32_pair(scaled, sf, mask):
        """Round-trip one logical f32 strip through e4m3 and dequantize.

        ASC's cast_and_cast_back_f32_pair round-trips a 64-lane pair; VMI's
        logical strip already spans both lanes, so the pair folds into this
        single logical chain (scaled is the scaled pair, sf its matching scale).
        When use_e4m3_sf is set the normalized values are clamped to the finite
        E4M3 range before the conversion (ASC clamp_e4m3).  The packed-FP4 arms
        keep their physical 2x64-lane pack in cast_and_cast_back_fp4_pair.
        """
        if out_config.use_e4m3_sf:
            scaled = clamp_e4m3(scaled, mask)
        cast_back = T.vmi.vcvt(
            T.vmi.vcvt(scaled, 'float8_e4m3fn', rounding='R', saturate='SAT'), 'float32'
        )
        return T.vmi.vmul(cast_back, sf, mask)

    @T.macro
    def cast_and_cast_back(x_ub, out_ub, row, col, sf, sf_inv, zero_bf16, lanes=VL):
        """One logical tile: scale by the reciprocal, round-trip, rescale.

        A full tile is one 256-lane logical vector; the 128-lane tail passes
        lanes=128.  zero_bf16 is unused by the logical form and kept
        only for the ASC structural signature.

        For the per-16 E4M3-SF bf16-dequant path ASC converts the f32
        round-trip back to bf16 and multiplies by the bf16 SF directly
        (vmul(cast_back_bf16, sf0)), so this logical arm does the same: sf is a
        128-lane bf16 broadcast and the result is stored as bf16.
        """
        mask = T.vmi.create_mask(lanes, size=lanes)
        if is_bf16_in:
            values = T.vmi.vcvt(T.vmi.vload(x_ub[row, col], size=lanes), 'float32')
        else:
            values = T.vmi.vload(x_ub[row, col], size=lanes)
        scaled = T.vmi.vmul(values, sf_inv, mask)
        if use_bf16_dequant:
            cast_back = T.vmi.vcvt(
                T.vmi.vcvt(clamp_e4m3(scaled, mask), 'float8_e4m3fn',
                           rounding='R', saturate='SAT'),
                'float32',
            )
            T.vmi.vstore(
                T.vmi.vmul(T.vmi.vcvt(cast_back, 'bfloat16'), sf, mask),
                out_ub[row, col], mask,
            )
        else:
            output = cast_and_cast_back_f32_pair(scaled, sf, mask)
            if is_bf16_in:
                T.vmi.vstore(T.vmi.vcvt(output, 'bfloat16'), out_ub[row, col], mask)
            else:
                T.vmi.vstore(output, out_ub[row, col], mask)

    @T.macro
    def cast_and_cast_back_bf16_no_round(x_ub, out_ub, row, col, sf, sf_inv, lanes=VL):
        """256-element bf16 no-round tile: widen, e4m3 round trip, rescale.

        ASC's vld2 DINTLV dual-64 plus vintlv re-pack folds into one logical
        widening vcvt and one epilogue; the 128-lane tail passes
        lanes=128.
        """
        mask = T.vmi.create_mask(lanes, size=lanes)
        values = T.vmi.vcvt(T.vmi.vload(x_ub[row, col], size=lanes), 'float32')
        scaled = T.vmi.vmul(values, sf_inv, mask)
        cast_back = T.vmi.vcvt(
            T.vmi.vcvt(scaled, 'float8_e4m3fn', rounding='R', saturate='SAT'), 'float32'
        )
        T.vmi.vstore(
            T.vmi.vcvt(T.vmi.vmul(cast_back, sf, mask), 'bfloat16'), out_ub[row, col], mask
        )

    @T.macro
    def cast_and_cast_back_bf16(scaled, sf, lanes):
        """bf16 round_sf tile: bf16 -> e4m3/e2m1 -> bf16, rescaled by bf16 sf."""
        mask = T.vmi.create_mask(lanes, size=lanes)
        if is_fp4_out:
            # ASC multiplies the recovered bf16 value by the bf16 scale directly
            # in the bf16 domain (S.vmul(cast_back, sf)).  The previous f32
            # widen/multiply/narrow added two vcvt per element plus a 2x-issue
            # f32 vmul for the one config that reaches this body
            # (bf16 x round_sf x e2m1).
            quantized = T.vmi.vcvt(scaled, 'float4_e2m1fn', rounding='R')
            recovered = T.vmi.vcvt(quantized, 'bfloat16')
            return T.vmi.vmul(recovered, sf, mask)
        else:
            scaled_f = T.vmi.vcvt(scaled, 'float32')
            quantized = T.vmi.vcvt(scaled_f, 'float8_e4m3fn', rounding='R', saturate='SAT')
            rescaled = T.vmi.vmul(T.vmi.vcvt(quantized, 'float32'), T.vmi.vcvt(sf, 'float32'), mask)
        return T.vmi.vcvt(rescaled, 'bfloat16')

    @T.macro
    def collect_amax(x_ub, amax_ub):
        """Collect per-group maxima across all rows and tiles in the UB block."""
        with T.SimdVF(latency=eval_vf_latency('bf16' if is_bf16_in else 'fp32', round_sf, hidden, 'amax',
                                              profile=vf_latency_profile)):
            tile_k = num_per_channels * 8
            for row in T.serial(block_m):
                for tile in T.serial(block_k // tile_k):
                    collect_amax_tile(x_ub, amax_ub, row, tile * tile_k, tile_k)
                if block_k % tile_k:
                    collect_amax_tile(x_ub, amax_ub, row, block_k - 128, 128)

    @T.macro
    def compute_sf(amax_ub, sf_ub, sf_inv_ub):
        with T.SimdVF(latency=eval_vf_latency('bf16' if is_bf16_in else 'fp32', round_sf, hidden, 'scale',
                                              profile=vf_latency_profile)):
            mask64 = T.vmi.create_mask(64, size=64)
            if round_sf:
                clamp_min = T.vmi.vbrc(T.float32(out_config.clamp_min_value), size=64)
                exp_mask = T.vmi.vbrc(T.uint32(_EXP_MASK_F32), size=64)
                sf_inv_bias_u32 = T.vmi.vbrc(T.uint32(_F32_EXP_BIAS + quant_max_exp), size=64)
            else:
                quant_max_f32 = T.vmi.vbrc(T.float32(quant_max), size=64)
                if out_config.use_e4m3_sf:
                    clamp_min = T.vmi.vbrc(T.float32(out_config.clamp_min_value), size=64)
                    one_f32 = T.vmi.vbrc(T.float32(1.0), size=64)

            for sf_batch in T.serial(num_sf // 64):
                sf_offset = sf_batch * 64
                amax = T.vmi.vload(amax_ub[sf_offset], size=64)
                if round_sf or out_config.use_e4m3_sf:
                    clamped_amax = T.vmi.vmax(amax, clamp_min, mask64)
                else:
                    clamped_amax = amax
                if round_sf:
                    rounded_exp_u32 = T.vmi.vadds(
                        T.vmi.vinterpret_cast(clamped_amax, 'uint32'), round_carry, mask64
                    )
                    rounded_exp_u32_masked = T.vmi.vand(rounded_exp_u32, exp_mask, mask64)
                    sf_inv = T.vmi.vinterpret_cast(
                        T.vmi.vsub(sf_inv_bias_u32, rounded_exp_u32_masked, mask64), 'float32'
                    )
                    if is_bf16_in:
                        T.vmi.vstore(T.vmi.vcvt(sf_inv, 'bfloat16'), sf_inv_ub[sf_offset], mask64)
                    else:
                        T.vmi.vstore(sf_inv, sf_inv_ub[sf_offset], mask64)
                elif out_config.use_e4m3_sf:
                    # ASC: the bare S.vdiv is the precise expansion; precision
                    # 'ftz_true' is the hardware divide.  The E4M3-SF path uses
                    # ftz_true for the bf16-dequant scale and for the inverse.
                    ftz = use_bf16_dequant and out_config.clamp_min_value == quant_max * 2**-9
                    if ftz:
                        sf = T.vmi.vdiv(clamped_amax, quant_max_f32, mask64)
                    else:
                        sf = vdiv_with_fast_math(
                            clamped_amax, quant_max_f32, mask64, size=64, fast_math=FAST_MATH
                        )
                    sf_f32 = T.vmi.vcvt(
                        T.vmi.vcvt(
                            T.vmi.vmins(sf, T.float32(448.0), mask64),
                            'float8_e4m3fn', rounding='R', saturate='SAT',
                        ),
                        'float32',
                    )
                    if is_bf16_in:
                        T.vmi.vstore(T.vmi.vcvt(sf_f32, 'bfloat16'), sf_ub[sf_offset], mask64)
                    else:
                        T.vmi.vstore(sf_f32, sf_ub[sf_offset], mask64)
                    if is_bf16_in:
                        sf_inv = T.vmi.vdiv(one_f32, sf_f32, mask64)
                    else:
                        sf_inv = vdiv_with_fast_math(
                            one_f32, sf_f32, mask64, size=64, fast_math=FAST_MATH
                        )
                    T.vmi.vstore(sf_inv, sf_inv_ub[sf_offset], mask64)
                else:
                    # ASC S.vdiv is the CANN 0-ULP / FTZ-true division unless
                    # kernel fast-math is on; vdiv_with_fast_math mirrors that.
                    sf = vdiv_with_fast_math(
                        clamped_amax, quant_max_f32, mask64, size=64, fast_math=FAST_MATH
                    )
                    T.vmi.vstore(sf, sf_ub[sf_offset], mask64)
                    sf_inv = vdiv_with_fast_math(
                        quant_max_f32, clamped_amax, mask64, size=64, fast_math=FAST_MATH
                    )
                    T.vmi.vstore(sf_inv, sf_inv_ub[sf_offset], mask64)

    @T.macro
    def cast_and_cast_back_per16(x_ub, sf_ub, sf_inv_ub, out_ub):
        """Process each row with per-16 scaling in tiles of 128 elements."""
        with T.SimdVF(latency=eval_vf_latency('bf16' if is_bf16_in else 'fp32', round_sf, hidden, 'cast',
                                              profile=vf_latency_profile)):
            zero_bf16 = T.vmi.vbrc(T.bfloat16(0), size=128)
            for row in T.serial(block_m):
                for tile in T.serial(block_k // 128):
                    sf_offset = row * num_groups + tile * 8
                    if is_fp4_out:
                        base = tile * 128
                        if use_bf16_dequant:
                            mask64 = T.vmi.create_mask(64, size=64)
                            mask128 = T.vmi.create_mask(128, size=128)
                            sf_inv0 = T.vmi.vload(sf_inv_ub[sf_offset], size=64, stride=1,
                                                  dist_mode='brc', group=4)
                            sf_inv1 = T.vmi.vload(sf_inv_ub[sf_offset + 4], size=64, stride=1,
                                                  dist_mode='brc', group=4)
                            values0 = T.vmi.vcvt(T.vmi.vload(x_ub[row, base], size=64), 'float32')
                            values1 = T.vmi.vcvt(
                                T.vmi.vload(x_ub[row, base + 64], size=64), 'float32'
                            )
                            sf = T.vmi.vload(sf_ub[sf_offset], size=128, stride=1,
                                             dist_mode='brc', group=8)
                            T.vmi.vstore(
                                cast_and_cast_back_fp4_pair(
                                    T.vmi.vmul(values0, sf_inv0, mask64),
                                    T.vmi.vmul(values1, sf_inv1, mask64),
                                    sf, sf,
                                ),
                                out_ub[row, base], mask128,
                            )
                        else:
                            sf_inv0 = T.vmi.vload(sf_inv_ub[sf_offset], size=64, stride=1,
                                                  dist_mode='brc', group=4)
                            sf_inv1 = T.vmi.vload(sf_inv_ub[sf_offset + 4], size=64, stride=1,
                                                  dist_mode='brc', group=4)
                            sf0 = T.vmi.vload(sf_ub[sf_offset], size=64, stride=1,
                                              dist_mode='brc', group=4)
                            sf1 = T.vmi.vload(sf_ub[sf_offset + 4], size=64, stride=1,
                                              dist_mode='brc', group=4)
                            cast_and_cast_back_fp4_quad(
                                x_ub, out_ub, row, base, sf_inv0, sf_inv1, sf0, sf1
                            )

                    else:
                        # use_bf16_dequant (bf16 in) reads sf_ub as bf16 and folds
                        # ASC's E2B_B16 broadcast into the same grouped brc load.
                        sf = T.vmi.vload(sf_ub[sf_offset], size=128, stride=1,
                                         dist_mode='brc', group=8)
                        sf_inv = T.vmi.vload(sf_inv_ub[sf_offset], size=128,
                                             stride=1, dist_mode='brc', group=8)
                        cast_and_cast_back(
                            x_ub, out_ub, row, tile * 128, sf, sf_inv, zero_bf16,
                            128,
                        )

    @T.macro
    def cast_and_cast_back_per32(x_ub, sf_ub, sf_inv_ub, out_ub):
        """Process each row with per-32 scaling, including any 128-element tail."""
        with T.SimdVF(latency=eval_vf_latency('bf16' if is_bf16_in else 'fp32', round_sf, hidden, 'cast',
                                              profile=vf_latency_profile)):
            zero_bf16 = T.vmi.vbrc(T.bfloat16(0), size=VL)
            for row in T.serial(block_m):
                for tile in T.serial(block_k // 256):
                    sf_offset = row * num_groups + (tile * 256) // num_per_channels
                    if use_bf16_sf:
                        mask = T.vmi.create_mask(256, size=256)
                        values = T.vmi.vload(x_ub[row, (tile * 256)], size=256)
                        sf_inv = T.vmi.vload(
                            sf_inv_ub[sf_offset], size=256, stride=1, dist_mode='brc',
                            group=256 // MX_GROUP,
                        )
                        scaled = T.vmi.vmul(values, sf_inv, mask)
                        sf_inv_bias = T.vmi.vbrc(T.uint16(_BF16_EXP_BIAS), size=256)
                        sf = get_sf_from_inv(sf_inv, sf_inv_bias)
                        T.vmi.vstore(
                            cast_and_cast_back_bf16(scaled, sf, 256), out_ub[row, (tile * 256)], mask
                        )
                    elif is_fp4_out:
                        # FP4 tile: 64-lane chunks because VMI cannot reinterpret a
                        # 256-lane f32 vector to u16.
                        sf_inv_bias = T.vmi.vbrc(T.uint32(_F32_EXP_BIAS), size=64)
                        sf_invs = load_sf_fp4(sf_inv_ub, sf_offset)
                        if round_sf:
                            sfs = (
                                get_sf_from_inv(sf_invs[0], sf_inv_bias),
                                get_sf_from_inv(sf_invs[1], sf_inv_bias),
                                get_sf_from_inv(sf_invs[2], sf_inv_bias),
                                get_sf_from_inv(sf_invs[3], sf_inv_bias),
                            )
                        else:
                            sfs = load_sf_fp4(sf_ub, sf_offset)
                        cast_and_cast_back_fp4_quad(
                            x_ub, out_ub, row, (tile * 256), sf_invs[0], sf_invs[1], sfs[0], sfs[1]
                        )
                        if 256 == VL:
                            cast_and_cast_back_fp4_quad(
                                x_ub, out_ub, row, (tile * 256) + 128,
                                sf_invs[2], sf_invs[3], sfs[2], sfs[3],
                            )
                    elif is_bf16_in:
                        cast_and_cast_back_bf16_no_round(
                            x_ub, out_ub, row, (tile * 256),
                            load_sf_dintlv(sf_ub, sf_offset, 256),
                            load_sf_dintlv(sf_inv_ub, sf_offset, 256),
                            256,
                        )
                    elif round_sf:
                        sf_inv = load_sf(sf_inv_ub, sf_offset, 256)
                        sf_inv_bias = T.vmi.vbrc(T.uint32(_F32_EXP_BIAS), size=256)
                        sf = get_sf_from_inv(sf_inv, sf_inv_bias)
                        cast_and_cast_back(x_ub, out_ub, row, (tile * 256), sf, sf_inv, zero_bf16, 256)
                    else:
                        sf = load_sf(sf_ub, sf_offset, 256)
                        sf_inv = load_sf(sf_inv_ub, sf_offset, 256)
                        cast_and_cast_back(x_ub, out_ub, row, (tile * 256), sf, sf_inv, zero_bf16, 256)

                if block_k % 256:
                    sf_offset = row * num_groups + (block_k - 128) // num_per_channels
                    if use_bf16_sf:
                        mask = T.vmi.create_mask(128, size=128)
                        values = T.vmi.vload(x_ub[row, (block_k - 128)], size=128)
                        sf_inv = T.vmi.vload(
                            sf_inv_ub[sf_offset], size=128, stride=1, dist_mode='brc',
                            group=128 // MX_GROUP,
                        )
                        scaled = T.vmi.vmul(values, sf_inv, mask)
                        sf_inv_bias = T.vmi.vbrc(T.uint16(_BF16_EXP_BIAS), size=128)
                        sf = get_sf_from_inv(sf_inv, sf_inv_bias)
                        T.vmi.vstore(
                            cast_and_cast_back_bf16(scaled, sf, 128), out_ub[row, (block_k - 128)], mask
                        )
                    elif is_fp4_out:
                        # FP4 tile: 64-lane chunks because VMI cannot reinterpret a
                        # 256-lane f32 vector to u16.
                        sf_inv_bias = T.vmi.vbrc(T.uint32(_F32_EXP_BIAS), size=64)
                        sf_invs = load_sf_fp4(sf_inv_ub, sf_offset)
                        if round_sf:
                            sfs = (
                                get_sf_from_inv(sf_invs[0], sf_inv_bias),
                                get_sf_from_inv(sf_invs[1], sf_inv_bias),
                                get_sf_from_inv(sf_invs[2], sf_inv_bias),
                                get_sf_from_inv(sf_invs[3], sf_inv_bias),
                            )
                        else:
                            sfs = load_sf_fp4(sf_ub, sf_offset)
                        cast_and_cast_back_fp4_quad(
                            x_ub, out_ub, row, (block_k - 128), sf_invs[0], sf_invs[1], sfs[0], sfs[1]
                        )
                        if 128 == VL:
                            cast_and_cast_back_fp4_quad(
                                x_ub, out_ub, row, (block_k - 128) + 128,
                                sf_invs[2], sf_invs[3], sfs[2], sfs[3],
                            )
                    elif is_bf16_in:
                        cast_and_cast_back_bf16_no_round(
                            x_ub, out_ub, row, (block_k - 128),
                            load_sf_dintlv(sf_ub, sf_offset, 128),
                            load_sf_dintlv(sf_inv_ub, sf_offset, 128),
                            128,
                        )
                    elif round_sf:
                        sf_inv = load_sf(sf_inv_ub, sf_offset, 128)
                        sf_inv_bias = T.vmi.vbrc(T.uint32(_F32_EXP_BIAS), size=128)
                        sf = get_sf_from_inv(sf_inv, sf_inv_bias)
                        cast_and_cast_back(x_ub, out_ub, row, (block_k - 128), sf, sf_inv, zero_bf16, 128)
                    else:
                        sf = load_sf(sf_ub, sf_offset, 128)
                        sf_inv = load_sf(sf_inv_ub, sf_offset, 128)
                        cast_and_cast_back(x_ub, out_ub, row, (block_k - 128), sf, sf_inv, zero_bf16, 128)

    @T.prim_func
    def per_token_cast_and_cast_back_kernel(x: T.StridedTensor[(num_tokens, hidden), (token_stride, 1), dtype]):
        with T.Kernel(num_vec_cores) as core_id:
            x_ub = T.alloc_shared((block_m, block_k), dtype)
            amax_ub = T.alloc_shared((num_sf,), T.float32)
            sf_ub = T.alloc_shared((num_sf,), sf_dtype)
            sf_inv_ub = T.alloc_shared((num_sf,), sf_inv_dtype)
            out_ub = T.alloc_shared((block_m, block_k), dtype)
            versions = {x_ub: num_stages, amax_ub: num_stages, sf_inv_ub: num_stages, out_ub: num_stages}
            if not round_sf:
                versions[sf_ub] = num_stages
            T.annotate_buffer_versions(versions)

            for pid_token, pid_hidden in T.Persistent(
                [T.ceildiv(num_tokens, block_m), hidden // block_k],
                num_vec_cores,
                core_id,
                group_size=1,
                num_stages=num_stages,
                annotations={'enable_offset': True},
            ):
                T.copy(x[pid_token * block_m, pid_hidden * block_k], x_ub)
                collect_amax(x_ub, amax_ub)
                compute_sf(amax_ub, sf_ub, sf_inv_ub)
                if num_per_channels == 16:
                    cast_and_cast_back_per16(x_ub, sf_ub, sf_inv_ub, out_ub)
                else:
                    cast_and_cast_back_per32(x_ub, sf_ub, sf_inv_ub, out_ub)
                T.copy(out_ub, x[pid_token * block_m, pid_hidden * block_k])

    return per_token_cast_and_cast_back_kernel
