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
    cast_and_cast_back_per32 / compute_sf, with the logical packed-FP4 arm
    folded into cast_and_cast_back_f32_pair.  The per-stage tile helpers
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
packing is real, so it keeps vunzip/vor/vmins/vinterpret_cast (the sticky
round) on one 128-lane logical strip instead of ASC's two 64-lane registers.

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
    def load_sf_dintlv(sf_ub, sf_offset, lanes=VL, groups=None):
        """Load eight FP32 scales and broadcast each to 16 lanes."""
        return T.vmi.vload(
            sf_ub[sf_offset], size=lanes, stride=1, dist_mode='brc',
            group=lanes // MX_GROUP if groups is None else groups,
        )

    @T.macro
    def load_sf(sf_ub, sf_offset, lanes=VL, groups=None):
        """Load eight FP32 scales and broadcast each to 32 lanes."""
        return T.vmi.vload(
            sf_ub[sf_offset],
            size=lanes,
            stride=1,
            dist_mode='brc',
            group=lanes // MX_GROUP if groups is None else groups,
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
        """Compute scales and inverse scales from group maxima in the requested scale format."""
        with T.SimdVF(latency=eval_vf_latency('bf16' if is_bf16_in else 'fp32', round_sf, hidden, 'scale',
                                              profile=vf_latency_profile)):
            if round_sf:
                exp_mask = T.vmi.vbrc(T.uint32(_EXP_MASK_F32), size=64)
                sf_inv_bias_u32 = T.vmi.vbrc(T.uint32(_F32_EXP_BIAS + quant_max_exp), size=64)
            else:
                quant_max_f32 = T.vmi.vbrc(T.float32(quant_max), size=64)

            for sf_batch in T.serial(num_sf // 64):
                sf_offset = sf_batch * 64
                amax = T.vmi.vload(amax_ub[sf_offset], size=64)
                clamped_amax = T.vmi.vmaxs(amax, out_config.clamp_min_value)
                if round_sf:
                    rounded_amax_exp_u32 = T.vmi.vadds(T.vmi.vinterpret_cast(clamped_amax, 'uint32'), round_carry)
                    rounded_amax_exp_u32 = T.vmi.vand(rounded_amax_exp_u32, exp_mask)
                    sf_inv_bits_u32 = T.vmi.vsub(sf_inv_bias_u32, rounded_amax_exp_u32)
                    sf_inv = T.vmi.vinterpret_cast(sf_inv_bits_u32, 'float32')
                else:
                    ftz = use_bf16_dequant and out_config.clamp_min_value == quant_max * 2**-9
                    sf = vdiv_with_fast_math(clamped_amax, quant_max_f32, size=64, fast_math=ftz)
                    if out_config.use_e4m3_sf:
                        one_f32 = T.vmi.vbrc(T.float32(1.0), size=64)
                        sf_f32 = T.vmi.vcvt(T.vmi.vcvt(T.vmi.vmins(sf, T.float32(448.0)), 'float8_e4m3fn', rounding='R', saturate='SAT'), 'float32')
                        if is_bf16_in:
                            T.vmi.vstore(T.vmi.vcvt(sf_f32, 'bfloat16'), sf_ub[sf_offset])
                        else:
                            T.vmi.vstore(sf_f32, sf_ub[sf_offset])
                        sf_inv = vdiv_with_fast_math(one_f32, sf_f32, size=64, fast_math=is_bf16_in)
                    else:
                        T.vmi.vstore(sf, sf_ub[sf_offset])
                        sf_inv = vdiv_with_fast_math(
                            quant_max_f32, clamped_amax, size=64, fast_math=FAST_MATH
                        )
                if use_bf16_sf:
                    T.vmi.vstore(T.vmi.vcvt(sf_inv, 'bfloat16'), sf_inv_ub[sf_offset])
                else:
                    T.vmi.vstore(sf_inv, sf_inv_ub[sf_offset])

    @T.macro
    def clamp_e4m3(values):
        """Clamp normalized values to the finite E4M3 range."""
        # Rounding SF down can make normalized values exceed E4M3's finite range.
        return T.vmi.vmins(T.vmi.vmaxs(values, T.float32(-448.0)), T.float32(448.0))

    @T.macro
    def cast_and_cast_back_f32_pair(scaled, sf, mask=None):
        """Round-trip the logical f32 strip through e4m3/fp4 and rescale by sf."""
        if is_fp4_out:
            low, high = T.vmi.vunzip(scaled, 'uint16')
            scaled_bf16 = T.vmi.vinterpret_cast(
                T.vmi.vor(high, T.vmi.vmins(low, T.uint16(1), mask), mask),
                'bfloat16',
            )
            cast_back_bf16 = T.vmi.vcvt(T.vmi.vcvt(scaled_bf16, 'float4_e2m1fn', rounding='R'), 'bfloat16')
            if use_bf16_dequant:
                cast_back = cast_back_bf16
            else:
                cast_back = T.vmi.vcvt(cast_back_bf16, 'float32')
        else:
            cast_back = T.vmi.vcvt(
                T.vmi.vcvt(
                    clamp_e4m3(scaled) if out_config.use_e4m3_sf else scaled,
                    'float8_e4m3fn', rounding='R', saturate='SAT',
                ),
                'float32',
            )
            if use_bf16_dequant:
                cast_back = T.vmi.vcvt(cast_back, 'bfloat16')
        return T.vmi.vmul(cast_back, sf, mask)

    @T.macro
    def cast_and_cast_back_bf16(scaled, sf):
        """bf16 round_sf tile: bf16 -> e4m3/e2m1 -> bf16, rescaled by bf16 sf."""
        if is_fp4_out:
            quantized = T.vmi.vcvt(scaled, 'float4_e2m1fn', rounding='R')
            recovered = T.vmi.vcvt(quantized, 'bfloat16')
            return T.vmi.vmul(recovered, sf)
        else:
            scaled_f = T.vmi.vcvt(scaled, 'float32')
            quantized = T.vmi.vcvt(scaled_f, 'float8_e4m3fn', rounding='R', saturate='SAT')
            rescaled = T.vmi.vmul(T.vmi.vcvt(quantized, 'float32'), T.vmi.vcvt(sf, 'float32'))
        return T.vmi.vcvt(rescaled, 'bfloat16')

    @T.macro
    def cast_and_cast_back(x_ub, out_ub, row, col, sf, sf_inv, lanes=VL):
        """Load, scale, quantize, dequantize, and store 128 contiguous input values."""
        if is_bf16_in:
            values = T.vmi.vcvt(T.vmi.vload(x_ub[row, col], size=lanes), 'float32')
        else:
            values = T.vmi.vload(x_ub[row, col], size=lanes)
        scaled = T.vmi.vmul(values, sf_inv)
        output = cast_and_cast_back_f32_pair(scaled, sf)
        if use_bf16_dequant:
            T.vmi.vstore(output, out_ub[row, col])
        elif is_bf16_in:
            T.vmi.vstore(T.vmi.vcvt(output, 'bfloat16'), out_ub[row, col])
        else:
            T.vmi.vstore(output, out_ub[row, col])

    @T.macro
    def cast_and_cast_back_bf16_no_round(x_ub, out_ub, row, col, sf, sf_inv, lanes=VL):
        """Process 256 BF16 inputs with per-32 FP32 scales and deinterleaved loads."""
        values = T.vmi.vcvt(T.vmi.vload(x_ub[row, col], size=lanes), 'float32')
        scaled = T.vmi.vmul(values, sf_inv)
        output = cast_and_cast_back_f32_pair(scaled, sf)
        T.vmi.vstore(T.vmi.vcvt(output, 'bfloat16'), out_ub[row, col])

    @T.macro
    def cast_and_cast_back_per16(x_ub, sf_ub, sf_inv_ub, out_ub):
        """Process each row with per-16 scaling in tiles of 128 elements."""
        with T.SimdVF(latency=eval_vf_latency('bf16' if is_bf16_in else 'fp32', round_sf, hidden, 'cast',
                                              profile=vf_latency_profile)):
            for row in T.serial(block_m):
                for tile in T.serial(block_k // 128):
                    sf_offset = row * num_groups + tile * 8
                    if use_bf16_dequant:
                        sf = T.vmi.vload(sf_ub[sf_offset], size=128, stride=1,
                                         dist_mode='brc', group=8)
                    else:
                        sf = load_sf_dintlv(sf_ub, sf_offset, 128, groups=8)
                    sf_inv = load_sf_dintlv(sf_inv_ub, sf_offset, 128, groups=8)
                    cast_and_cast_back(x_ub, out_ub, row, tile * 128, sf, sf_inv,
                                       128)

    @T.macro
    def cast_and_cast_back_per32_tile(x_ub, sf_ub, sf_inv_ub, out_ub, row, col, tile_k, sf_inv_bias):
        """Quantize and dequantize one per-32 tile of 256 or 128 elements."""
        sf_offset = row * num_groups + col // num_per_channels
        if use_bf16_sf:
            mask = T.vmi.create_mask(tile_k, size=tile_k)
            values = T.vmi.vload(x_ub[row, col], size=tile_k)
            sf_inv = T.vmi.vload(
                sf_inv_ub[sf_offset], size=tile_k, stride=1, dist_mode='brc',
                group=tile_k // MX_GROUP,
            )
            scaled = T.vmi.vmul(values, sf_inv, mask)
            sf = get_sf_from_inv(sf_inv, sf_inv_bias)
            T.vmi.vstore(
                cast_and_cast_back_bf16(scaled, sf), out_ub[row, col], mask
            )
        elif is_fp4_out:
            sf_inv = T.vmi.vload(
                sf_inv_ub[sf_offset], size=tile_k, stride=1, dist_mode='brc',
                group=tile_k // MX_GROUP,
            )
            if round_sf:
                sf = get_sf_from_inv(sf_inv, sf_inv_bias)
            else:
                sf = T.vmi.vload(
                    sf_ub[sf_offset], size=tile_k, stride=1, dist_mode='brc',
                    group=tile_k // MX_GROUP,
                )
            cast_and_cast_back(
                x_ub, out_ub, row, col, sf, sf_inv,
                tile_k,
            )
        elif is_bf16_in:
            cast_and_cast_back_bf16_no_round(
                x_ub, out_ub, row, col,
                load_sf(sf_ub, sf_offset, tile_k),
                load_sf(sf_inv_ub, sf_offset, tile_k),
                tile_k,
            )
        elif round_sf:
            sf_inv = load_sf(sf_inv_ub, sf_offset, tile_k)
            sf = get_sf_from_inv(sf_inv, sf_inv_bias)
            cast_and_cast_back(x_ub, out_ub, row, col, sf, sf_inv, tile_k)
        else:
            sf = load_sf(sf_ub, sf_offset, tile_k)
            sf_inv = load_sf(sf_inv_ub, sf_offset, tile_k)
            cast_and_cast_back(x_ub, out_ub, row, col, sf, sf_inv, tile_k)

    @T.macro
    def cast_and_cast_back_per32(x_ub, sf_ub, sf_inv_ub, out_ub):
        """Process each row with per-32 scaling, including any 128-element tail."""
        with T.SimdVF(latency=eval_vf_latency('bf16' if is_bf16_in else 'fp32', round_sf, hidden, 'cast',
                                              profile=vf_latency_profile)):
            if use_bf16_sf or is_fp4_out or round_sf:
                sf_inv_bias_value = T.uint16(_BF16_EXP_BIAS) if use_bf16_sf else T.uint32(_F32_EXP_BIAS)
                if block_k // 256:
                    sf_inv_bias = T.vmi.vbrc(sf_inv_bias_value, size=VL)
                if block_k % 256:
                    sf_inv_bias_tail = T.vmi.vbrc(sf_inv_bias_value, size=128)
            else:
                sf_inv_bias = None
                sf_inv_bias_tail = None
            for row in T.serial(block_m):
                if block_k // 256:
                    for tile in T.serial(block_k // 256):
                        cast_and_cast_back_per32_tile(
                            x_ub, sf_ub, sf_inv_ub, out_ub, row, tile * 256, 256,
                            sf_inv_bias,
                        )
                if block_k % 256:
                    cast_and_cast_back_per32_tile(
                        x_ub, sf_ub, sf_inv_ub, out_ub, row, block_k - 128, 128,
                        sf_inv_bias_tail,
                    )

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
