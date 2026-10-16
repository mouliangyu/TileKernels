import tilelang
from tilelang.ascend import language as T
from tilelang.transform import PassConfigKey

from tile_kernels.config import get_max_ub_per_vector_core
from tile_kernels.quant.common import CastOutputConfig
from tile_kernels.utils import align

# BF16 bit-field constants for power-of-two round_sf (ASC values).
_ABS_MASK_BF16 = 0x7FFF
_BF16_EXP_BIAS = 0x7F00
_EXP_MASK_F32 = 0x7F800000
_F32_EXP_BIAS = 0x7F000000

# MXFP8 per-group scale geometry and the PTOAS full-vector strip length,
# inlined from the former tile_kernels.quant.vmi_common (PR #3 review).
MX_GROUP = 32
VL = 256

# Match TileLang ASC's default for a bare S.vdiv: precise FP32 division unless a
# caller explicitly enables fast math for the whole kernel.
FAST_MATH = False
PTO_FAST_MATH = {PassConfigKey.TL_ENABLE_FAST_MATH: FAST_MATH}


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
):
    """ASC-shaped VMI entry; mirrors get_per_token_cast_and_cast_back_kernel_asc.

    The only structural differences from the ASC source of truth are the jit
    decorator's target/pass_configs and the logical T.vmi.* bodies inside the
    four T.SimdVF regions.
    """
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
        group = lanes // MX_GROUP if groups is None else groups
        return T.vmi.vbrc(T.vmi.vload(sf_ub[sf_offset], size=group, group=group, stride=1), size=lanes, group=group)

    @T.macro
    def load_sf(sf_ub, sf_offset, lanes=VL, groups=None):
        """Load eight FP32 scales and broadcast each to 32 lanes."""
        group = lanes // MX_GROUP if groups is None else groups
        return T.vmi.vbrc(T.vmi.vload(sf_ub[sf_offset], size=group, group=group, stride=1), size=lanes, group=group)

    @T.macro
    def collect_amax_tile(x_ub, amax_ub, row, col, tile_k):
        """Compute and store the maximum absolute value of each scale group in one tile."""
        sf_offset = row * num_groups + col // num_per_channels
        groups = tile_k // num_per_channels
        if is_bf16_in:
            abs_mask_u16 = T.vmi.vbrc(T.uint16(_ABS_MASK_BF16), size=tile_k)
            values = T.vmi.vload(x_ub[row, col], size=tile_k)
            abs_u16 = T.vmi.vand(T.vmi.vinterpret_cast(values, 'uint16'), abs_mask_u16)
            amax_u16 = T.vmi.vcmax(abs_u16, group=groups)
            amax_u32 = T.vmi.vcvt(amax_u16, 'uint32')
            amax = T.vmi.vinterpret_cast(T.vmi.vshls(amax_u32, 16), 'float32')
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
        with T.SimdVF():
            tile_k = num_per_channels * 8
            for row in T.serial(block_m):
                for tile in T.serial(block_k // tile_k):
                    collect_amax_tile(x_ub, amax_ub, row, tile * tile_k, tile_k)
                if block_k % tile_k:
                    collect_amax_tile(x_ub, amax_ub, row, block_k - 128, 128)

    @T.macro
    def compute_sf(amax_ub, sf_ub, sf_inv_ub):
        """Compute scales and inverse scales from group maxima in the requested scale format."""
        with T.SimdVF():
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
                    if ftz:
                        sf = T.vmi.vdiv(clamped_amax, quant_max_f32)
                    else:
                        sf = T.vmi.vdiv(
                            clamped_amax,
                            quant_max_f32,
                            T.vmi.create_mask(64, size=64),
                            precision='exact',
                        )
                    if out_config.use_e4m3_sf:
                        one_f32 = T.vmi.vbrc(T.float32(1.0), size=64)
                        sf_f32 = T.vmi.vcvt(T.vmi.vcvt(T.vmi.vmins(sf, T.float32(448.0)), 'float8_e4m3fn', rounding='R', saturate='SAT'), 'float32')
                        if is_bf16_in:
                            T.vmi.vstore(T.vmi.vcvt(sf_f32, 'bfloat16'), sf_ub[sf_offset])
                            sf_inv = T.vmi.vdiv(one_f32, sf_f32)
                        else:
                            T.vmi.vstore(sf_f32, sf_ub[sf_offset])
                            sf_inv = T.vmi.vdiv(one_f32, sf_f32, T.vmi.create_mask(64, size=64), precision='exact')
                    else:
                        T.vmi.vstore(sf, sf_ub[sf_offset])
                        sf_inv = T.vmi.vdiv(quant_max_f32, clamped_amax, T.vmi.create_mask(64, size=64), precision='exact')
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
            scaled_bf16 = T.vmi.vinterpret_cast(T.vmi.vor(high, T.vmi.vmins(low, T.uint16(1), mask), mask), 'bfloat16')
            cast_back_bf16 = T.vmi.vcvt(T.vmi.vcvt(scaled_bf16, 'float4_e2m1fn', rounding='R'), 'bfloat16')
            if use_bf16_dequant:
                cast_back = cast_back_bf16
            else:
                cast_back = T.vmi.vcvt(cast_back_bf16, 'float32')
        else:
            cast_back = T.vmi.vcvt(T.vmi.vcvt(clamp_e4m3(scaled) if out_config.use_e4m3_sf else scaled, 'float8_e4m3fn', rounding='R', saturate='SAT'), 'float32')
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
        with T.SimdVF():
            for row in T.serial(block_m):
                for tile in T.serial(block_k // 128):
                    sf_offset = row * num_groups + tile * 8
                    if use_bf16_dequant:
                        sf = T.vmi.vbrc(T.vmi.vload(sf_ub[sf_offset], size=8, group=8, stride=1), size=128, group=8)
                    else:
                        sf = load_sf_dintlv(sf_ub, sf_offset, 128, groups=8)
                    sf_inv = load_sf_dintlv(sf_inv_ub, sf_offset, 128, groups=8)
                    cast_and_cast_back(x_ub, out_ub, row, tile * 128, sf, sf_inv, 128)

    @T.macro
    def cast_and_cast_back_per32_tile(x_ub, sf_ub, sf_inv_ub, out_ub, row, col, tile_k, sf_inv_bias):
        """Quantize and dequantize one per-32 tile of 256 or 128 elements."""
        sf_offset = row * num_groups + col // 32
        if use_bf16_sf and tile_k == 256:
            values = T.vmi.vload(x_ub[row, col], size=tile_k)
            sf_inv = T.vmi.vbrc(T.vmi.vload(sf_inv_ub[sf_offset], size=tile_k // MX_GROUP, group=tile_k // MX_GROUP, stride=1), size=tile_k, group=tile_k // MX_GROUP)
            scaled = T.vmi.vmul(values, sf_inv)
            sf = get_sf_from_inv(sf_inv, sf_inv_bias)
            T.vmi.vstore(cast_and_cast_back_bf16(scaled, sf), out_ub[row, col])
        elif use_bf16_sf:
            values = T.vmi.vload(x_ub[row, col], size=tile_k)
            sf_inv = T.vmi.vbrc(T.vmi.vload(sf_inv_ub[sf_offset], size=tile_k // MX_GROUP, group=tile_k // MX_GROUP, stride=1), size=tile_k, group=tile_k // MX_GROUP)
            scaled = T.vmi.vmul(values, sf_inv)
            sf = get_sf_from_inv(sf_inv, sf_inv_bias)
            T.vmi.vstore(cast_and_cast_back_bf16(scaled, sf), out_ub[row, col])
        elif is_bf16_in and tile_k == 256:
            cast_and_cast_back_bf16_no_round(x_ub, out_ub, row, col, load_sf(sf_ub, sf_offset, tile_k), load_sf(sf_inv_ub, sf_offset, tile_k), tile_k)
        else:
            if round_sf:
                sf_inv = load_sf(sf_inv_ub, sf_offset, tile_k)
                sf = get_sf_from_inv(sf_inv, sf_inv_bias)
            else:
                sf = load_sf(sf_ub, sf_offset, tile_k)
                sf_inv = load_sf(sf_inv_ub, sf_offset, tile_k)
            cast_and_cast_back(x_ub, out_ub, row, col, sf, sf_inv, tile_k)
    @T.macro
    def cast_and_cast_back_per32(x_ub, sf_ub, sf_inv_ub, out_ub):
        """Process each row with per-32 scaling, including any 128-element tail."""
        with T.SimdVF():
            bias_u16 = T.uint16(_BF16_EXP_BIAS)
            sf_inv_bias_u32 = T.uint32(_F32_EXP_BIAS)
            sf_inv_bias_value = bias_u16 if use_bf16_sf else sf_inv_bias_u32
            sf_inv_bias = T.vmi.vbrc(sf_inv_bias_value, size=VL)
            sf_inv_bias_tail = T.vmi.vbrc(sf_inv_bias_value, size=128)
            for row in T.serial(block_m):
                for tile in T.serial(block_k // 256):
                    cast_and_cast_back_per32_tile(x_ub, sf_ub, sf_inv_ub, out_ub, row, tile * 256, 256, sf_inv_bias)
                if block_k % 256:
                    cast_and_cast_back_per32_tile(x_ub, sf_ub, sf_inv_ub, out_ub, row, block_k - 128, 128, sf_inv_bias_tail)

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
