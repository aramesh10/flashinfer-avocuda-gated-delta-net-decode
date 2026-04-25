"""
Triton kernel for Gated Delta Net decode (k-last layout).

Definition: gdn_decode_qk4_v8_d128_k_last
  - num_q_heads=4, num_k_heads=4, num_v_heads=8, D=128
  - GVA ratio: each q/k head (hv//2) serves 2 v-heads
  - State layout: [B, Hv, D, D] where state[b,h] is [V=D, K=D] (k-last)
  - T=1 decode

DPS: output tensors pre-allocated, written in-place.
  kernel(q, k, v, state, A_log, a, dt_bias, b, scale, output, new_state)

Math per (b, hv), row vi of state:
  g        = exp(-exp(A_log[hv]) * softplus(a[b,hv] + dt_bias[hv]))
  beta     = sigmoid(b[b,hv])
  S_vi     = state[b, hv, vi, :]          # [K=D] row of state
  old_v_vi = dot(k, g * S_vi)            # scalar
  delta_vi = beta * (v[b, hv, vi] - old_v_vi)   # scalar
  S_new_vi = g * S_vi + k * delta_vi     # [K=D]
  out[vi]  = scale * dot(q, S_new_vi)    # scalar

Grid: (B, Hv=8, D=128) -- one program per (batch, v-head, state-row vi)
"""

import math
import torch
import triton
import triton.language as tl

NUM_Q_HEADS = 4
NUM_V_HEADS = 8
GVA_RATIO   = NUM_V_HEADS // NUM_Q_HEADS  # 2
HEAD_DIM    = 128


@triton.jit
def _gdn_decode_kernel(
    # Inputs (q/k/v pre-squeezed to remove T=1 dim)
    q_ptr,        # [B, Hq, D]      bf16
    k_ptr,        # [B, Hq, D]      bf16
    v_ptr,        # [B, Hv, D]      bf16
    state_ptr,    # [B, Hv, D, D]   f32   [V-row, K-col]
    A_log_ptr,    # [Hv]            f32
    a_ptr,        # [B, Hv]         f32
    dt_bias_ptr,  # [Hv]            f32
    b_ptr,        # [B, Hv]         f32
    scale,        # scalar          f32
    # Outputs (output keeps T=1 dim; new_state same shape as state)
    out_ptr,      # [B, Hv, D]      f32  (squeezed, cast to bf16 at store)
    ns_ptr,       # [B, Hv, D, D]   f32
    # Compile-time constants
    Hv:  tl.constexpr,
    Hq:  tl.constexpr,
    GVA: tl.constexpr,
    D:   tl.constexpr,
):
    b_idx  = tl.program_id(0)
    hv_idx = tl.program_id(1)
    vi     = tl.program_id(2)  # V-dimension row index into state

    hqk_idx = hv_idx // GVA

    # Gate scalars
    A_log = tl.load(A_log_ptr + hv_idx)
    a_val = tl.load(a_ptr     + b_idx * Hv + hv_idx)
    dt    = tl.load(dt_bias_ptr + hv_idx)
    b_val = tl.load(b_ptr     + b_idx * Hv + hv_idx)

    x    = a_val + dt
    sp   = tl.log(1.0 + tl.exp(x))          # softplus(x)
    g    = tl.exp(-tl.exp(A_log) * sp)       # gate scalar
    beta = tl.sigmoid(b_val)                 # beta scalar

    # Load q, k vectors [D] and v scalar at position vi
    d_offs = tl.arange(0, D)

    q_vec = tl.load(q_ptr + b_idx * Hq * D + hqk_idx * D + d_offs)  # [D] bf16
    k_vec = tl.load(k_ptr + b_idx * Hq * D + hqk_idx * D + d_offs)  # [D] bf16
    v_vi  = tl.load(v_ptr + b_idx * Hv * D + hv_idx  * D + vi)      # scalar bf16

    # Promote to f32
    q_vec = q_vec.to(tl.float32)
    k_vec = k_vec.to(tl.float32)
    v_vi  = v_vi.to(tl.float32)

    # Load state row S[vi, :] -- shape [D] in K-dimension
    s_base   = b_idx * Hv * D * D + hv_idx * D * D + vi * D
    s_row    = tl.load(state_ptr + s_base + d_offs)  # [D] f32

    # Compute gated state row and old_v for this vi
    gs_row   = g * s_row                          # [D]
    old_v_vi = tl.sum(k_vec * gs_row)             # scalar

    # Delta update
    delta_vi  = beta * (v_vi - old_v_vi)          # scalar
    s_new_row = gs_row + k_vec * delta_vi          # [D]

    # Write updated state row
    tl.store(ns_ptr + s_base + d_offs, s_new_row)

    # Output scalar for this vi
    out_vi = scale * tl.sum(q_vec * s_new_row)    # scalar

    # Store into out_ptr[b, hv, vi]  (squeezed, no T dim)
    tl.store(out_ptr + b_idx * Hv * D + hv_idx * D + vi,
             out_vi.to(tl.bfloat16))


def kernel(q, k, v, state, A_log, a, dt_bias, b, scale, output, new_state):
    """
    Gated Delta Net decode (DPS).

    Inputs:
      q         [B, 1, 4, 128]    bf16
      k         [B, 1, 4, 128]    bf16
      v         [B, 1, 8, 128]    bf16
      state     [B, 8, 128, 128]  f32
      A_log     [8]               f32
      a         [B, 1, 8]         bf16
      dt_bias   [8]               f32
      b         [B, 1, 8]         bf16
      scale     scalar            f32
    Outputs (pre-allocated, written in-place):
      output    [B, 1, 8, 128]   bf16
      new_state [B, 8, 128, 128] f32
    """
    B  = q.shape[0]
    Hv = NUM_V_HEADS   # 8
    Hq = NUM_Q_HEADS   # 4
    D  = HEAD_DIM      # 128

    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(D)

    # Squeeze T=1 dim, ensure contiguous
    q_sq  = q.squeeze(1).contiguous()                        # [B, Hq, D] bf16
    k_sq  = k.squeeze(1).contiguous()                        # [B, Hq, D] bf16
    v_sq  = v.squeeze(1).contiguous()                        # [B, Hv, D] bf16
    a_f32 = a.squeeze(1).to(torch.float32).contiguous()      # [B, Hv]    f32
    b_f32 = b.squeeze(1).to(torch.float32).contiguous()      # [B, Hv]    f32

    # output is [B, 1, Hv, D] -- use a squeezed view for the kernel to write into
    out_sq = output.squeeze(1)   # [B, Hv, D] bf16, shares storage

    grid = (B, Hv, D)
    _gdn_decode_kernel[grid](
        q_sq, k_sq, v_sq, state,
        A_log, a_f32, dt_bias, b_f32,
        scale,
        out_sq, new_state,
        Hv=Hv, Hq=Hq, GVA=GVA_RATIO, D=D,
    )
