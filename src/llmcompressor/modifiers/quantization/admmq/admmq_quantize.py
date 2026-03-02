# =====================================================================
# FILE: llmcompressor/modifiers/quantization/admmq/admmq_quantize.py
# =====================================================================
import time
from copy import copy
from typing import Optional, Tuple, Dict

import torch
import transformers
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationStrategy,
    fake_quantize,
)
from compressed_tensors.utils import update_offload_parameter
from loguru import logger

from llmcompressor.observers.base import Observer

ADMMQ_PRECISION = torch.float32

__all__ = ["quantize_weight_admmq"]


# -----------------------------------------------------------------------------
# Your ADMMQ core helpers (kept close to your version)
# -----------------------------------------------------------------------------

def get_quantize(
    matrix: torch.Tensor,
    maxq: int,
    sym: bool = True,
    per_column: bool = True,
    clip_ratio: torch.Tensor = None,
    groupsize: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert matrix.ndim == 2, "Input must be a 2D tensor"
    m, n = matrix.shape

    expand_groups = False
    if groupsize > 0 and per_column:
        assert m % groupsize == 0, f"m={m} not divisible by groupsize={groupsize}"
        ngroups = m // groupsize
        mat_g = matrix.reshape(ngroups, groupsize, n)
        xmin = mat_g.min(dim=1).values
        xmax = mat_g.max(dim=1).values
        expand_groups = True
    elif per_column:
        xmin = matrix.min(dim=0).values
        xmax = matrix.max(dim=0).values
    else:
        xmin = matrix.min()
        xmax = matrix.max()

    if clip_ratio is not None:
        xmin = clip_ratio * xmin
        xmax = clip_ratio * xmax

    if sym:
        xmax = torch.max(torch.abs(xmin), xmax).clamp(min=1e-5)
        scale = xmax / maxq
        zero = torch.zeros_like(scale)
    else:
        scale = (xmax - xmin).clamp(min=1e-8) / maxq
        zero = torch.round(-xmin / scale)

    if expand_groups:
        scale = scale.repeat_interleave(groupsize, dim=0)
        zero = zero.repeat_interleave(groupsize, dim=0)
    elif per_column:
        scale = scale.view(1, n)
        zero = zero.view(1, n)
    else:
        scale = scale.view(1, 1)
        zero = zero.view(1, 1)

    return scale, zero


def quantize_dequant(
    matrix: torch.Tensor,
    maxq: int,
    scale: torch.Tensor,
    zero: torch.Tensor,
) -> torch.Tensor:
    q = torch.clamp(torch.round(matrix / scale + zero), -(maxq + 1), maxq)
    return scale * (q - zero)


@torch.no_grad()
def ADMM_quantize(
    W: torch.Tensor,
    XtX: torch.Tensor,
    rho: float = 0.1,
    step_rho=lambda x: 1.1,
    maxq: int = 127,
    sym: bool = True,
    per_column: bool = True,
    max_iter: int = 300,
    update_iter: int = 3,
    switch_iter: int = 30,
    update_quant: bool = False,
    verbose: bool = False,
    use_norm: bool = True,
    clip_ratio=None,
    groupsize: int = -1,
) -> torch.Tensor:
    dev = W.device

    XtX = XtX.clone()
    diag = torch.arange(XtX.shape[0], device=dev)
    XtX[diag, diag] += 0.01 * torch.mean(torch.diag(XtX)).item()

    if use_norm:
        X_norm = (torch.diag(XtX).sqrt() + 1e-8).to(dev)
    else:
        X_norm = torch.ones((XtX.shape[0],), device=dev)

    XtX = XtX / X_norm
    XtX = (XtX.T / X_norm).T
    YtX = torch.matmul(W * X_norm, XtX)

    B = (W * X_norm).t().clone()  # (in, out)
    B_orig = B.clone()
    V = torch.zeros_like(B)

    L, Q = torch.linalg.eigh(XtX.double())
    XTX_inv = (Q @ ((1 / (L + rho)) * Q).T).float().to(dev)

    Res0 = torch.sum(B_orig * YtX.T)

    scale, zero = get_quantize(
        B / X_norm[:, None], maxq, sym, per_column, clip_ratio=clip_ratio, groupsize=groupsize
    )
    D = quantize_dequant(B / X_norm[:, None], maxq, scale, zero) * X_norm[:, None]

    D_prev = D.clone()
    D_base = torch.sum(D ** 2).clamp_min(1e-12)

    for i_admm in range(int(max_iter)):
        D_quant = (V + rho * B) / rho
        D = quantize_dequant(D_quant / X_norm[:, None], maxq, scale, zero) * X_norm[:, None]

        if update_quant:
            scale2, zero2 = get_quantize(
                D_quant / X_norm[:, None], maxq, sym, per_column, clip_ratio=clip_ratio, groupsize=groupsize
            )
            D_update = quantize_dequant(D_quant / X_norm[:, None], maxq, scale2, zero2) * X_norm[:, None]
            if torch.sum((D_quant - D_update) ** 2) < torch.sum((D_quant - D) ** 2):
                D = D_update
                scale, zero = scale2, zero2

        B = XTX_inv @ (YtX.T - V + rho * D)
        V = V + rho * (B - D)

        if (i_admm + 1) % int(update_iter) == 0:
            D_change = torch.sum((D - D_prev) ** 2)
            D_prev = D.clone()

            rho_update = step_rho((D_change / D_base).item())
            if rho_update > 0:
                rho *= rho_update
            else:
                break
            rho = min(float(rho), 1e6)

            XTX_inv = (Q @ ((1 / (L + rho)) * Q).T).float().to(dev)

            if verbose:
                Btest = quantize_dequant(B / X_norm[:, None], maxq, scale, zero) * X_norm[:, None]
                Resc = torch.matmul(XtX, Btest) - YtX.T
                Resc = torch.sum((Btest - B_orig) * Resc)
                errorc = (torch.sum(Resc) / Res0).item()
                logger.info(
                    "iter %d, error %.6f D %.6e, rho %.4e",
                    i_admm + 1,
                    errorc,
                    (D_change / D_base).item(),
                    rho,
                )

            if i_admm >= int(switch_iter) and (D_change / D_base) < 1e-6:
                break

    B = quantize_dequant(B / X_norm[:, None], maxq, scale, zero) * X_norm[:, None]
    return (B.t() / X_norm)


# -----------------------------------------------------------------------------
# llm-compressor-compatible wrapper (mirrors GPTQ.quantize_weight signature)
# -----------------------------------------------------------------------------

def _maxq_from_bits(num_bits: int) -> int:
    return (1 << (num_bits - 1)) - 1


def _export_group_qparams_from_broadcast(
    *,
    scale_bt: torch.Tensor,        # broadcastable to (in, out) (usually (in,out))
    zp_bt: torch.Tensor,
    out_features: int,
    in_features: int,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert ADMM's internal qparams (broadcast over W.T) into compressed-tensors
    group format:
      scale: (out, n_groups)
      zp:    (out, n_groups)
      g_idx: (in,)
    """
    if in_features % group_size != 0:
        raise ValueError(f"in_features={in_features} not divisible by group_size={group_size}")
    n_groups = in_features // group_size

    # scale_bt is either (in,out) (expanded) or (n_groups,out) expanded; ensure expanded:
    if scale_bt.shape[0] != in_features:
        # expect it is (n_groups,out) -> expand
        scale_bt = scale_bt.repeat_interleave(group_size, dim=0)
        zp_bt = zp_bt.repeat_interleave(group_size, dim=0)

    scale_g = scale_bt[::group_size, :].reshape(n_groups, out_features).t().contiguous()
    zp_g = zp_bt[::group_size, :].reshape(n_groups, out_features).t().contiguous()
    g_idx = (torch.arange(in_features, device=scale_bt.device, dtype=torch.int) // group_size)
    return scale_g, zp_g, g_idx


@torch.no_grad()
def quantize_weight_admmq(
    module: torch.nn.Module,
    quant_args: QuantizationArgs,
    xtx: torch.Tensor,
    rho: float = 0.1,
    max_iter: int = 300,
    update_iter: int = 3,
    switch_iter: int = 30,
    update_quant: bool = False,
    clip_ratio: Optional[float] = None,
) -> tuple[float, Dict[str, torch.Tensor]]:
    """
    Quantize a module weight using ADMMQ.

    Returns: (loss, q_param_dict) where q_param_dict contains:
      weight, weight_scale, weight_zero_point, optionally weight_g_idx
    """
    if quant_args.type != "int":
        raise ValueError("ADMMQ only supports integer weight quantization.")
    if not quant_args.symmetric:
        raise ValueError("ADMMQ implementation assumes symmetric quantization.")
    if quant_args.strategy not in (
        QuantizationStrategy.TENSOR,
        QuantizationStrategy.CHANNEL,
        QuantizationStrategy.GROUP,
        QuantizationStrategy.TENSOR_GROUP,
    ):
        raise ValueError(f"Quantization strategy not supported for ADMMQ: {quant_args.strategy}")

    strategy = quant_args.strategy
    final_shape = module.weight.shape
    final_dtype = module.weight.dtype
    W = module.weight.clone()

    # Observer is used ONLY to ensure output qparam shapes/dtypes match llm-compressor conventions
    observer = Observer.load_from_registry(
        quant_args.observer if quant_args.observer else "memoryless_minmax",
        base_name="weight",
        args=quant_args,
        module=module,
    )

    # Standardize shape
    match module:
        case torch.nn.Conv2d():
            W = W.flatten(1)
        case transformers.Conv1D():
            W.transpose_(0, 1)

    W = W.to(dtype=ADMMQ_PRECISION)
    out_features, in_features = W.shape

    if xtx.shape != (in_features, in_features):
        raise ValueError(f"xtx must be (in,in)=({in_features},{in_features}), got {tuple(xtx.shape)}")

    # Setup ADMMQ quant params style:
    # Your get_quantize expects matrix shaped (m,n) where per_column means per "n",
    # and groupsize groups along m. We run it on W.T = (in,out), so:
    # - per_column=True gives per-out-channel scales (CHANNEL in llm-compressor terms)
    # - groupsize=group_size groups along in_features => GROUP in llm-compressor terms
    per_column = (strategy != QuantizationStrategy.TENSOR)
    groupsize = -1
    if strategy in (QuantizationStrategy.GROUP, QuantizationStrategy.TENSOR_GROUP):
        groupsize = int(quant_args.group_size)

    maxq = _maxq_from_bits(int(quant_args.num_bits))

    # clip_ratio: scalar -> tensor broadcast
    clip_t = None
    if clip_ratio is not None:
        clip_t = torch.tensor(float(clip_ratio), device=W.device, dtype=W.dtype)

    t0 = time.time()
    Wq = ADMM_quantize(
        W=W,
        XtX=xtx.to(dtype=ADMMQ_PRECISION),
        rho=float(rho),
        maxq=maxq,
        sym=True,
        per_column=per_column,
        max_iter=int(max_iter),
        update_iter=int(update_iter),
        switch_iter=int(switch_iter),
        update_quant=bool(update_quant),
        verbose=False,
        use_norm=True,
        clip_ratio=clip_t,
        groupsize=groupsize,
    )  # (out,in) fp32

    # Loss (simple MSE; GPTQ uses a Hessian-weighted internal objective; keep simple for logging)
    loss = torch.sum((W - Wq) ** 2).item()

    # Export qparams in llm-compressor expected shapes:
    # For tensor/channel/group schemes, Observer already returns correct-format qparams for *Wq*.
    # That makes the stored checkpoint maximally compatible with the ecosystem.
    if strategy in (QuantizationStrategy.GROUP, QuantizationStrategy.TENSOR_GROUP):
        # GPTQ’s group path uses Observer(W) -> scale:(out,n_groups), zp:(out,n_groups) and g_idx mapping.
        # We replicate g_idx identity (no actorder permutations).
        scale, zero_point = observer(Wq)
        g_idx = (torch.arange(in_features, device=W.device, dtype=torch.int) // int(quant_args.group_size))
    else:
        scale, zero_point = observer(Wq)
        g_idx = None

    # Restore module-native layout
    if isinstance(module, transformers.Conv1D):
        Wq.transpose_(0, 1)
    Wq = Wq.reshape(final_shape).to(final_dtype)

    q_param_dict: Dict[str, torch.Tensor] = {
        "weight": Wq,
        "weight_scale": scale.to(dtype=final_dtype),
        "weight_zero_point": zero_point.to(dtype=quant_args.zp_dtype),
    }
    if g_idx is not None:
        q_param_dict["weight_g_idx"] = g_idx

    logger.info(f"ADMMQ time {time.time() - t0:.2f}s mse={loss:.4e} shape={tuple(final_shape)}")
    return loss, q_param_dict
