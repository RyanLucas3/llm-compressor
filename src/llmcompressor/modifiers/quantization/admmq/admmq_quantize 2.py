# =====================================================================
# FILE: llmcompressor/modifiers/quantization/admmq/admmq_quantize.py
# =====================================================================
import math
import time
from typing import Optional, Dict, Tuple

import torch
import transformers
from compressed_tensors.quantization import QuantizationArgs, QuantizationStrategy
from loguru import logger

from llmcompressor.observers.base import Observer

ADMMQ_PRECISION = torch.float32

__all__ = ["quantize_weight_admmq", "accumulate_xtx_admm"]


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


def find_clip_ratio(
    matrix: torch.Tensor,
    maxq: int,
    sym: bool = True,
    per_column: bool = True,
    grid: int = 100,
    maxshrink: float = 0.8,
    norm: float = 2.4,
    groupsize: int = -1,
) -> torch.Tensor:
    """
    MSE grid search for per-column (or per-group) clipping ratio.
    Copied from your original solvers.py with minimal changes.
    """
    assert matrix.ndim == 2, "Input must be a 2D tensor"
    m, n = matrix.shape

    if groupsize > 0 and per_column:
        assert m % groupsize == 0, f"m={m} not divisible by groupsize={groupsize}"
        ngroups = m // groupsize
        mat_g = matrix.reshape(ngroups, groupsize, n)
        xmin = mat_g.min(dim=1).values
        xmax = mat_g.max(dim=1).values

        if sym:
            _ = torch.max(torch.abs(xmin), xmax).clamp(min=1e-5)

        best = torch.full((ngroups, n), float("inf"), device=matrix.device, dtype=matrix.dtype)
        best_p = torch.ones((ngroups, n), device=matrix.device, dtype=matrix.dtype)

        for i in range(int(maxshrink * grid)):
            p = 1 - i / grid
            xmin1 = p * xmin
            xmax1 = p * xmax
            if sym:
                scale1 = torch.max(torch.abs(xmin1), xmax1).clamp(min=1e-5) / maxq
                zero1 = torch.zeros_like(scale1)
            else:
                scale1 = (xmax1 - xmin1).clamp(min=1e-8) / maxq
                zero1 = torch.round(-xmin1 / scale1)

            s1 = scale1.repeat_interleave(groupsize, dim=0)
            z1 = zero1.repeat_interleave(groupsize, dim=0)
            q = torch.clamp(torch.round(matrix / s1 + z1), -(maxq + 1), maxq)
            deq = s1 * (q - z1)
            err = (matrix - deq).abs().pow(norm).reshape(ngroups, groupsize, n).sum(dim=1)

            tmp = err < best
            if torch.any(tmp):
                best[tmp] = err[tmp]
                best_p[tmp] = p

        return best_p

    if per_column:
        xmin = matrix.min(dim=0).values
        xmax = matrix.max(dim=0).values
    else:
        xmin = matrix.min()
        xmax = matrix.max()

    if sym:
        _ = torch.max(torch.abs(xmin), xmax).clamp(min=1e-5)

    best = torch.full_like(xmin if per_column else xmin.unsqueeze(0), float("inf"))
    best_p = torch.ones_like(best)

    for i in range(int(maxshrink * grid)):
        p = 1 - i / grid
        xmin1 = p * xmin
        xmax1 = p * xmax
        if sym:
            scale1 = torch.max(torch.abs(xmin1), xmax1).clamp(min=1e-5) / maxq
            zero1 = torch.zeros_like(scale1)
        else:
            scale1 = (xmax1 - xmin1).clamp(min=1e-8) / maxq
            zero1 = torch.round(-xmin1 / scale1)

        if per_column:
            s1 = scale1.view(1, n)
            z1 = zero1.view(1, n)
        else:
            s1 = scale1.view(1, 1)
            z1 = zero1.view(1, 1)

        q = torch.clamp(torch.round(matrix / s1 + z1), -(maxq + 1), maxq)
        deq = s1 * (q - z1)
        err = (matrix - deq).abs().pow(norm).sum(dim=0)

        tmp = err < best
        if torch.any(tmp):
            best[tmp] = err[tmp]
            best_p[tmp] = p

    return best_p


# -----------------------------------------------------------------------------
# ADMM XtX accumulator (parity with your original pipeline)
# -----------------------------------------------------------------------------

@torch.no_grad()
def accumulate_xtx_admm(
    inp: torch.Tensor,
    module: torch.nn.Module,
    H: torch.Tensor,
    num_samples: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Accumulate raw XtX = X^T X (sum), with num_samples counting the number
    of vectors contributing to XtX (tokens/patches), not batches.
    Also: no sqrt(2) scaling (that was GPTQ-specific in llm-compressor).
    """
    inp = inp.to(device=H.device)
    if inp.ndim == 2:
        inp = inp.unsqueeze(0)

    match module:
        case torch.nn.Linear() | transformers.Conv1D():
            # inp: (B,S,in) or (B,in)
            if inp.ndim == 3:
                inp = inp.reshape((-1, inp.shape[-1]))  # (B*S, in)
            inp = inp.t()  # (in, N)
            num_added = inp.shape[1]

        case torch.nn.Conv2d():
            unfold = torch.nn.Unfold(
                module.kernel_size,
                dilation=module.dilation,
                padding=module.padding,
                stride=module.stride,
            )
            x = unfold(inp)                    # (B, in*kH*kW, L)
            x = x.permute(1, 0, 2).reshape(x.shape[1], -1)  # (in*kH*kW, B*L)
            inp = x
            num_added = inp.shape[1]

        case _:
            raise TypeError(f"Unsupported module type for XtX accumulation: {type(module)}")

    num_samples += num_added
    inp = inp.to(dtype=ADMMQ_PRECISION)
    H += inp.matmul(inp.t())
    return H, num_samples


# -----------------------------------------------------------------------------
# ADMM solver (minor parity tweaks vs your original)
# -----------------------------------------------------------------------------

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
        B / X_norm[:, None],
        maxq,
        sym,
        per_column,
        clip_ratio=clip_ratio,
        groupsize=groupsize,
    )
    D = quantize_dequant(B / X_norm[:, None], maxq, scale, zero) * X_norm[:, None]

    D_prev = D.clone()
    D_base = torch.sum(D ** 2)  # parity with your original (no clamp)

    for i_admm in range(int(max_iter)):
        D_quant = (V + rho * B) / rho
        D = quantize_dequant(D_quant / X_norm[:, None], maxq, scale, zero) * X_norm[:, None]

        if update_quant:
            scale2, zero2 = get_quantize(
                D_quant / X_norm[:, None],
                maxq,
                sym,
                per_column,
                clip_ratio=clip_ratio,
                groupsize=groupsize,
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

            # parity: step_rho called on ratio (float)
            ratio = float((D_change / D_base).item()) if D_base.item() != 0 else 0.0
            rho_update = step_rho(ratio)
            rho_update = float(rho_update)

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
                    float((D_change / D_base).item()) if D_base.item() != 0 else 0.0,
                    rho,
                )

            if i_admm >= int(switch_iter):
                if D_base.item() != 0 and (D_change / D_base) < 1e-6:
                    break

    B = quantize_dequant(B / X_norm[:, None], maxq, scale, zero) * X_norm[:, None]
    return (B.t() / X_norm)


# -----------------------------------------------------------------------------
# Export helpers
# -----------------------------------------------------------------------------

def _maxq_from_bits(num_bits: int) -> int:
    return (1 << (num_bits - 1)) - 1


def _export_group_qparams_from_broadcast(
    *,
    scale_bt: torch.Tensor,        # broadcastable to (in, out)
    zp_bt: torch.Tensor,
    out_features: int,
    in_features: int,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert broadcast qparams (shape (in,out) after expansion) into
    compressed-tensors group format:
      scale: (out, n_groups)
      zp:    (out, n_groups)
      g_idx: (in,)
    """
    if in_features % group_size != 0:
        raise ValueError(f"in_features={in_features} not divisible by group_size={group_size}")
    n_groups = in_features // group_size

    if scale_bt.shape[0] != in_features:
        scale_bt = scale_bt.repeat_interleave(group_size, dim=0)
        zp_bt = zp_bt.repeat_interleave(group_size, dim=0)

    scale_g = scale_bt[::group_size, :].reshape(n_groups, out_features).t().contiguous()
    zp_g = zp_bt[::group_size, :].reshape(n_groups, out_features).t().contiguous()
    g_idx = (torch.arange(in_features, device=scale_bt.device, dtype=torch.int) // group_size)
    return scale_g, zp_g, g_idx


# -----------------------------------------------------------------------------
# llm-compressor-compatible wrapper
# -----------------------------------------------------------------------------

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
    ADMMQ weight quantization with export parity to your original pipeline + vLLM packing.

    Behavior:
      - per_column=True always (parity with your original)
      - if clip_ratio is None: run MSE grid-search like your original find_clip_ratio
      - for GROUP/TENSOR_GROUP: export qparams from the same clipped min/max quantizer,
        and snap exported weight to the dequant implied by those qparams.
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

    observer = Observer.load_from_registry(
        quant_args.observer if quant_args.observer else "memoryless_minmax",
        base_name="weight",
        args=quant_args,
        module=module,
    )

    # Standardize shape to (out, in)
    match module:
        case torch.nn.Conv2d():
            W = W.flatten(1)
        case transformers.Conv1D():
            W.transpose_(0, 1)

    W = W.to(dtype=ADMMQ_PRECISION)
    out_features, in_features = W.shape

    if xtx.shape != (in_features, in_features):
        raise ValueError(f"xtx must be (in,in)=({in_features},{in_features}), got {tuple(xtx.shape)}")

    # Parity: per_column always True for ADMMQ (like your original pipeline)
    per_column = True

    groupsize = -1
    if strategy in (QuantizationStrategy.GROUP, QuantizationStrategy.TENSOR_GROUP):
        groupsize = int(quant_args.group_size)

    maxq = _maxq_from_bits(int(quant_args.num_bits))

    # Clip ratios: if user provided scalar, use it; else run MSE search (parity)
    if clip_ratio is not None:
        clip_t = torch.tensor(float(clip_ratio), device=W.device, dtype=W.dtype)
    else:
        # parity with your original: compute on W.T (shape (in,out)), per-column/group
        clip_t = find_clip_ratio(
            W.t(),
            maxq=maxq,
            sym=True,
            per_column=True,
            groupsize=groupsize,
        )

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

    loss = torch.sum((W - Wq) ** 2).item()

    # --- Export qparams with packing parity ---
    # Compute qparams from the same clipped min/max quantizer on Wq.T (in,out)
    # and snap the exported weight to the dequant implied by those qparams.
    scale_bt, zp_bt = get_quantize(
        Wq.t(),
        maxq=maxq,
        sym=True,
        per_column=True,
        clip_ratio=clip_t,
        groupsize=groupsize,
    )
    Wq_snapped = quantize_dequant(Wq.t(), maxq, scale_bt, zp_bt).t().contiguous()  # (out,in)

    if strategy in (QuantizationStrategy.GROUP, QuantizationStrategy.TENSOR_GROUP):
        scale_g, zp_g, g_idx = _export_group_qparams_from_broadcast(
            scale_bt=scale_bt,   # (in,out) broadcast
            zp_bt=zp_bt,
            out_features=out_features,
            in_features=in_features,
            group_size=int(quant_args.group_size),
        )
        scale = scale_g
        zero_point = zp_g
    else:

        scale, zero_point = observer(Wq_snapped)
        g_idx = None

    # Restore module-native layout
    if isinstance(module, transformers.Conv1D):
        Wq_snapped.transpose_(0, 1)
    W_export = Wq_snapped.reshape(final_shape).to(final_dtype)

    q_param_dict: Dict[str, torch.Tensor] = {
        "weight": W_export,
        "weight_scale": scale.to(dtype=final_dtype),
        "weight_zero_point": zero_point.to(dtype=quant_args.zp_dtype),
    }
    if g_idx is not None:
        q_param_dict["weight_g_idx"] = g_idx

    logger.info(
        f"ADMMQ time {time.time() - t0:.2f}s mse={loss:.4e} shape={tuple(final_shape)} "
        f"strategy={strategy} groupsize={groupsize} clip={'mse' if clip_ratio is None else clip_ratio}"
    )
    return loss, q_param_dict