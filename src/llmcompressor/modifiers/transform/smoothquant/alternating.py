import math
import sys
from importlib import util as importlib_util
from pathlib import Path
from types import ModuleType
from typing import ClassVar, Literal

import torch
from compressed_tensors.utils import align_module_device
from loguru import logger

from llmcompressor.modifiers.transform.smoothquant.base import (
    MINIMUM_SMOOTHING_SCALE,
    SmoothQuantModifier,
)

__all__ = ["AlternatingSmoothQuantModifier"]


class AlternatingSmoothQuantModifier(SmoothQuantModifier):
    """
    Separate SmoothQuant modifier that uses alternating scale optimization.
    This keeps recipe selection explicit while reusing SmoothQuant mapping
    resolution and smoothing application behavior.
    """

    smoothing_strategy: Literal["alternating"] = "alternating"
    alternating_outer_iters: int = 1
    alternating_s_init: Literal["heuristic", "ones", "gd"] = "heuristic"
    alternating_accept_tol: float = 0.0
    alternating_ls_ridge: float = 1e-4
    alternating_u_update: Literal["ls_project", "gptq"] = "ls_project"
    alternating_external_impl_path: str | None = None
    alternating_use_external_impl: bool = True

    _external_impl_module_cache: ClassVar[dict[str, ModuleType]] = {}

    def _load_external_impl_module(self) -> ModuleType | None:
        if not self.alternating_use_external_impl:
            return None
        if not self.alternating_external_impl_path:
            return None
    
        script_path = Path(self.alternating_external_impl_path).expanduser().resolve()
        cache_key = str(script_path)
        if cache_key in self._external_impl_module_cache:
            return self._external_impl_module_cache[cache_key]
    
        if not script_path.is_file():
            logger.warning(
                "alternating_external_impl_path does not exist: "
                f"{self.alternating_external_impl_path}"
            )
            return None
    
        module_name = f"_sq_alternating_external_{abs(hash(cache_key))}"
        spec = importlib_util.spec_from_file_location(module_name, str(script_path))
        if spec is None or spec.loader is None:
            logger.warning(
                "Failed to construct module spec for external implementation at "
                f"{script_path}"
            )
            return None
    
        module = importlib_util.module_from_spec(spec)
        inserted = False
        parent_path = str(script_path.parent)
    
        try:
            if parent_path not in sys.path:
                sys.path.insert(0, parent_path)
                inserted = True
    
            # Important: register module before exec_module so decorators like
            # @dataclass can resolve module globals via sys.modules.
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
    
        except Exception as err:
            # Cleanup partial/failed import
            sys.modules.pop(module_name, None)
            logger.warning(
                "Failed to import external alternating implementation from "
                f"{script_path}: {err}"
            )
            return None
    
        finally:
            if inserted:
                try:
                    sys.path.remove(parent_path)
                except ValueError:
                    pass
    
        self._external_impl_module_cache[cache_key] = module
        return module


    def _build_external_sq_cfg(self, external_module: ModuleType):
        sq_cfg_cls = getattr(external_module, "SQCfg", None)
        if sq_cfg_cls is None:
            return None

        return sq_cfg_cls(
            act_bits=int(self.alternating_act_bits),
            w_bits=int(self.alternating_weight_bits),
            alt_iters=int(self.alternating_outer_iters),
            s_sweeps=1,
            golden_max_iter=20,
            s_bounds_mult=float(self.alternating_scale_bounds_mult),
            s_eps=1e-8,
            ls_ridge=float(self.alternating_ls_ridge),
            accept_tol=float(self.alternating_accept_tol),
            coord_backtrack_steps=3,
            outer_accept=True,
            alt_s_init=str(self.alternating_s_init),
            gptq_blocksize=128,
            gptq_percdamp=0.01,
            gptq_actorder=False,
            show_tqdm=False,
            show_coord_tqdm=False,
            screen_gamma=1.0,
            u_update=str(self.alternating_u_update),
        )

    def _optimize_group_with_external_impl(
        self,
        activation_rows: torch.Tensor,
        activation_scales: torch.Tensor,
        balance_layers: list[torch.nn.Module],
    ) -> tuple[torch.Tensor, list[torch.Tensor]] | None:
        external_module = self._load_external_impl_module()
        if external_module is None:
            return None
    
        optimize_fn = getattr(
            external_module, "alternating_optimize_s_and_What_group", None
        )
        heuristic_fn = getattr(external_module, "smoothquant_heuristic_s_group", None)
        gd_init_fn = getattr(external_module, "gd_init_s_ste_group", None)
        if optimize_fn is None or heuristic_fn is None:
            logger.warning(
                "External module missing required functions. Expected "
                "'smoothquant_heuristic_s_group' and "
                "'alternating_optimize_s_and_What_group'."
            )
            return None
    
        cfg = self._build_external_sq_cfg(external_module)
        if cfg is None:
            logger.warning("External module missing SQCfg class; cannot use it.")
            return None
    
        with align_module_device(balance_layers[0]):
            compute_device = balance_layers[0].weight.device
    
        x_rows = activation_rows.to(device=compute_device, dtype=torch.float32)
        ci = int(x_rows.shape[1])
    
        fp_weights_ci_co: list[torch.Tensor] = []
        # True => layer stores (Co, Ci), False => layer stores (Ci, Co)
        layer_is_co_ci: list[bool] = []
    
        for layer in balance_layers:
            with align_module_device(layer):
                w = layer.weight.detach().to(device=compute_device, dtype=torch.float32)
    
            if w.ndim != 2:
                raise ValueError(f"Expected 2D weight, got shape {tuple(w.shape)}")
    
            if w.shape[1] == ci:
                # standard nn.Linear layout (Co, Ci)
                fp_weights_ci_co.append(w.t().contiguous())
                layer_is_co_ci.append(True)
            elif w.shape[0] == ci:
                # already (Ci, Co)
                fp_weights_ci_co.append(w.contiguous())
                layer_is_co_ci.append(False)
            else:
                raise ValueError(
                    f"Cannot align weight shape {tuple(w.shape)} with activation Ci={ci}"
                )
    
        if self.alternating_s_init == "ones":
            s_init = torch.ones((ci,), device=compute_device, dtype=torch.float32)
        elif self.alternating_s_init == "gd" and gd_init_fn is not None:
            s_init = gd_init_fn(X=x_rows, W_fp_list=fp_weights_ci_co, cfg=cfg)
        else:
            s_init = heuristic_fn(
                x_rows,
                fp_weights_ci_co,
                alpha=float(self.smoothing_strength),
                normalize_geom_mean=True,
            )
    
        s_star, w_hat_ci_co, _u_star_ci_co, _f_alt, _f_init = optimize_fn(
            x_rows, fp_weights_ci_co, s_init, cfg, plots_dir=None, module_name=""
        )
    
        # convert back to each layer's native storage layout
        hat_weights_native = []
        for is_co_ci, w_hat in zip(layer_is_co_ci, w_hat_ci_co):
            hat_weights_native.append(w_hat.t().contiguous() if is_co_ci else w_hat.contiguous())
    
        scales = s_star.to(dtype=activation_scales.dtype, device=activation_scales.device)
        return scales, hat_weights_native


    @staticmethod
    def _quantize_per_token_maxabs(
        values: torch.Tensor, bits: int, eps: float = 1e-8
    ) -> torch.Tensor:
        qmax = max((1 << (int(bits) - 1)) - 1, 1)
        denom = values.abs().amax(dim=-1, keepdim=True).clamp_min(eps)
        scale = denom / float(qmax)
        return torch.round(values / scale).clamp(-qmax, qmax) * scale

    @staticmethod
    def _quantize_per_output_channel(
        weights: torch.Tensor, bits: int, eps: float = 1e-8
    ) -> torch.Tensor:
        qmax = max((1 << (int(bits) - 1)) - 1, 1)
        denom = weights.abs().amax(dim=1, keepdim=True).clamp_min(eps)
        scale = denom / float(qmax)
        return torch.round(weights / scale).clamp(-qmax, qmax) * scale

    def _calculate_smoothing_scales_from_weights(
        self, weights: list[torch.Tensor], activation_scales: torch.Tensor
    ) -> torch.Tensor:
        weight_scales = []
        for weight in weights:
            weight_scales.append(weight.abs().max(dim=0)[0])

        stacked = 2.0 * torch.stack(weight_scales, dim=0).max(dim=0)[0]
        scales = activation_scales.pow(self.smoothing_strength) / stacked.pow(
            1 - self.smoothing_strength
        )
        scales = torch.where(stacked > 0.0, scales, activation_scales)
        return scales

    def _objective_for_group(
        self,
        activation_rows: torch.Tensor,
        fp_weights: list[torch.Tensor],
        hat_weights: list[torch.Tensor],
        scales: torch.Tensor,
    ) -> float:
        eps = 1e-8
        xq = self._quantize_per_token_maxabs(
            activation_rows / scales.unsqueeze(0),
            bits=int(self.alternating_act_bits),
            eps=eps,
        )
        loss = torch.zeros((), device=activation_rows.device, dtype=torch.float32)
        for fp_weight, hat_weight in zip(fp_weights, hat_weights):
            target = activation_rows @ fp_weight.t()
            uq = self._quantize_per_output_channel(
                hat_weight * scales.unsqueeze(0),
                bits=int(self.alternating_weight_bits),
                eps=eps,
            )
            pred = xq @ uq.t()
            residual = target - pred
            loss = loss + torch.sum(residual * residual)
        return float(loss.item())

    def _optimize_scales_for_fixed_weights(
        self,
        activation_rows: torch.Tensor,
        fp_weights: list[torch.Tensor],
        hat_weights: list[torch.Tensor],
        initial_scales: torch.Tensor,
    ) -> torch.Tensor:
        eps = 1e-8
        steps = int(self.alternating_steps)
        if steps <= 0:
            return initial_scales

        batch_size = int(self.alternating_batch_size)
        if batch_size <= 0:
            batch_size = activation_rows.shape[0]

        bound_mult = max(float(self.alternating_scale_bounds_mult), 1.0 + 1e-8)
        log_lo = math.log(1.0 / bound_mult)
        log_hi = math.log(bound_mult)

        log_scales = torch.log(initial_scales.clamp_min(eps)).detach().clone()
        log_scales = log_scales - log_scales.mean()
        log_scales.requires_grad_(True)
        optimizer = torch.optim.Adam([log_scales], lr=float(self.alternating_lr))

        with torch.enable_grad():
            for _ in range(steps):
                optimizer.zero_grad(set_to_none=True)

                if activation_rows.shape[0] > batch_size:
                    idx = torch.randperm(
                        activation_rows.shape[0], device=activation_rows.device
                    )[:batch_size]
                    xb = activation_rows.index_select(0, idx)
                else:
                    xb = activation_rows

                scales = torch.exp(log_scales).clamp_min(eps)
                xq = self._ste_quantize_per_token_maxabs(
                    xb / scales.unsqueeze(0),
                    bits=int(self.alternating_act_bits),
                    eps=eps,
                )

                loss = torch.zeros(
                    (), device=activation_rows.device, dtype=torch.float32
                )
                for fp_weight, hat_weight in zip(fp_weights, hat_weights):
                    target = xb @ fp_weight.t()
                    uq = self._ste_quantize_per_output_channel(
                        hat_weight * scales.unsqueeze(0),
                        bits=int(self.alternating_weight_bits),
                        eps=eps,
                    )
                    pred = xq @ uq.t()
                    residual = target - pred
                    loss = loss + torch.sum(residual * residual)

                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    log_scales.clamp_(log_lo, log_hi)
                    log_scales.sub_(log_scales.mean())

        with torch.no_grad():
            scales = torch.exp(log_scales).clamp_min(eps)
            scales = scales / torch.exp(torch.mean(torch.log(scales)))
        return scales

    def _update_hat_weights_from_ls(
        self,
        activation_rows: torch.Tensor,
        fp_weights: list[torch.Tensor],
        scales: torch.Tensor,
    ) -> list[torch.Tensor]:
        if self.alternating_u_update != "ls_project":
            raise ValueError(
                "Unsupported alternating_u_update. Expected 'ls_project', "
                f"received '{self.alternating_u_update}'."
            )

        eps = 1e-8
        xq = self._quantize_per_token_maxabs(
            activation_rows / scales.unsqueeze(0),
            bits=int(self.alternating_act_bits),
            eps=eps,
        )

        gram = xq.t() @ xq
        diag_mean = torch.mean(torch.diag(gram)).clamp_min(eps)
        lam = float(self.alternating_ls_ridge) * float(diag_mean.item())
        gram_reg = gram + lam * torch.eye(
            gram.shape[0], device=gram.device, dtype=gram.dtype
        )

        updated_hat_weights = []
        for fp_weight in fp_weights:
            target = activation_rows @ fp_weight.t()
            xty = xq.t() @ target
            solution = torch.linalg.solve(gram_reg, xty).t()
            uq = self._quantize_per_output_channel(
                solution, bits=int(self.alternating_weight_bits), eps=eps
            )
            updated_hat_weights.append(uq / scales.unsqueeze(0))
        return updated_hat_weights

    def _optimize_group(
        self,
        activation_rows: torch.Tensor,
        activation_scales: torch.Tensor,
        balance_layers: list[torch.nn.Module],
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        external_result = self._optimize_group_with_external_impl(
            activation_rows=activation_rows,
            activation_scales=activation_scales,
            balance_layers=balance_layers,
        )
        if external_result is not None:
            return external_result

        if self.alternating_u_update == "gptq":
            logger.warning(
                "alternating_u_update='gptq' requested but external implementation "
                "is unavailable. Falling back to internal ls_project update."
            )

        with align_module_device(balance_layers[0]):
            compute_device = balance_layers[0].weight.device

        x_rows = activation_rows.to(device=compute_device, dtype=torch.float32)
        fp_weights: list[torch.Tensor] = []
        for layer in balance_layers:
            with align_module_device(layer):
                fp_weights.append(layer.weight.detach().to(device=compute_device, dtype=torch.float32))

        hat_weights = [weight.clone() for weight in fp_weights]
        act_scales = activation_scales.to(device=compute_device, dtype=torch.float32)

        if self.alternating_s_init == "ones":
            scales = torch.ones_like(act_scales)
        else:
            scales = self._calculate_smoothing_scales_from_weights(
                fp_weights, act_scales
            )
            if self.alternating_s_init == "gd":
                scales = self._optimize_scales_for_fixed_weights(
                    activation_rows=x_rows,
                    fp_weights=fp_weights,
                    hat_weights=hat_weights,
                    initial_scales=scales,
                )

        best_scales = scales.clone()
        best_hat_weights = [weight.clone() for weight in hat_weights]
        best_obj = self._objective_for_group(
            activation_rows=x_rows,
            fp_weights=fp_weights,
            hat_weights=hat_weights,
            scales=scales,
        )

        outer_iters = int(max(1, self.alternating_outer_iters))
        for _ in range(outer_iters):
            obj_before = self._objective_for_group(
                activation_rows=x_rows,
                fp_weights=fp_weights,
                hat_weights=hat_weights,
                scales=scales,
            )

            heuristic = self._calculate_smoothing_scales_from_weights(
                hat_weights, act_scales
            )
            s_candidate = self._optimize_scales_for_fixed_weights(
                activation_rows=x_rows,
                fp_weights=fp_weights,
                hat_weights=hat_weights,
                initial_scales=heuristic,
            )
            obj_after_s = self._objective_for_group(
                activation_rows=x_rows,
                fp_weights=fp_weights,
                hat_weights=hat_weights,
                scales=s_candidate,
            )
            if obj_after_s + float(self.alternating_accept_tol) < obj_before:
                scales = s_candidate
                obj_before = obj_after_s

            hat_candidate = self._update_hat_weights_from_ls(
                activation_rows=x_rows,
                fp_weights=fp_weights,
                scales=scales,
            )
            obj_after_u = self._objective_for_group(
                activation_rows=x_rows,
                fp_weights=fp_weights,
                hat_weights=hat_candidate,
                scales=scales,
            )
            if obj_after_u + float(self.alternating_accept_tol) < obj_before:
                hat_weights = hat_candidate
                obj_before = obj_after_u

            if obj_before < best_obj:
                best_obj = obj_before
                best_scales = scales.clone()
                best_hat_weights = [weight.clone() for weight in hat_weights]

        return best_scales, best_hat_weights

    @torch.no_grad()
    def _apply_smoothing(self, model: torch.nn.Module):
        for mapping in self.resolved_mappings_:
            if mapping.smooth_name not in self.scales_:
                continue
            logger.info(f"Alternating smoothing with {mapping.smooth_name}")

            activation_scales = (
                self.scales_[mapping.smooth_name].max_channel_vals
                - self.scales_[mapping.smooth_name].min_channel_vals
            )
            smooth_layer = mapping.smooth_layer
            balance_layers = mapping.balance_layers

            scales = self._calculate_smoothing_scales(balance_layers, activation_scales)
            hat_weights = None

            activation_rows = self.activation_samples_.get(mapping.smooth_name)
            if activation_rows is None or activation_rows.numel() == 0:
                logger.warning(
                    "No cached activation rows for "
                    f"{mapping.smooth_name}; using heuristic SmoothQuant scales."
                )
            else:
                try:
                    scales, hat_weights = self._optimize_group(
                        activation_rows=activation_rows,
                        activation_scales=activation_scales,
                        balance_layers=balance_layers,
                    )
                except Exception as err:
                    logger.warning(
                        "Alternating optimization failed for "
                        f"{mapping.smooth_name}; falling back to heuristic scales. "
                        f"Error: {err}"
                    )

            scales = torch.maximum(
                scales, torch.tensor([MINIMUM_SMOOTHING_SCALE], device=scales.device)
            )

            for idx, layer in enumerate(balance_layers):
                with align_module_device(layer):
                    scale_view = scales.to(
                        device=layer.weight.device, dtype=layer.weight.dtype
                    ).view(1, -1)
                    if hat_weights is not None:
                        layer.weight.copy_(
                            hat_weights[idx].to(
                                device=layer.weight.device, dtype=layer.weight.dtype
                            )
                        )
                    layer.weight.mul_(scale_view)

            with align_module_device(smooth_layer):
                smooth_scales = scales.to(
                    device=smooth_layer.weight.device, dtype=smooth_layer.weight.dtype
                )
                if smooth_layer.weight.ndim == 1:
                    smooth_layer.weight.div_(smooth_scales)
                else:
                    smooth_layer.weight.div_(smooth_scales.view(-1, 1))
                if (
                    hasattr(smooth_layer, "bias")
                    and smooth_layer.bias is not None
                    and smooth_layer.bias.ndim == 1
                ):
                    smooth_layer.bias.div_(smooth_scales.to(smooth_layer.bias.dtype))

            del self.scales_[mapping.smooth_name]
            if (
                self.activation_samples_ is not None
                and mapping.smooth_name in self.activation_samples_
            ):
                del self.activation_samples_[mapping.smooth_name]
