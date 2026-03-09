# =====================================================================
# FILE: llmcompressor/modifiers/quantization/admmq/base.py
# =====================================================================
import contextlib
import inspect
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
from compressed_tensors.offload.dist_utils import as_broadcastable, is_distributed
from compressed_tensors.quantization import (
    QuantizationConfig,
    QuantizationScheme,
    QuantizationStrategy,
)
from compressed_tensors.quantization.quant_args import ActivationOrdering
from compressed_tensors.utils import (
    align_module_device,
    get_execution_device,
    getattr_chain,
    match_named_modules,
    update_offload_parameter,
)
from loguru import logger
from pydantic import PrivateAttr
from torch import distributed as dist

from llmcompressor.core import Event, EventType, State
from llmcompressor.modifiers import Modifier
from llmcompressor.modifiers.quantization.calibration import update_weight_global_scale
from llmcompressor.modifiers.quantization.gptq.gptq_quantize import make_empty_hessian
from llmcompressor.modifiers.quantization.quantization import QuantizationMixin
from llmcompressor.modifiers.utils import update_fused_layer_weight_global_scales
from llmcompressor.sentinel import Sentinel
from llmcompressor.utils import greedy_bin_packing, wait_for_comms
from llmcompressor.utils.metric_logging import CompressionLogger

from .admmq_quantize import quantize_weight_admmq, accumulate_xtx_admm

__all__ = ["ADMMQModifier"]

_ADMMQ_Q_PARAMS = ["weight", "weight_scale", "weight_zero_point", "weight_g_idx"]


def _to_device_any(x: Any, device: torch.device | str):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, (tuple, list)):
        return type(x)(_to_device_any(t, device) for t in x)
    if isinstance(x, dict):
        return {k: _to_device_any(v, device) for k, v in x.items()}
    return x


def _to_cpu_any(x: Any):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().cpu()
    if isinstance(x, (tuple, list)):
        return type(x)(_to_cpu_any(t) for t in x)
    if isinstance(x, dict):
        return {k: _to_cpu_any(v) for k, v in x.items()}
    return x


def _slice_batch_any(x: Any, b: int, B: int):
    if x is None:
        return None
    if torch.is_tensor(x):
        if x.ndim >= 1 and x.shape[0] == B:
            return x[b : b + 1]
        return x
    if isinstance(x, (tuple, list)):
        return type(x)(_slice_batch_any(t, b, B) for t in x)
    if isinstance(x, dict):
        return {k: _slice_batch_any(v, b, B) for k, v in x.items()}
    return x


def _layer_accepts_kw(layer: torch.nn.Module, kw: str) -> bool:
    try:
        sig = inspect.signature(layer.forward)
        return kw in sig.parameters
    except Exception:
        # Most HF layers accept **kwargs; safe to pass.
        return True


class ADMMQModifier(Modifier, QuantizationMixin):
    """
    ADMMQ Modifier similar to GPTQModifier
    """

    sequential_targets: Union[str, List[str], None] = None
    offload_hessians: bool = False

    rho: float = 0.1
    max_iter: int = 300
    update_iter: int = 3
    switch_iter: int = 30
    update_quant: bool = False

    clip_ratio: Optional[float] = None

    pipeline_parity_calibration: bool = True
    nsamples: int = 128
    stop_collect_after_nsamples: bool = False

    _module_names: Dict[torch.nn.Module, str] = PrivateAttr(default_factory=dict)

    _hessians: Dict[torch.nn.Module, torch.Tensor] = PrivateAttr(default_factory=dict)
    _num_samples: Dict[torch.nn.Module, torch.Tensor] = PrivateAttr(default_factory=dict)

    _replay_cache: List[Tuple[torch.Tensor, Any, Any]] = PrivateAttr(default_factory=list)
    _layer0_handle: Optional[object] = PrivateAttr(default=None)

    _qwen3_rotary_emb: Optional[torch.nn.Module] = PrivateAttr(default=None)

    def resolve_quantization_config(self) -> QuantizationConfig:
        config = super().resolve_quantization_config()

        def resolve_actorder(existing):
            if getattr(self, "actorder", Sentinel("static")) == Sentinel("static"):
                return ActivationOrdering.STATIC if existing is None else existing
            if existing is None or self.actorder == existing:
                return self.actorder
            raise ValueError(
                "Cannot resolve activation ordering when both "
                "`ADMMQModifier.actorder` and `QuantizationScheme.actorder` "
                f"are provided and differ ({self.actorder}, {existing}). "
                "Either unset `ADMMQModifier.actorder` or remove `actorder` from config groups."
            )

        for scheme in config.config_groups.values():
            assert isinstance(scheme, QuantizationScheme)
            if getattr_chain(scheme, "weights.strategy", None) == QuantizationStrategy.GROUP:
                scheme.weights.actorder = resolve_actorder(scheme.weights.actorder)
        return config

    def on_initialize(self, state: State, **kwargs) -> bool:
        if QuantizationMixin.has_config(self):
            QuantizationMixin.initialize_quantization(self, state.model)

        self._module_names = {
            m: name
            for name, m in match_named_modules(state.model, self.resolved_targets, self.ignore)
        }
        return True

    def _get_decoder_layers(self, model: torch.nn.Module):
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model.layers
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            return model.transformer.h
        raise ValueError(
            "pipeline_parity_calibration could not locate decoder layers. "
            "Expected `model.model.layers` or `model.transformer.h`."
        )

    def _get_qwen3_rotary_emb(self, model: torch.nn.Module) -> Optional[torch.nn.Module]:
        cand = getattr_chain(model, "model.rotary_emb", None)
        if cand is not None:
            return cand
        cand = getattr(model, "rotary_emb", None)
        if cand is not None:
            return cand
        cand = getattr_chain(model, "base_model.model.rotary_emb", None)
        if cand is not None:
            return cand
        cand = getattr_chain(model, "transformer.rotary_emb", None)
        if cand is not None:
            return cand
        return None

    def _ensure_bsh(self, x: torch.Tensor, *, B_hint: Optional[int] = None) -> torch.Tensor:
        """
        Normalize hidden states to (B,S,H).

        Cases handled:
          - (S,H)           -> (1,S,H)
          - (B,S,H)         -> unchanged
          - (B*S,H)         -> (B,S,H) if B_hint provided and divisible
        """
        if x is None or (not torch.is_tensor(x)):
            return x
        if x.ndim == 3:
            return x
        if x.ndim == 2:
            # (S,H) or (B*S,H)
            if B_hint is not None and B_hint > 0 and (x.shape[0] % B_hint == 0):
                S = x.shape[0] // B_hint
                return x.view(B_hint, S, x.shape[1])
            return x.unsqueeze(0)
        # unexpected shapes: best effort, leave as-is
        return x

    def _capture_layer0_input(self, module, args, kwargs):
        if len(self._replay_cache) >= int(self.nsamples):
            if self.stop_collect_after_nsamples:
                raise RuntimeError("__ADMMQ_STOP_CALIB__")
            return

        hidden = args[0].detach()
        hidden = self._ensure_bsh(hidden)
        B = hidden.shape[0]

        attn = kwargs.get("attention_mask", None)
        pos_ids = kwargs.get("position_ids", None)

        for b in range(B):
            if len(self._replay_cache) >= int(self.nsamples):
                break
            hb = hidden[b : b + 1].detach().cpu()
            ab = _to_cpu_any(_slice_batch_any(attn, b, B))
            pb = _to_cpu_any(_slice_batch_any(pos_ids, b, B))
            self._replay_cache.append((hb, ab, pb))

    def _ensure_position_ids(
        self, x: torch.Tensor, position_ids: Optional[torch.Tensor]
    ) -> torch.Tensor:
        x = self._ensure_bsh(x)
        B, S, _ = x.shape
        dev = x.device
        if position_ids is not None and torch.is_tensor(position_ids):
            pid = position_ids.to(dev)
            if pid.ndim == 1:
                pid = pid.view(1, -1).expand(B, -1)
            return pid
        return torch.arange(S, device=dev, dtype=torch.long).view(1, S).expand(B, S)

    def _compute_qwen3_position_embeddings(
        self,
        hidden_states: torch.Tensor,   # (B,S,H)
        position_ids: torch.Tensor,    # (B,S)
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        rotary = self._qwen3_rotary_emb
        if rotary is None or (not callable(rotary)):
            return None

        hidden_states = self._ensure_bsh(hidden_states)
        for call in (
            lambda: rotary(hidden_states, position_ids),
            lambda: rotary(hidden_states, position_ids, hidden_states.shape[1]),
            lambda: rotary(position_ids),
        ):
            try:
                out = call()
                if (
                    isinstance(out, (tuple, list))
                    and len(out) == 2
                    and torch.is_tensor(out[0])
                    and torch.is_tensor(out[1])
                ):
                    return out[0], out[1]
            except TypeError:
                continue
            except Exception:
                continue
        return None

    def _build_replay_kwargs(
        self,
        *,
        layer: torch.nn.Module,
        x: torch.Tensor,
        attention_mask: Any,
        position_ids_cached: Any,
        dev: torch.device | str,
    ) -> Dict[str, Any]:
        kw: Dict[str, Any] = {}

        x = self._ensure_bsh(x)

        if _layer_accepts_kw(layer, "use_cache"):
            kw["use_cache"] = False
        if _layer_accepts_kw(layer, "past_key_values"):
            kw["past_key_values"] = None

        if attention_mask is not None and _layer_accepts_kw(layer, "attention_mask"):
            kw["attention_mask"] = _to_device_any(attention_mask, dev)

        pid = None
        if position_ids_cached is not None and torch.is_tensor(position_ids_cached):
            pid = position_ids_cached
        pid = self._ensure_position_ids(x, pid)
        if _layer_accepts_kw(layer, "position_ids"):
            kw["position_ids"] = pid

        if _layer_accepts_kw(layer, "cache_position"):
            _, S, _ = x.shape
            kw["cache_position"] = torch.arange(S, device=x.device, dtype=torch.long)

        if _layer_accepts_kw(layer, "position_embeddings"):
            pe = self._compute_qwen3_position_embeddings(x, pid)
            if pe is None:
                raise RuntimeError(
                    "Could not compute Qwen3 position_embeddings=(cos,sin) during replay. "
                    "Expected to find a callable model-level rotary embedding at `model.model.rotary_emb` "
                    "or `model.rotary_emb`."
                )
            kw["position_embeddings"] = pe

        return kw

    @torch.no_grad()
    def _sequential_replay_compress(self, model: torch.nn.Module):

        layers = self._get_decoder_layers(model)
        dev = get_execution_device(model)

        inps = [x for (x, _, _) in self._replay_cache]
        masks = [m for (_, m, _) in self._replay_cache]
        posids = [p for (_, _, p) in self._replay_cache]

        for li, layer in enumerate(layers):
            logger.info(f"[ADMMQ] Sequential replay quantizing decoder layer {li}/{len(layers)}")
            layer = layer.to(dev)
            layer.eval()

            target_modules: List[torch.nn.Module] = []
            for m in layer.modules():
                if m in self._module_names:
                    if getattr_chain(m, "quantization_scheme.weights", None) is not None:
                        if not isinstance(m, torch.nn.Embedding):
                            target_modules.append(m)

            # infer a B hint from the cached sample itself (it is always (1,S,H) as cached)
            B_hint = None
            if len(inps) > 0 and torch.is_tensor(inps[0]):
                x0 = inps[0]
                if x0.ndim == 3:
                    B_hint = x0.shape[0]

            if len(target_modules) == 0:
                outs_cpu: List[torch.Tensor] = []
                for j in range(len(inps)):
                    x = self._ensure_bsh(inps[j].to(dev), B_hint=B_hint)
                    kw = self._build_replay_kwargs(
                        layer=layer,
                        x=x,
                        attention_mask=masks[j],
                        position_ids_cached=posids[j],
                        dev=dev,
                    )
                    y = layer(x, **kw)
                    y0 = y[0] if isinstance(y, (tuple, list)) else y
                    y0 = self._ensure_bsh(y0, B_hint=x.shape[0])
                    outs_cpu.append(y0.detach().cpu())
                layers[li] = layer.cpu()
                del layer
                torch.cuda.empty_cache()
                inps = outs_cpu
                continue

            local_H: Dict[torch.nn.Module, torch.Tensor] = {}
            local_n: Dict[torch.nn.Module, torch.Tensor] = {}

            def make_hook(mod):
                def _hook(_m, args, out):
                    inp = args[0]
                    inp = self._ensure_bsh(inp)
                    if mod not in local_H:
                        local_H[mod] = make_empty_hessian(mod, device=get_execution_device(mod))
                        local_n[mod] = torch.zeros((), device=get_execution_device(mod))
                    local_H[mod], local_n[mod] = accumulate_xtx_admm(inp, mod, local_H[mod], local_n[mod])
                return _hook

            handles = [m.register_forward_hook(make_hook(m)) for m in target_modules]

            # Pass 1: collect XtX
            tmp_outs: List[torch.Tensor] = []
            for j in range(len(inps)):
                x = self._ensure_bsh(inps[j].to(dev), B_hint=B_hint)
                kw = self._build_replay_kwargs(
                    layer=layer,
                    x=x,
                    attention_mask=masks[j],
                    position_ids_cached=posids[j],
                    dev=dev,
                )
                y = layer(x, **kw)
                y0 = y[0] if isinstance(y, (tuple, list)) else y
                y0 = self._ensure_bsh(y0, B_hint=x.shape[0])
                tmp_outs.append(y0.detach().cpu())

            for h in handles:
                h.remove()

            # Quantize targets
            for m in target_modules:
                quant_args = getattr_chain(m, "quantization_scheme.weights")
                if quant_args is None:
                    continue
                xtx = local_H.get(m, None)
                if xtx is None:
                    continue

                with (
                    torch.no_grad(),
                    align_module_device(m),
                    CompressionLogger(m) as comp_logger,
                ):
                    loss, q_param_dict = quantize_weight_admmq(
                        module=m,
                        quant_args=quant_args,
                        xtx=xtx,
                        rho=self.rho,
                        max_iter=self.max_iter,
                        update_iter=self.update_iter,
                        switch_iter=self.switch_iter,
                        update_quant=self.update_quant,
                        clip_ratio=self.clip_ratio,
                    )
                    comp_logger.set_loss(loss)

                for attr, val in q_param_dict.items():
                    update_offload_parameter(m, attr, val)

            # Pass 2: recompute outputs with quantized weights
            outs2_cpu: List[torch.Tensor] = []
            for j in range(len(inps)):
                x = self._ensure_bsh(inps[j].to(dev), B_hint=B_hint)
                kw = self._build_replay_kwargs(
                    layer=layer,
                    x=x,
                    attention_mask=masks[j],
                    position_ids_cached=posids[j],
                    dev=dev,
                )
                y = layer(x, **kw)
                y0 = y[0] if isinstance(y, (tuple, list)) else y
                y0 = self._ensure_bsh(y0, B_hint=x.shape[0])
                outs2_cpu.append(y0.detach().cpu())

            layers[li] = layer.cpu()
            del layer
            torch.cuda.empty_cache()

            inps = outs2_cpu

    def on_start(self, state: State, event: Event, **kwargs):
        self.started_ = True
        QuantizationMixin.start_calibration(self, state.model)

        named_modules = list(match_named_modules(state.model, self.resolved_targets, self.ignore))

        for _, module in named_modules:
            update_weight_global_scale(module)
        for module in state.model.modules():
            update_fused_layer_weight_global_scales(module)

        if self.pipeline_parity_calibration:
            self._qwen3_rotary_emb = self._get_qwen3_rotary_emb(state.model)

            layers = self._get_decoder_layers(state.model)
            self._replay_cache = []
            if self._layer0_handle is None:
                self._layer0_handle = layers[0].register_forward_pre_hook(
                    lambda mod, args, kwargs: self._capture_layer0_input(mod, args, kwargs),
                    with_kwargs=True,
                )
            logger.info(f"[ADMMQ] pipeline_parity_calibration enabled; caching nsamples={self.nsamples}")
            return

        # Legacy hook mode (kept for completeness)
        added_hook = False
        for _, module in named_modules:
            if getattr_chain(module, "quantization_scheme.weights", None) is not None:
                if not isinstance(module, torch.nn.Embedding):
                    self.register_hook(module, self.calibrate_module, "forward")
                    added_hook = True
        if not added_hook:
            raise ValueError(
                "ADMMQModifier requires a weight quantization config be specified by "
                "this modifier or a modifier preceding it"
            )

    def on_event(self, state: State, event: Event, **kwargs):
        if event.type_ == EventType.CALIBRATION_EPOCH_START:
            if not self.started_:
                self.on_start(state, None)

        if self.pipeline_parity_calibration:
            if event.type_ == EventType.CALIBRATION_EPOCH_END:
                if self._layer0_handle is not None:
                    try:
                        self._layer0_handle.remove()
                    finally:
                        self._layer0_handle = None

                logger.info(
                    f"[ADMMQ] Collected {len(self._replay_cache)} cached samples. Starting sequential replay."
                )
                self._sequential_replay_compress(state.model)

                if not self.ended_:
                    self.on_end(state, None)
            return

        # Legacy hook mode
        if event.type_ == EventType.SEQUENTIAL_EPOCH_END:
            self.compress_modules()
        if event.type_ == EventType.CALIBRATION_EPOCH_END:
            self.compress_modules()
            if not self.ended_:
                self.on_end(state, None)

    def calibrate_module(
        self,
        module: torch.nn.Module,
        args: Tuple[torch.Tensor, ...],
        _output: torch.Tensor,
    ):
        inp = args[0]
        inp = self._ensure_bsh(inp)

        if module not in self._num_samples:
            init_device = "cpu" if self.offload_hessians else get_execution_device(module)
            self._hessians[module] = make_empty_hessian(module, device=init_device)
            self._num_samples[module] = torch.zeros(tuple(), device=get_execution_device(module))

        with self._maybe_onload_hessian(module):
            self._hessians[module], self._num_samples[module] = accumulate_xtx_admm(
                inp,
                module,
                self._hessians[module],
                self._num_samples[module],
            )

    def compress_modules(self):
        if not is_distributed():
            self.compress_module_list(list(self._num_samples.keys()))
            return

        rank = dist.get_rank()
        world_size = dist.get_world_size()

        module_list, rank_to_modules, module_to_rank = greedy_bin_packing(
            list(self._hessians.keys()),
            world_size,
            item_weight_fn=lambda mod: self._hessians[mod].shape[0],
        )

        self._reduce_hessian_to_target_rank(module_list, module_to_rank)
        self.compress_module_list(rank_to_modules[rank])
        self._broadcast_quantized_params(module_list, module_to_rank)

    def compress_module_list(self, module_list):
        for module in module_list:
            quant_args = getattr_chain(module, "quantization_scheme.weights")

            with (
                torch.no_grad(),
                align_module_device(module),
                self._maybe_onload_hessian(module),
                CompressionLogger(module) as comp_logger,
            ):
                xtx = self._hessians.pop(module)
                _ = self._num_samples.pop(module, None)

                loss, q_param_dict = quantize_weight_admmq(
                    module=module,
                    quant_args=quant_args,
                    xtx=xtx,
                    rho=self.rho,
                    max_iter=self.max_iter,
                    update_iter=self.update_iter,
                    switch_iter=self.switch_iter,
                    update_quant=self.update_quant,
                    clip_ratio=self.clip_ratio,
                )
                comp_logger.set_loss(loss)

            for attr, val in q_param_dict.items():
                update_offload_parameter(module, attr, val)

    def _reduce_hessian_to_target_rank(self, module_list, module_to_rank):
        rank = dist.get_rank()
        pending_comms = []
        for module in module_list:
            target_rank = module_to_rank[module]
            with self._maybe_onload_hessian(module):
                pending_comms.append(
                    dist.reduce(
                        self._hessians[module],
                        op=dist.ReduceOp.SUM,
                        dst=target_rank,
                        async_op=True,
                    )
                )
                pending_comms.append(
                    dist.reduce(
                        self._num_samples[module],
                        op=dist.ReduceOp.SUM,
                        dst=target_rank,
                        async_op=True,
                    )
                )
                if rank != target_rank:
                    self._hessians.pop(module, None)
                    self._num_samples.pop(module, None)
        wait_for_comms(pending_comms)

    def _broadcast_quantized_params(self, module_list, module_to_rank):
        pending_comms = []
        for module in module_list:
            src_rank = module_to_rank[module]
            for attr in _ADMMQ_Q_PARAMS:
                if getattr(module, attr, None) is not None:
                    pending_comms.append(
                        dist.broadcast(
                            as_broadcastable(getattr(module, attr)),
                            src=src_rank,
                            async_op=True,
                        )
                    )
        wait_for_comms(pending_comms)

    def on_end(self, state: State, event: Event, **kwargs):
        self.ended_ = True
        QuantizationMixin.end_calibration(self, state.model)
        self.remove_hooks()

        self._replay_cache = []
        if self._layer0_handle is not None:
            try:
                self._layer0_handle.remove()
            finally:
                self._layer0_handle = None

        self._qwen3_rotary_emb = None

    def on_finalize(self, state: State, **kwargs) -> bool:
        if not self.ended_:
            self.on_end(state, None)

        if len(self._num_samples) > 0:
            raise ValueError(f"Failed to compress {len(self._num_samples)} modules")

        self._hessians = dict()
        self._num_samples = dict()
        return True

    @contextlib.contextmanager
    def _maybe_onload_hessian(self, module: torch.nn.Module):
        if self.offload_hessians:
            device = get_execution_device(module)
            self._hessians[module] = self._hessians[module].to(device=device)

        yield

        if self.offload_hessians:
            if module in self._hessians:
                self._hessians[module] = self._hessians[module].to(device="cpu")