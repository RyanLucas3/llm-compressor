# =====================================================================
# FILE: llmcompressor/modifiers/quantization/admmq/base.py
# =====================================================================
import contextlib
from typing import Dict, List, Optional, Tuple, Union

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
from llmcompressor.modifiers.quantization.gptq.gptq_quantize import (
    accumulate_hessian,
    make_empty_hessian,
)
from llmcompressor.modifiers.quantization.quantization import QuantizationMixin
from llmcompressor.modifiers.utils import update_fused_layer_weight_global_scales
from llmcompressor.sentinel import Sentinel
from llmcompressor.utils import greedy_bin_packing, wait_for_comms
from llmcompressor.utils.metric_logging import CompressionLogger

from .admmq_quantize import quantize_weight_admmq

__all__ = ["ADMMQModifier"]

_ADMMQ_Q_PARAMS = ["weight", "weight_scale", "weight_zero_point", "weight_g_idx"]


class ADMMQModifier(Modifier, QuantizationMixin):
    """
    Implements an ADMM-based second-order weight quantization algorithm using
    per-layer XtX (Hessian approximation) accumulated from activations.

    This class is intentionally structured to mirror GPTQModifier for maximum
    compatibility with llm-compressor lifecycle and distributed behavior.

    Sample yaml:

    ```yaml
    test_stage:
      obcq_modifiers:
        ADMMQModifier:
          rho: 0.1
          max_iter: 300
          update_iter: 3
          switch_iter: 30
          update_quant: false
          offload_hessians: false
          config_groups:
            group_0:
              targets:
                - "Linear"
              weights:
                num_bits: 8
                type: "int"
                symmetric: true
                strategy: group
                group_size: 128
    ```
    """

    # modifier arguments (mirrors GPTQ)
    sequential_targets: Union[str, List[str], None] = None
    offload_hessians: bool = False

    # ADMMQ algorithm arguments
    rho: float = 0.1
    max_iter: int = 300
    update_iter: int = 3
    switch_iter: int = 30
    update_quant: bool = False

    # Optional: allow user to override per-column clipping ratio (None disables)
    clip_ratio: Optional[float] = None

    # private variables
    _module_names: Dict[torch.nn.Module, str] = PrivateAttr(default_factory=dict)
    _hessians: Dict[torch.nn.Module, torch.Tensor] = PrivateAttr(default_factory=dict)
    _num_samples: Dict[torch.nn.Module, torch.Tensor] = PrivateAttr(default_factory=dict)

    def resolve_quantization_config(self) -> QuantizationConfig:
        """
        Keep the same actorder resolution logic as GPTQ to avoid conflicts when users
        set actorder at either modifier or scheme level.

        ADMMQ does not use actorder internally, but GROUP/TENSOR_GROUP schemes may
        carry actorder; we preserve the same validation to remain consistent.
        """
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
                "Either unset `ADMMQModifier.actorder` or "
                "remove `actorder` from config groups."
            )

        # No-op unless user provided actorder; still validates conflicting configs.
        for scheme in config.config_groups.values():
            assert isinstance(scheme, QuantizationScheme)
            if getattr_chain(scheme, "weights.strategy", None) == QuantizationStrategy.GROUP:
                scheme.weights.actorder = resolve_actorder(scheme.weights.actorder)
        return config

    def on_initialize(self, state: State, **kwargs) -> bool:
        # apply config to model and prepare calibration hooks
        if QuantizationMixin.has_config(self):
            QuantizationMixin.initialize_quantization(self, state.model)

        # prepare module names
        self._module_names = {
            m: name
            for name, m in match_named_modules(
                state.model, self.resolved_targets, self.ignore
            )
        }
        return True

    def on_start(self, state: State, event: Event, **kwargs):
        self.started_ = True

        # register quantization calibration hooks
        QuantizationMixin.start_calibration(self, state.model)

        added_hook = False
        named_modules = list(match_named_modules(state.model, self.resolved_targets, self.ignore))

        for _, module in named_modules:
            if getattr_chain(module, "quantization_scheme.weights", None) is not None:
                # keep same embedding skip as GPTQ for parity
                if not isinstance(module, torch.nn.Embedding):
                    self.register_hook(module, self.calibrate_module, "forward")
                    added_hook = True

        # Optionally generate global scales if using TENSOR_GROUP quantization
        for _, module in named_modules:
            update_weight_global_scale(module)
        for module in state.model.modules():
            update_fused_layer_weight_global_scales(module)

        if not added_hook:
            raise ValueError(
                "ADMMQModifier requires a weight quantization config be specified by "
                "this modifier or a modifier preceding it"
            )

    def on_event(self, state: State, event: Event, **kwargs):
        if event.type_ == EventType.CALIBRATION_EPOCH_START:
            if not self.started_:
                self.on_start(state, None)

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
        # Assume that first argument is the input
        inp = args[0]

        # Initialize hessian if not present
        if module not in self._num_samples:
            init_device = "cpu" if self.offload_hessians else get_execution_device(module)
            self._hessians[module] = make_empty_hessian(module, device=init_device)
            self._num_samples[module] = torch.zeros(tuple(), device=get_execution_device(module))

        # Accumulate hessian with input with optional offloading
        with self._maybe_onload_hessian(module):
            self._hessians[module], self._num_samples[module] = accumulate_hessian(
                inp,
                module,
                self._hessians[module],
                self._num_samples[module],
            )

    def compress_modules(self):
        # Not Distributed
        if not is_distributed():
            self.compress_module_list(list(self._num_samples.keys()))
            return

        # Distributed
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
            name = self._module_names[module]
            num_samples = self._num_samples[module]
            quant_args = getattr_chain(module, "quantization_scheme.weights")

            logger.info(f"Quantizing {name} using {num_samples} samples")
            with (
                torch.no_grad(),
                align_module_device(module),
                self._maybe_onload_hessian(module),
                CompressionLogger(module) as comp_logger,
            ):
                loss, q_param_dict = quantize_weight_admmq(
                    module=module,
                    quant_args=quant_args,
                    xtx=self._hessians.pop(module) / self._num_samples.pop(module),
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
