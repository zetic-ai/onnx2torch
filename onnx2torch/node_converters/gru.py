# pylint: disable=missing-class-docstring
__all__ = [
    'OnnxGRU',
]

from typing import Any, Dict, List, Mapping, Optional, Tuple
import torch
from torch import nn

from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OperationConverterResult, onnx_mapping_from_node
from onnx2torch.utils.custom_export_to_onnx import OnnxToTorchModuleWithCustomExport


class OnnxGRU(nn.Module, OnnxToTorchModuleWithCustomExport):
    def __init__(
        self,
        hidden_size: int,
        direction: str = 'forward',
        activations: Optional[List[str]] = None,
        activation_alpha: Optional[List[float]] = None,
        activation_beta: Optional[List[float]] = None,
        clip: Optional[float] = None,
        linear_before_reset: int = 0,
        layout: int = 0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.direction = direction
        self.activations = activations or ['Sigmoid', 'Tanh']
        self.activation_alpha = activation_alpha
        self.activation_beta = activation_beta
        self.clip = clip
        self.linear_before_reset = linear_before_reset
        self.layout = layout

        # No nn.GRU instance: W/R/B arrive as forward() inputs and are run
        # through the functional torch.gru, matching the LSTM converter.

    def _onnx_attrs(self, opset_version: int) -> Dict[str, Any]:
        return {
            'hidden_size_i': self.hidden_size,
            'direction_s': self.direction,
            'activations_s': self.activations,
            'activation_alpha_floats': self.activation_alpha or [],
            'activation_beta_floats': self.activation_beta or [],
            'clip_f': self.clip if self.clip is not None else 0.0,
            'linear_before_reset_i': self.linear_before_reset,
            'layout_i': self.layout,
        }

    def forward(
        self,
        X: torch.Tensor,
        W: torch.Tensor,
        R: torch.Tensor,
        B: Optional[torch.Tensor] = None,
        sequence_lens: Optional[torch.Tensor] = None,
        initial_h: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        batch_first = self.layout == 1

        if not batch_first:
            X = X.transpose(0, 1)  # [S, B, I] -> [B, S, I]

        batch_size = X.size(0)
        num_directions = 2 if self.direction == 'bidirectional' else 1
        bidirectional = self.direction == 'bidirectional'
        H = self.hidden_size

        if initial_h is not None:
            h0 = initial_h
        else:
            h0 = torch.zeros(num_directions, batch_size, H, device=X.device, dtype=X.dtype)

        # ONNX packs gates as [z, r, h] (update, reset, hidden);
        # PyTorch expects [r, z, n]. Swapping the first two is the whole change.
        def _reorder_gates(t: torch.Tensor) -> torch.Tensor:
            z, r, h = t.chunk(3, dim=0)
            return torch.cat([r, z, h], dim=0)

        params: List[torch.Tensor] = []
        has_biases = B is not None
        for d in range(num_directions):
            params.append(_reorder_gates(W[d]))  # weight_ih
            params.append(_reorder_gates(R[d]))  # weight_hh
            if has_biases:
                # ONNX concatenates Wb and Rb into a single [6*H] row per direction.
                params.append(_reorder_gates(B[d, : 3 * H]))  # bias_ih
                params.append(_reorder_gates(B[d, 3 * H :]))  # bias_hh

        Y, Y_h = torch.gru(
            X,
            h0,
            params,
            has_biases,
            1,  # num_layers
            0.0,  # dropout
            False,  # train
            bidirectional,
            True,  # batch_first
        )

        # torch Y: [B, S, H*num_directions] -> ONNX Y: [S, num_directions, B, H]
        Bsz, S, _ = Y.shape
        nd = num_directions
        Y = Y.view(Bsz, S, nd, H).transpose(0, 1)  # [S, B, nd, H]
        Y = Y.transpose(1, 2)  # [S, nd, B, H]

        return Y, Y_h


# The default ONNX activations, which is what torch.gru computes.
_DEFAULT_ACTIVATIONS = ['Sigmoid', 'Tanh']

# ONNX GRU optional-input positions (per the operator spec).
_SEQUENCE_LENS_INPUT_INDEX = 4


def _assert_supported(node: OnnxNode, attrs: Mapping[str, Any], num_directions: int) -> None:
    """Raise for ONNX GRU features torch.gru cannot represent (else they would be silently miscomputed)."""
    direction = attrs.get('direction', 'forward')
    if direction not in ('forward', 'bidirectional'):
        raise NotImplementedError(
            f"ONNX GRU direction={direction!r} is not supported (only 'forward' and 'bidirectional')."
        )

    # The single most dangerous attribute here. ONNX defines two different
    # formulations of the hidden gate and defaults to the one PyTorch does NOT
    # implement:
    #   linear_before_reset=0: ht = g(Wh*xt + rt o (Rh*Ht-1) + Rbh + Wbh)
    #   linear_before_reset=1: ht = g(Wh*xt + rt o (Rh*Ht-1 + Rbh) + Wbh)
    # torch.gru only computes the second. Accepting the first would produce a
    # numerically wrong model that converts, benchmarks and ships silently.
    if not attrs.get('linear_before_reset', 0):
        raise NotImplementedError(
            "ONNX GRU 'linear_before_reset=0' is not supported: torch.gru applies the reset "
            "gate after the hidden-state matmul plus its bias (equivalent to "
            "linear_before_reset=1), so converting a linear_before_reset=0 graph would "
            "silently change what the model computes. Re-export with linear_before_reset=1."
        )

    if attrs.get('clip', None) is not None:
        raise NotImplementedError("ONNX GRU 'clip' attribute is not supported.")
    if attrs.get('layout', 0):
        raise NotImplementedError("ONNX GRU 'layout=1' is not supported (only the default layout=0).")
    if attrs.get('activation_alpha', None) or attrs.get('activation_beta', None):
        raise NotImplementedError("ONNX GRU custom 'activation_alpha'/'activation_beta' is not supported.")

    activations = attrs.get('activations', None)
    if activations is not None:
        expected = [a.lower() for a in _DEFAULT_ACTIVATIONS * num_directions]
        if [a.lower() for a in activations] != expected:
            raise NotImplementedError(
                f"ONNX GRU custom activations {list(activations)} are not supported "
                "(only the default Sigmoid/Tanh)."
            )

    inputs = node.input_values
    if len(inputs) > _SEQUENCE_LENS_INPUT_INDEX and inputs[_SEQUENCE_LENS_INPUT_INDEX]:
        raise NotImplementedError("ONNX GRU 'sequence_lens' input is not supported.")


@add_converter(operation_type='GRU', version=1)
@add_converter(operation_type='GRU', version=3)
@add_converter(operation_type='GRU', version=7)
@add_converter(operation_type='GRU', version=14)
@add_converter(operation_type='GRU', version=22)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:
    attrs = node.attributes
    hidden_size = attrs['hidden_size']
    direction = attrs.get('direction', 'forward')
    activations = attrs.get('activations', None)
    activation_alpha = attrs.get('activation_alpha', None)
    activation_beta = attrs.get('activation_beta', None)
    clip = attrs.get('clip', None)
    linear_before_reset = attrs.get('linear_before_reset', 0)
    layout = attrs.get('layout', 0)

    num_directions = 2 if direction == 'bidirectional' else 1
    _assert_supported(node, attrs, num_directions)

    return OperationConverterResult(
        torch_module=OnnxGRU(
            hidden_size=hidden_size,
            direction=direction,
            activations=activations,
            activation_alpha=activation_alpha,
            activation_beta=activation_beta,
            clip=clip,
            linear_before_reset=linear_before_reset,
            layout=layout,
        ),
        onnx_mapping=onnx_mapping_from_node(node),
    )
