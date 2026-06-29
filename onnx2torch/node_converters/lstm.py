# pylint: disable=missing-class-docstring
__all__ = [
    'OnnxLSTM',
]

from typing import Any, Dict, List, Mapping, Optional, Tuple, cast
import torch
from torch import nn

from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OnnxMapping, OnnxToTorchModule, OperationConverterResult, get_const_value, get_onnx_version, onnx_mapping_from_node
from onnx2torch.utils.custom_export_to_onnx import DefaultExportToOnnx, OnnxToTorchModuleWithCustomExport


class OnnxLSTM(nn.Module, OnnxToTorchModuleWithCustomExport):
    def __init__(
        self,
        hidden_size: int,
        direction: str = 'forward',
        activations: Optional[List[str]] = None,
        activation_alpha: Optional[List[float]] = None,
        activation_beta: Optional[List[float]] = None,
        clip: Optional[float] = None,
        input_forget: int = 0,
        layout: int = 0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.direction = direction
        self.activations = activations or ['Sigmoid', 'Tanh', 'Tanh']
        self.activation_alpha = activation_alpha
        self.activation_beta = activation_beta
        self.clip = clip
        self.input_forget = input_forget
        self.layout = layout

        # NOTE: nn.LSTM is intentionally NOT instantiated here. The ONNX weight
        # tensors (W/R/B) are delivered as forward() inputs, and forward() runs
        # the functional torch.lstm directly. Creating an nn.LSTM(input_size=0)
        # placeholder both fails on newer PyTorch (input_size must be > 0) and
        # would otherwise force an un-traceable runtime weight copy.

    def _onnx_attrs(self, opset_version: int) -> Dict[str, Any]:
        return {
            'hidden_size_i': self.hidden_size,
            'direction_s': self.direction,
            'activations_s': self.activations,
            'activation_alpha_floats': self.activation_alpha or [],
            'activation_beta_floats': self.activation_beta or [],
            'clip_f': self.clip if self.clip is not None else 0.0,
            'input_forget_i': self.input_forget,
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
        initial_c: Optional[torch.Tensor] = None,
        P: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        batch_first = self.layout == 1

        # Run the ONNX LSTM through the functional torch.lstm using the weight
        # tensors directly. This avoids both:
        #   1. re-instantiating nn.LSTM with a runtime-derived input_size
        #      (a Tensor under tracing; also rejected by newer PyTorch), and
        #   2. the in-place copy_/select/slice weight assignment, whose
        #      tensor-assignment ops the coremltools TorchScript frontend
        #      cannot lower ("No matching select or slice.").
        if not batch_first:
            # X: [S, B, I] -> [B, S, I]
            X = X.transpose(0, 1)

        # After the (possible) transpose, batch is always dim 0.
        batch_size = X.size(0)
        num_directions = 2 if self.direction == 'bidirectional' else 1
        bidirectional = self.direction == 'bidirectional'
        H = self.hidden_size

        if initial_h is not None:
            h0 = initial_h
        else:
            h0 = torch.zeros(num_directions, batch_size, H, device=X.device, dtype=X.dtype)

        if initial_c is not None:
            c0 = initial_c
        else:
            c0 = torch.zeros(num_directions, batch_size, H, device=X.device, dtype=X.dtype)

        # ONNX packs the four gates as [input, output, forget, cell], whereas
        # PyTorch expects [input, forget, cell, output]. Reorder the gate blocks
        # (rows) of each weight/bias accordingly: [i, o, f, c] -> [i, f, c, o].
        def _reorder_gates(t: torch.Tensor) -> torch.Tensor:
            i, o, f, c = t.chunk(4, dim=0)
            return torch.cat([i, f, c, o], dim=0)

        params: List[torch.Tensor] = []
        has_biases = B is not None
        for d in range(num_directions):
            params.append(_reorder_gates(W[d]))  # weight_ih
            params.append(_reorder_gates(R[d]))  # weight_hh
            if has_biases:
                params.append(_reorder_gates(B[d, : 4 * H]))  # bias_ih
                params.append(_reorder_gates(B[d, 4 * H :]))  # bias_hh

        # Signature: torch.lstm(input, hx, params, has_biases, num_layers,
        #            dropout, train, bidirectional, batch_first).
        # X is already batch-first at this point.
        Y, Y_h, Y_c = torch.lstm(
            X,
            (h0, c0),
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

        return Y, Y_h, Y_c


# Default ONNX LSTM activations (per direction): f=Sigmoid, g=Tanh, h=Tanh.
# These are exactly what the functional torch.lstm computes.
_DEFAULT_ACTIVATIONS = ['Sigmoid', 'Tanh', 'Tanh']

# ONNX LSTM optional-input positions (see the ONNX operator spec).
_SEQUENCE_LENS_INPUT_INDEX = 4
_PEEPHOLE_INPUT_INDEX = 7


def _assert_supported(node: OnnxNode, attrs: Mapping[str, Any], num_directions: int) -> None:
    """Reject ONNX LSTM features that this converter does not faithfully implement.

    The converter lowers the ONNX LSTM onto the functional ``torch.lstm``, which
    only covers the default configuration: ``forward``/``bidirectional`` directions,
    the default Sigmoid/Tanh/Tanh activations, no gate clipping, no peephole
    connections, no coupled input-forget gate, no per-sequence masking, and the
    default ``layout=0``. Any other configuration would be silently miscomputed,
    so we fail loudly here instead of producing a wrong model.
    """
    direction = attrs.get('direction', 'forward')
    if direction not in ('forward', 'bidirectional'):
        raise NotImplementedError(
            f"ONNX LSTM direction={direction!r} is not supported (only 'forward' and 'bidirectional')."
        )
    if attrs.get('clip', None) is not None:
        raise NotImplementedError("ONNX LSTM 'clip' attribute is not supported.")
    if attrs.get('input_forget', 0):
        raise NotImplementedError("ONNX LSTM 'input_forget' attribute is not supported.")
    if attrs.get('layout', 0):
        raise NotImplementedError("ONNX LSTM 'layout=1' is not supported (only the default layout=0).")
    if attrs.get('activation_alpha', None) or attrs.get('activation_beta', None):
        raise NotImplementedError("ONNX LSTM custom 'activation_alpha'/'activation_beta' is not supported.")

    activations = attrs.get('activations', None)
    if activations is not None:
        expected = [a.lower() for a in _DEFAULT_ACTIVATIONS * num_directions]
        if [a.lower() for a in activations] != expected:
            raise NotImplementedError(
                f"ONNX LSTM custom activations {list(activations)} are not supported "
                "(only the default Sigmoid/Tanh/Tanh)."
            )

    inputs = node.input_values
    if len(inputs) > _SEQUENCE_LENS_INPUT_INDEX and inputs[_SEQUENCE_LENS_INPUT_INDEX]:
        raise NotImplementedError("ONNX LSTM 'sequence_lens' input is not supported.")
    if len(inputs) > _PEEPHOLE_INPUT_INDEX and inputs[_PEEPHOLE_INPUT_INDEX]:
        raise NotImplementedError("ONNX LSTM peephole 'P' input is not supported.")


@add_converter(operation_type='LSTM', version=1)
@add_converter(operation_type='LSTM', version=7)
@add_converter(operation_type='LSTM', version=14)
@add_converter(operation_type='LSTM', version=22)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:
    attrs = node.attributes
    hidden_size = attrs['hidden_size']
    direction = attrs.get('direction', 'forward')
    activations = attrs.get('activations', None)
    activation_alpha = attrs.get('activation_alpha', None)
    activation_beta = attrs.get('activation_beta', None)
    clip = attrs.get('clip', None)
    input_forget = attrs.get('input_forget', 0)
    layout = attrs.get('layout', 0)

    num_directions = 2 if direction == 'bidirectional' else 1
    _assert_supported(node, attrs, num_directions)

    return OperationConverterResult(
        torch_module=OnnxLSTM(
            hidden_size=hidden_size,
            direction=direction,
            activations=activations,
            activation_alpha=activation_alpha,
            activation_beta=activation_beta,
            clip=clip,
            input_forget=input_forget,
            layout=layout,
        ),
        onnx_mapping=onnx_mapping_from_node(node),
    )
