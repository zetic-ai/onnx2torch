# pylint: disable=missing-class-docstring
__all__ = [
    'OnnxLSTM',
]

from typing import Any, Dict, List, Optional, Tuple, cast
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

        num_directions = 2 if direction == 'bidirectional' else 1
        self.lstm = nn.LSTM(
            input_size=0,  # will be inferred at runtime
            hidden_size=hidden_size,
            num_layers=1,
            bias=True,
            batch_first=(layout == 1),
            bidirectional=(direction == 'bidirectional')
        )

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
        P: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        batch_first = (self.layout == 1)

        # If ONNX layout==0, transpose to batch-first for PyTorch
        if not batch_first:
            # X: [S, B, I] -> [B, S, I]
            X = X.transpose(0, 1)

        # After the (possible) transpose, batch is always dim 0
        batch_size = X.size(0)
        num_directions = 2 if self.direction == 'bidirectional' else 1
        input_size = X.size(-1)

        # (Re)build LSTM with correct input_size
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            bias=True,
            batch_first=True,  # we've made X batch-first now
            bidirectional=(self.direction == 'bidirectional'),
        )

        # Copy weights/biases from ONNX tensors
        with torch.no_grad():
            for d in range(num_directions):
                suffix = '_reverse' if d == 1 else ''
                getattr(self.lstm, f'weight_ih_l0{suffix}').copy_(W[d])
                getattr(self.lstm, f'weight_hh_l0{suffix}').copy_(R[d])

                if B is not None:
                    bias_w = B[d, :4*self.hidden_size]
                    bias_r = B[d, 4*self.hidden_size:]
                    getattr(self.lstm, f'bias_ih_l0{suffix}').copy_(bias_w)
                    getattr(self.lstm, f'bias_hh_l0{suffix}').copy_(bias_r)
                else:
                    getattr(self.lstm, f'bias_ih_l0{suffix}').zero_()
                    getattr(self.lstm, f'bias_hh_l0{suffix}').zero_()

        # Prepare h0/c0 with correct batch_size
        if initial_h is not None:
            h0 = initial_h
        else:
            h0 = torch.zeros(num_directions, batch_size, self.hidden_size, device=X.device, dtype=X.dtype)

        if initial_c is not None:
            c0 = initial_c
        else:
            c0 = torch.zeros(num_directions, batch_size, self.hidden_size, device=X.device, dtype=X.dtype)

        # Run LSTM (X is batch-first)
        Y, (Y_h, Y_c) = self.lstm(X, (h0, c0))

        # ONNX wants Y: [S, num_directions, B, H]
        # PyTorch returned Y: [B, S, H*num_directions] (because batch_first=True)
        Bsz, S, Hnd = Y.shape
        H = self.hidden_size
        nd = num_directions
        Y = Y.view(Bsz, S, nd, H).transpose(0, 1)  # [S, B, nd, H]
        Y = Y.transpose(1, 2)                      # [S, nd, B, H]

        return Y, Y_h, Y_c



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
