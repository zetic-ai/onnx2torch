__all__ = [
    'OnnxGelu20',
]

from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OnnxToTorchModule
from onnx2torch.utils.common import OperationConverterResult
from onnx2torch.utils.common import onnx_mapping_from_node


class OnnxGelu20(nn.Module, OnnxToTorchModule):  # pylint: disable=missing-class-docstring
    def __init__(self, approximate: str = 'none'):
        super().__init__()
        if approximate not in ('none', 'tanh'):
            raise ValueError(f"Unsupported Gelu approximation mode: {approximate}")
        self.approximate = approximate

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:  # pylint: disable=missing-function-docstring
        # torch.nn.functional.gelu supports 'approximate' kwarg with 'none' or 'tanh'
        return F.gelu(input_tensor, approximate=self.approximate)


@add_converter(operation_type='Gelu', version=20)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    approximate = node.attributes.get('approximate', 'none')
    return OperationConverterResult(
        torch_module=OnnxGelu20(approximate=approximate),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )
