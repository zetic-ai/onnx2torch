# pylint: disable=missing-docstring
__all__ = [
    'OnnxIsNaN',
]

import torch
from torch import nn

from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OnnxMapping
from onnx2torch.utils.common import OnnxToTorchModule
from onnx2torch.utils.common import OperationConverterResult


class OnnxIsNaN(nn.Module, OnnxToTorchModule):
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return torch.isnan(input_tensor)


# version 9:  T1 in (float16, float, double)
# version 13: T1 adds bfloat16
# version 20: T1 adds float8e4m3fn, float8e4m3fnuz, float8e5m2, float8e5m2fnuz
@add_converter(operation_type='IsNaN', version=9)
@add_converter(operation_type='IsNaN', version=13)
@add_converter(operation_type='IsNaN', version=20)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:
    del graph
    torch_module = OnnxIsNaN()

    return OperationConverterResult(
        torch_module=torch_module,
        onnx_mapping=OnnxMapping(
            inputs=(node.input_values[0],),
            outputs=node.output_values,
        ),
    )
