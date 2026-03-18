__all__ = [
    'OnnxOneHot',
]

import torch
from torch import nn
from typing import Optional

from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OnnxToTorchModule, OperationConverterResult, onnx_mapping_from_node


class OnnxOneHot(nn.Module, OnnxToTorchModule):
    def __init__(self, axis: int = -1, allow_negative_indices: bool = False):
        super().__init__()
        self.axis = axis
        self.allow_negative_indices = allow_negative_indices

    def forward(
        self,
        indices: torch.Tensor,
        depth: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        depth_val = int(depth.item())
        off_value, on_value = values[0], values[1]

        indices = indices.to(torch.int64)

        if self.allow_negative_indices:
            indices = torch.where(indices < 0, indices + depth_val, indices)
        else:
            pass

        valid_mask = (indices >= 0) & (indices < depth_val)

        one_hot = torch.nn.functional.one_hot(
            torch.clamp(indices, min=0), num_classes=depth_val
        ).to(values.dtype)

        one_hot = torch.where(valid_mask.unsqueeze(-1), one_hot, torch.zeros_like(one_hot))

        one_hot = one_hot * (on_value - off_value) + off_value

        if self.axis != -1 and self.axis != indices.dim():
            one_hot = one_hot.movedim(-1, self.axis)

        return one_hot


@add_converter(operation_type='OneHot', version=9)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:
    axis = node.attributes.get('axis', -1)
    torch_module = OnnxOneHot(axis=axis, allow_negative_indices=False)
    return OperationConverterResult(
        torch_module=torch_module,
        onnx_mapping=onnx_mapping_from_node(node),
    )


@add_converter(operation_type='OneHot', version=11)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:
    axis = node.attributes.get('axis', -1)
    torch_module = OnnxOneHot(axis=axis, allow_negative_indices=True)
    return OperationConverterResult(
        torch_module=torch_module,
        onnx_mapping=onnx_mapping_from_node(node),
    )
