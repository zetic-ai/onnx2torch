__all__ = [
    'OnnxClip',
]

from typing import Optional, Tuple

import torch
from torch import nn
from torch.types import Number

from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OnnxMapping
from onnx2torch.utils.common import OnnxToTorchModule
from onnx2torch.utils.common import OperationConverterResult
from onnx2torch.utils.common import get_const_value
from onnx2torch.utils.common import onnx_mapping_from_node


class OnnxClip(nn.Module, OnnxToTorchModule):
    """Static (constant) min/max version."""
    def __init__(
        self,
        min_val: Optional[Number] = None,
        max_val: Optional[Number] = None,
    ):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return torch.clamp(input_tensor, self.min_val, self.max_val)


class OnnxClipDynamic(nn.Module, OnnxToTorchModule):
    """Dynamic min/max version (min/max are tensors at runtime)."""
    def forward(
        self,
        input_tensor: torch.Tensor,
        min_tensor: Optional[torch.Tensor] = None,
        max_tensor: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        y = input_tensor
        if min_tensor is not None:
            # Ensure dtype/broadcasting ok
            y = torch.maximum(y, min_tensor.to(dtype=y.dtype))
        if max_tensor is not None:
            y = torch.minimum(y, max_tensor.to(dtype=y.dtype))
        return y


def _create_torch_module(min_val: Optional[Number], max_val: Optional[Number]) -> nn.Module:
    if min_val is None and max_val is None:
        return nn.Identity()
    if min_val == 0 and max_val is None:
        return nn.ReLU()
    if min_val == 0 and max_val == 6:
        return nn.ReLU6()
    return OnnxClip(min_val=min_val, max_val=max_val)


def _normalize_name(name: Optional[str]) -> Optional[str]:
    # Treat empty string as "not provided" (opset 11+ optional inputs)
    return name if name not in (None, '') else None


@add_converter(operation_type='Clip', version=11)
@add_converter(operation_type='Clip', version=12)
@add_converter(operation_type='Clip', version=13)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:
    # Optional inputs
    min_name = _normalize_name(node.input_values[1] if len(node.input_values) > 1 else None)
    max_name = _normalize_name(node.input_values[2] if len(node.input_values) > 2 else None)

    const_min: Optional[float] = None
    const_max: Optional[float] = None
    needs_dynamic = False

    # Try to resolve constants; if not found, we’ll go dynamic.
    if min_name is not None:
        try:
            const_min = float(get_const_value(min_name, graph))
        except Exception:
            needs_dynamic = True
    if max_name is not None:
        try:
            const_max = float(get_const_value(max_name, graph))
        except Exception:
            needs_dynamic = True

    if not needs_dynamic:
        torch_module = _create_torch_module(min_val=const_min, max_val=const_max)
        inputs: Tuple[str, ...] = (node.input_values[0],)
    else:
        # Build dynamic module and pass through any provided min/max tensors
        torch_module = OnnxClipDynamic()
        dyn_inputs = [node.input_values[0]]
        if min_name is not None:
            dyn_inputs.append(min_name)
        if max_name is not None:
            dyn_inputs.append(max_name)
        inputs = tuple(dyn_inputs)

    return OperationConverterResult(
        torch_module=torch_module,
        onnx_mapping=OnnxMapping(
            inputs=inputs,
            outputs=node.output_values,
        ),
    )


@add_converter(operation_type='Clip', version=6)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    node_attributes = node.attributes
    min_val = node_attributes.get('min', None)
    max_val = node_attributes.get('max', None)

    torch_module = _create_torch_module(min_val=min_val, max_val=max_val)

    return OperationConverterResult(
        torch_module=torch_module,
        onnx_mapping=onnx_mapping_from_node(node),
    )
