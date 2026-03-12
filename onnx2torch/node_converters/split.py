__all__ = [
    'OnnxSplit',
    'OnnxSplit13',
    'OnnxSplit18',
]

from typing import List
from typing import Optional

import torch
from torch import nn

from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OnnxToTorchModule
from onnx2torch.utils.common import OperationConverterResult
from onnx2torch.utils.common import onnx_mapping_from_node


class OnnxSplit13(nn.Module, OnnxToTorchModule):  # pylint: disable=missing-class-docstring
    def __init__(self, num_splits: int, axis: int = 0):
        super().__init__()

        self.axis = axis
        self.num_splits = num_splits

    def forward(  # pylint: disable=missing-function-docstring
        self,
        input_tensor: torch.Tensor,
        split: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if split is None:
            axis_len = input_tensor.shape[self.axis]
            split_size_or_sections = axis_len // self.num_splits
        else:
            split_size_or_sections = split.tolist()

        return torch.split(input_tensor, split_size_or_sections, dim=self.axis)


class OnnxSplit(nn.Module, OnnxToTorchModule):  # pylint: disable=missing-class-docstring
    def __init__(self, num_splits: int, axis: int = 0, split: Optional[List[int]] = None):
        super().__init__()

        self.axis = axis
        self.num_splits = num_splits
        self.split = split

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:  # pylint: disable=missing-function-docstring
        if self.split is None:
            axis_len = input_tensor.shape[self.axis]
            split_size_or_sections = axis_len // self.num_splits
        else:
            split_size_or_sections = self.split

        return torch.split(input_tensor, split_size_or_sections, dim=self.axis)


class OnnxSplit18(nn.Module, OnnxToTorchModule):  # pylint: disable=missing-class-docstring
    """
    Opset 18 behavior:
    - Either 'split' input (sizes) OR 'num_outputs' attribute specifies splitting.
    - If 'num_outputs' is used and the axis length is not divisible, the last chunk is smaller.
    """
    def __init__(self, axis: int = 0, num_outputs: Optional[int] = None, num_splits_fallback: Optional[int] = None):
        super().__init__()
        self.axis = axis
        # Prefer explicit num_outputs attribute; if absent, fall back to the number of graph outputs.
        self.num_outputs = num_outputs
        self.num_splits_fallback = num_splits_fallback

    def forward(
        self,
        input_tensor: torch.Tensor,
        split: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if split is not None:
            # Explicit per-output sizes provided via second input
            sizes = split.tolist()
            return torch.split(input_tensor, sizes, dim=self.axis)

        # Use number of sections (opset 18 allows uneven last chunk)
        sections = self.num_outputs if self.num_outputs is not None else self.num_splits_fallback
        if sections is None:
            # Defensive fallback (should not happen in valid models)
            # Use full length -> single output (no real split)
            return (input_tensor,)

        return torch.tensor_split(input_tensor, sections, dim=self.axis)


@add_converter(operation_type='Split', version=18)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    axis = node.attributes.get('axis', 0)
    # Opset 18 introduces 'num_outputs' attribute
    num_outputs_attr = node.attributes.get('num_outputs', None)
    # Fallback to the number of outputs defined in the graph, to keep output arity consistent
    num_splits_fallback = len(node.output_values)

    return OperationConverterResult(
        torch_module=OnnxSplit18(axis=axis, num_outputs=num_outputs_attr, num_splits_fallback=num_splits_fallback),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )


@add_converter(operation_type='Split', version=13)
@add_converter(operation_type='Split', version=18)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    axis = node.attributes.get('axis', 0)
    num_splits = node.attributes.get('num_outputs', None) or len(node.output_values)
    return OperationConverterResult(
        torch_module=OnnxSplit13(axis=axis, num_splits=num_splits),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )


@add_converter(operation_type='Split', version=11)
@add_converter(operation_type='Split', version=2)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    axis = node.attributes.get('axis', 0)
    split = node.attributes.get('split', None)
    num_splits = len(node.output_values)
    return OperationConverterResult(
        torch_module=OnnxSplit(axis=axis, split=split, num_splits=num_splits),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )
