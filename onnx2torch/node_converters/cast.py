__all__ = [
    'OnnxCast',
]

import warnings
import torch
from onnx import TensorProto  # pylint: disable=no-name-in-module
from torch import nn

from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OnnxToTorchModule
from onnx2torch.utils.common import OperationConverterResult
from onnx2torch.utils.common import onnx_mapping_from_node

def _has_attr(name: str) -> bool:
    try:
        getattr(torch, name)
        return True
    except AttributeError:
        return False

# Build dtype map guarded by availability in torch
TENSOR_TYPE_TO_TORCH_TYPE = {
    int(TensorProto.FLOAT): torch.float32,
    int(TensorProto.UINT8): torch.uint8,
    int(TensorProto.INT8): torch.int8,
    int(TensorProto.INT16): torch.int16,
    int(TensorProto.INT32): torch.int32,
    int(TensorProto.INT64): torch.int64,
    int(TensorProto.BOOL): torch.bool,
    int(TensorProto.FLOAT16): torch.float16,
    int(TensorProto.DOUBLE): torch.float64,
    int(TensorProto.BFLOAT16): torch.bfloat16,
    # Note: ONNX STRING has no torch dtype
    # Complex types are not supported by the Cast spec text provided
    # but if desired you may uncomment below (PyTorch supports them):
    # int(TensorProto.COMPLEX64): torch.complex64,
    # int(TensorProto.COMPLEX128): torch.complex128,
}

# Conditionally add float8 dtypes if PyTorch exposes them
if _has_attr('float8_e4m3fn'):
    TENSOR_TYPE_TO_TORCH_TYPE[int(TensorProto.FLOAT8E4M3FN)] = torch.float8_e4m3fn  # type: ignore[attr-defined]
if _has_attr('float8_e5m2'):
    TENSOR_TYPE_TO_TORCH_TYPE[int(TensorProto.FLOAT8E5M2)] = torch.float8_e5m2  # type: ignore[attr-defined]


class OnnxCast(nn.Module, OnnxToTorchModule):  # pylint: disable=missing-docstring
    def __init__(self, onnx_dtype: int, *, saturate: int | None = None, round_mode: str | None = None):
        super().__init__()
        # Resolve target torch dtype
        try:
            self.torch_dtype = TENSOR_TYPE_TO_TORCH_TYPE[onnx_dtype]
        except KeyError as exc:
            raise NotImplementedError(
                f'Conversion to ONNX dtype {onnx_dtype} is not implemented (no matching torch dtype).'
            ) from exc

        # Store attrs (no-ops today, but warn where applicable)
        self.saturate = saturate
        self.round_mode = round_mode

        # Attribute handling notes:
        # - For FLOAT8E4M3FN / FLOAT8E5M2: PyTorch does not expose saturating/rounding control; warn once.
        if self.torch_dtype in (getattr(torch, 'float8_e4m3fn', None), getattr(torch, 'float8_e5m2', None)):
            if saturate is not None or round_mode is not None:
                warnings.warn(
                    'Cast: float8 attributes (saturate/round_mode) are not controllable in PyTorch; '
                    'default rounding/saturation semantics will be used.',
                    RuntimeWarning,
                )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:  # pylint: disable=missing-function-docstring
        # PyTorch cannot hold string tensors; if an upstream graph produced strings,
        # this path won’t be reachable. Guard defensively:
        if input_tensor.dtype is torch.string if hasattr(torch, 'string') else False:  # pragma: no cover
            raise NotImplementedError('Casting from string tensors is not supported in this backend.')

        # No-op if dtype already matches
        if input_tensor.dtype == self.torch_dtype:
            return input_tensor

        return input_tensor.to(self.torch_dtype)


@add_converter(operation_type='Cast', version=9)
@add_converter(operation_type='Cast', version=13)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    onnx_dtype = node.attributes.get('to', None)
    if onnx_dtype is None:
        raise ValueError('Cast: missing required "to" attribute.')

    return OperationConverterResult(
        torch_module=OnnxCast(onnx_dtype),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )


@add_converter(operation_type='Cast', version=19)
@add_converter(operation_type='Cast', version=21)
@add_converter(operation_type='Cast', version=23)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    attrs = node.attributes
    onnx_dtype = attrs.get('to', None)
    if onnx_dtype is None:
        raise ValueError('Cast: missing required "to" attribute.')

    saturate = attrs.get('saturate', 1)

    return OperationConverterResult(
        torch_module=OnnxCast(onnx_dtype, saturate=saturate),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )


@add_converter(operation_type='Cast', version=24)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    attrs = node.attributes
    onnx_dtype = attrs.get('to', None)
    if onnx_dtype is None:
        raise ValueError('Cast: missing required "to" attribute.')

    round_mode = attrs.get('round_mode', 'up')
    saturate = attrs.get('saturate', 1)

    return OperationConverterResult(
        torch_module=OnnxCast(onnx_dtype, saturate=saturate, round_mode=round_mode),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )
