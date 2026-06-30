# onnx2torch/node_converters/grid_sample.py
# pylint: disable=missing-docstring
__all__ = [
    'OnnxGridSample',
]

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn

from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import (
    OnnxMapping,
    OnnxToTorchModule,
    OperationConverterResult,
    onnx_mapping_from_node,
)


def _to_torch_mode(onnx_mode: str) -> str:
    # ONNX v20: 'linear'|'nearest'|'cubic'
    # torch: 'bilinear'|'nearest'|'bicubic'
    onnx_mode = onnx_mode.lower()
    if onnx_mode == 'linear':
        return 'bilinear'
    if onnx_mode == 'nearest':
        return 'nearest'
    if onnx_mode == 'cubic':
        return 'bicubic'
    raise NotImplementedError(f'Unsupported GridSample mode: {onnx_mode}')


class OnnxGridSample(nn.Module, OnnxToTorchModule):
    def __init__(
        self,
        mode: str = 'bilinear',
        padding_mode: str = 'zeros',
        align_corners: bool = False,
    ):
        super().__init__()
        # Basic validation
        if mode not in ('bilinear', 'nearest', 'bicubic'):
            raise NotImplementedError(f'Unsupported torch.grid_sample mode: {mode}')
        if padding_mode not in ('zeros', 'border', 'reflection'):
            raise NotImplementedError(f'Unsupported padding_mode: {padding_mode}')

        self.mode = mode
        self.padding_mode = padding_mode
        self.align_corners = bool(align_corners)

    @staticmethod
    def _cast_grid_dtype(grid: torch.Tensor, target_dtype: torch.dtype) -> torch.Tensor:
        # grid should be float type and usually safe to match input dtype
        if grid.dtype != target_dtype:
            return grid.to(target_dtype)
        return grid

    def forward(self, x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        # PyTorch supports only 4D (NCHW) or 5D (NCDHW)
        if x.dim() not in (4, 5):
            raise NotImplementedError(
                f'GridSample supports only 4D/5D inputs in PyTorch, got {x.dim()}D.'
            )

        # bicubic is 2D-only (per torch documentation)
        if self.mode == 'bicubic' and x.dim() != 4:
            raise NotImplementedError(
                'ONNX GridSample(mode=cubic) on 3D volumes is not supported by '
                'torch.nn.functional.grid_sample (bicubic is 2D-only).'
            )

        # Complex: separate real/imaginary processing
        if torch.is_complex(x):
            # Real/imaginary parts have real dtype
            x_r, x_i = x.real, x.imag
            grid_cast = self._cast_grid_dtype(grid, x_r.dtype)
            y_r = F.grid_sample(
                x_r, grid_cast,
                mode=self.mode,
                padding_mode=self.padding_mode,
                align_corners=self.align_corners,
            )
            y_i = F.grid_sample(
                x_i, grid_cast,
                mode=self.mode,
                padding_mode=self.padding_mode,
                align_corners=self.align_corners,
            )
            return torch.complex(y_r, y_i)

        # Integer/Bool: temporarily convert to float32 → sample → cast to original dtype
        if not x.is_floating_point():
            x_f = x.to(torch.float32)
            grid_cast = self._cast_grid_dtype(grid, x_f.dtype)
            y = F.grid_sample(
                x_f, grid_cast,
                mode=self.mode,
                padding_mode=self.padding_mode,
                align_corners=self.align_corners,
            )
            return y.to(x.dtype)

        # Floating point: as-is (but align grid dtype)
        grid_cast = self._cast_grid_dtype(grid, x.dtype)
        return F.grid_sample(
            x, grid_cast,
            mode=self.mode,
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
        )


def _create_module_from_node_attrs(
    node: OnnxNode,
    default_mode: str,
) -> nn.Module:
    attrs = node.attributes

    onnx_mode = attrs.get('mode', default_mode)
    torch_mode = _to_torch_mode(onnx_mode)

    padding_mode = attrs.get('padding_mode', 'zeros')
    align_corners = bool(attrs.get('align_corners', 0))

    return OnnxGridSample(
        mode=torch_mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )


# opset 16: default mode='bilinear'
@add_converter(operation_type='GridSample', version=16)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    torch_module = _create_module_from_node_attrs(node=node, default_mode='bilinear')
    return OperationConverterResult(
        torch_module=torch_module,
        onnx_mapping=OnnxMapping(
            inputs=(node.input_values[0], node.input_values[1]),
            outputs=node.output_values,
        ),
    )


# opset 20+: default mode='linear' (spec extension: ND support)
@add_converter(operation_type='GridSample', version=20)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    torch_module = _create_module_from_node_attrs(node=node, default_mode='linear')
    return OperationConverterResult(
        torch_module=torch_module,
        onnx_mapping=OnnxMapping(
            inputs=(node.input_values[0], node.input_values[1]),
            outputs=node.output_values,
        ),
    )


# Optional: can register together if behavior is identical even with newer spec (e.g. 22)
@add_converter(operation_type='GridSample', version=22)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    torch_module = _create_module_from_node_attrs(node=node, default_mode='linear')
    return OperationConverterResult(
        torch_module=torch_module,
        onnx_mapping=OnnxMapping(
            inputs=(node.input_values[0], node.input_values[1]),
            outputs=node.output_values,
        ),
    )
