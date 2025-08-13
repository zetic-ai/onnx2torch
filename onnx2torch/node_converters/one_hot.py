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
        """
        :param axis: one-hot 차원을 삽입할 축
        :param allow_negative_indices: v11에서처럼 음수 인덱스 허용 여부
        """
        super().__init__()
        self.axis = axis
        self.allow_negative_indices = allow_negative_indices

    def forward(
        self,
        indices: torch.Tensor,
        depth: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        # depth와 values는 scalar 또는 1D tensor 형식이므로 Python 값으로 변환
        depth_val = int(depth.item())
        off_value, on_value = values[0], values[1]

        # 인덱스 정수 변환
        indices = indices.to(torch.int64)

        if self.allow_negative_indices:
            # v11: 음수 인덱스를 depth 기준 wrap-around 처리
            indices = torch.where(indices < 0, indices + depth_val, indices)
        else:
            # v9: 음수 인덱스는 모두 범위 밖 처리
            pass

        # 범위 밖 인덱스 마스킹
        valid_mask = (indices >= 0) & (indices < depth_val)

        # PyTorch one_hot: 마지막 차원에 depth 추가
        one_hot = torch.nn.functional.one_hot(
            torch.clamp(indices, min=0), num_classes=depth_val
        ).to(values.dtype)

        # 범위 밖은 모두 off_value
        one_hot = torch.where(valid_mask.unsqueeze(-1), one_hot, torch.zeros_like(one_hot))

        # on_value / off_value 적용
        one_hot = one_hot * (on_value - off_value) + off_value

        # axis 위치에 맞게 차원 이동
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
