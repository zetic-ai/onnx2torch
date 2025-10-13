# pylint: disable=missing-class-docstring
__all__ = [
    'OnnxSTFT',
]

from typing import Any, Dict, Optional, Tuple, cast

import torch
from torch import nn

from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OnnxToTorchModule
from onnx2torch.utils.common import OperationConverterResult
from onnx2torch.utils.common import get_const_value
from onnx2torch.utils.common import get_onnx_version
from onnx2torch.utils.common import onnx_mapping_from_node


class OnnxSTFT(nn.Module, OnnxToTorchModule):
    def __init__(self, onesided: int = 1):
        super().__init__()
        self.onesided = bool(onesided)

    def forward(
        self,
        signal: torch.Tensor,
        frame_step: torch.Tensor,
        window: Optional[torch.Tensor] = None,
        frame_length: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        hop_length = int(frame_step.item())

        # ---- Normalize input shape ----
        # ONNX real:  [B, L, 1]  -> [B, L]
        # ONNX cplx:  [B, L, 2]  -> complex [B, L]
        is_complex_input = False
        if signal.dim() == 3 and signal.size(-1) == 1:
            signal = signal.squeeze(-1)
        elif signal.dim() == 3 and signal.size(-1) == 2:
            signal = torch.complex(signal[..., 0], signal[..., 1])
            is_complex_input = True
        # allow [L] or [B, L] as-is; other ranks can be handled as needed

        # ---- n_fft / window / win_length ----
        n_fft = None
        if frame_length is not None:
            n_fft = int(frame_length.item())
        elif window is not None and window.dim() == 1:
            n_fft = int(window.numel())
        else:
            n_fft = hop_length * 2  # fallback

        win = None
        if window is not None:
            # match device (and keep float dtype; torch.stft expects real window)
            win = window.to(device=signal.device)
            # If window length != n_fft, torch.stft still allows specifying win_length separately.
            # We'll pass win_length=n_fft to match ONNX's DFT size.
            # (Optionally, you could validate len(win)==n_fft and pad/truncate if needed.)

        # ---- onesided handling ----
        # ONNX: complex input -> onesided must be False
        onesided = False if is_complex_input else self.onesided

        # ---- call torch.stft ----
        stft_out = torch.stft(
            input=signal,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=win,
            center=False,            # ONNX 명세상 패딩/센터링 정의가 없으니 center=False가 보수적
            onesided=onesided,
            return_complex=False     # ONNX는 마지막 차원 [.., 2] 요구
        )
        # torch.stft output: [..., freq, frames, 2]
        # ONNX expects:      [..., frames, freq, 2]
        if stft_out.dim() >= 3:
            # swap freq <-> frames (마지막 3,4번째 축)
            stft_out = stft_out.movedim(-3, -2)  # or stft_out.permute(..., -2, -3, -1)

        return stft_out



@add_converter(operation_type='STFT', version=17)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:
    node_attributes = node.attributes
    onesided: int = node_attributes.get('onesided', 1)

    return OperationConverterResult(
        torch_module=OnnxSTFT(onesided=onesided),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )
