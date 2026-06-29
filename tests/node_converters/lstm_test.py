from typing import Dict
from typing import Tuple

import numpy as np
import onnx
import pytest

from tests.utils.common import check_onnx_model
from tests.utils.common import make_model_from_nodes


def _make_lstm_model(  # pylint: disable=missing-function-docstring
    direction: str,
    *,
    seq_len: int,
    batch: int,
    input_size: int,
    hidden: int,
    with_bias: bool,
) -> Tuple[onnx.ModelProto, Dict[str, np.ndarray]]:
    np.random.seed(0)
    num_directions = 2 if direction == 'bidirectional' else 1

    x = np.random.randn(seq_len, batch, input_size).astype(np.float32)
    w = np.random.randn(num_directions, 4 * hidden, input_size).astype(np.float32)
    r = np.random.randn(num_directions, 4 * hidden, hidden).astype(np.float32)

    inputs = ['X', 'W', 'R']
    initializers = {'W': w, 'R': r}
    if with_bias:
        initializers['B'] = np.random.randn(num_directions, 8 * hidden).astype(np.float32)
        inputs.append('B')

    node = onnx.helper.make_node(
        op_type='LSTM',
        inputs=inputs,
        outputs=['Y', 'Y_h', 'Y_c'],
        hidden_size=hidden,
        direction=direction,
    )

    model = make_model_from_nodes(
        nodes=node,
        initializers=initializers,
        inputs_example={'X': x},
        opset_version=13,
    )
    return model, {'X': x}


@pytest.mark.parametrize('direction', ['forward', 'bidirectional'])
@pytest.mark.parametrize('with_bias', [True, False])
def test_lstm(direction: str, with_bias: bool) -> None:  # pylint: disable=missing-function-docstring
    model, test_inputs = _make_lstm_model(
        direction,
        seq_len=5,
        batch=2,
        input_size=3,
        hidden=4,
        with_bias=with_bias,
    )
    # Guards the ONNX [i, o, f, c] -> PyTorch [i, f, c, o] gate reordering:
    # a missing/incorrect permutation makes outputs diverge well above atol.
    check_onnx_model(model, test_inputs, atol_onnx_torch=1e-4)
