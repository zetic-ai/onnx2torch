from typing import Dict
from typing import Optional
from typing import Tuple

import numpy as np
import onnx
import pytest

from onnx2torch import convert
from tests.utils.common import check_onnx_model
from tests.utils.common import make_model_from_nodes


def _make_lstm_model(  # pylint: disable=missing-function-docstring,too-many-locals
    direction: str,
    *,
    seq_len: int,
    batch: int,
    input_size: int,
    hidden: int,
    with_bias: bool,
    extra_attrs: Optional[Dict] = None,
    with_sequence_lens: bool = False,
    with_peephole: bool = False,
    opset_version: int = 13,
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
    else:
        inputs.append('')

    if with_sequence_lens:
        # sequence_lens occupies input index 4.
        initializers['sequence_lens'] = np.full((batch,), seq_len, dtype=np.int32)
        inputs.append('sequence_lens')

    if with_peephole:
        # Peephole tensor P occupies input index 7; pad the skipped optionals.
        while len(inputs) < 7:
            inputs.append('')
        initializers['P'] = np.random.randn(num_directions, 3 * hidden).astype(np.float32)
        inputs.append('P')

    attrs = {'hidden_size': hidden, 'direction': direction}
    if extra_attrs:
        attrs.update(extra_attrs)

    node = onnx.helper.make_node(
        op_type='LSTM',
        inputs=inputs,
        outputs=['Y', 'Y_h', 'Y_c'],
        **attrs,
    )

    model = make_model_from_nodes(
        nodes=node,
        initializers=initializers,
        inputs_example={'X': x},
        opset_version=opset_version,
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
    # Guards the gate reordering: a wrong permutation diverges well above atol.
    check_onnx_model(model, test_inputs, atol_onnx_torch=1e-4)


# Configs the converter rejects rather than silently miscomputing.
_UNSUPPORTED_CASES = [
    ('direction_reverse', dict(direction='reverse')),
    ('clip', dict(extra_attrs={'clip': 1.0})),
    ('input_forget', dict(extra_attrs={'input_forget': 1})),
    ('layout', dict(extra_attrs={'layout': 1}, opset_version=14)),
    ('custom_activations', dict(extra_attrs={'activations': ['Relu', 'Tanh', 'Tanh']})),
    ('sequence_lens', dict(with_sequence_lens=True)),
    ('peephole', dict(with_peephole=True)),
]


@pytest.mark.parametrize('label, kwargs', _UNSUPPORTED_CASES, ids=[c[0] for c in _UNSUPPORTED_CASES])
def test_lstm_unsupported_raises(
    label: str, kwargs: Dict
) -> None:  # pylint: disable=missing-function-docstring,unused-argument
    params = dict(seq_len=5, batch=2, input_size=3, hidden=4, with_bias=True)
    params.update(kwargs)
    direction = params.pop('direction', 'forward')
    model, _ = _make_lstm_model(direction, **params)

    with pytest.raises(NotImplementedError):
        convert(model)
