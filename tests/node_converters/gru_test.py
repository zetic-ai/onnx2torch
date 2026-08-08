from typing import Dict
from typing import Optional
from typing import Tuple

import numpy as np
import onnx
import pytest

from onnx2torch import convert
from tests.utils.common import check_onnx_model
from tests.utils.common import make_model_from_nodes


def _make_gru_model(  # pylint: disable=missing-function-docstring,too-many-locals
    direction: str,
    *,
    seq_len: int,
    batch: int,
    input_size: int,
    hidden: int,
    with_bias: bool,
    with_initial_h: bool = False,
    extra_attrs: Optional[Dict] = None,
    with_sequence_lens: bool = False,
    opset_version: int = 13,
) -> Tuple[onnx.ModelProto, Dict[str, np.ndarray]]:
    np.random.seed(0)
    num_directions = 2 if direction == 'bidirectional' else 1

    x = np.random.randn(seq_len, batch, input_size).astype(np.float32)
    w = np.random.randn(num_directions, 3 * hidden, input_size).astype(np.float32)
    r = np.random.randn(num_directions, 3 * hidden, hidden).astype(np.float32)

    inputs = ['X', 'W', 'R']
    initializers = {'W': w, 'R': r}
    if with_bias:
        # ONNX concatenates Wb and Rb into one 6*hidden row per direction.
        initializers['B'] = np.random.randn(num_directions, 6 * hidden).astype(np.float32)
        inputs.append('B')
    else:
        inputs.append('')

    if with_sequence_lens:
        # sequence_lens occupies input index 4.
        initializers['sequence_lens'] = np.full((batch,), seq_len, dtype=np.int32)
        inputs.append('sequence_lens')
    else:
        inputs.append('')

    if with_initial_h:
        # initial_h occupies input index 5.
        initializers['initial_h'] = np.random.randn(num_directions, batch, hidden).astype(np.float32)
        inputs.append('initial_h')

    # linear_before_reset=1 is the only formulation torch.gru implements, so it
    # is the default everywhere in these tests; =0 is covered as a rejection.
    attrs = {'hidden_size': hidden, 'direction': direction, 'linear_before_reset': 1}
    if extra_attrs:
        attrs.update(extra_attrs)

    node = onnx.helper.make_node(
        op_type='GRU',
        inputs=inputs,
        outputs=['Y', 'Y_h'],
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
def test_gru(direction: str, with_bias: bool) -> None:  # pylint: disable=missing-function-docstring
    model, test_inputs = _make_gru_model(
        direction,
        seq_len=5,
        batch=2,
        input_size=3,
        hidden=4,
        with_bias=with_bias,
    )
    # Guards the gate reordering: ONNX packs [z, r, h] and torch expects
    # [r, z, n], so a wrong permutation diverges well above atol.
    check_onnx_model(model, test_inputs, atol_onnx_torch=1e-4)


def test_gru_initial_h() -> None:
    """initial_h must be honoured; ignoring it silently changes the first step."""
    model, test_inputs = _make_gru_model(
        'forward',
        seq_len=5,
        batch=2,
        input_size=3,
        hidden=4,
        with_bias=True,
        with_initial_h=True,
    )
    check_onnx_model(model, test_inputs, atol_onnx_torch=1e-4)


def test_gru_hidden_state_on_graph_boundary() -> None:
    """The shape a static NPU export needs: hidden state carried in and out.

    Matches the NVIDIA Audio2Face-3D graph that motivated this converter
    (hidden_size=256, forward, linear_before_reset=1, initial_h supplied),
    where the GRU hidden state is a named graph input.
    """
    model, test_inputs = _make_gru_model(
        'forward',
        seq_len=6,
        batch=1,
        input_size=256,
        hidden=256,
        with_bias=True,
        with_initial_h=True,
    )
    check_onnx_model(model, test_inputs, atol_onnx_torch=1e-4)


# Configs the converter rejects rather than silently miscomputing.
_UNSUPPORTED_CASES = [
    ('direction_reverse', dict(direction='reverse')),
    ('linear_before_reset_0', dict(extra_attrs={'linear_before_reset': 0})),
    ('clip', dict(extra_attrs={'clip': 1.0})),
    ('layout', dict(extra_attrs={'layout': 1}, opset_version=14)),
    ('custom_activations', dict(extra_attrs={'activations': ['Relu', 'Tanh']})),
    ('sequence_lens', dict(with_sequence_lens=True)),
]


@pytest.mark.parametrize('label, kwargs', _UNSUPPORTED_CASES, ids=[c[0] for c in _UNSUPPORTED_CASES])
def test_gru_unsupported_raises(
    label: str, kwargs: Dict
) -> None:  # pylint: disable=missing-function-docstring,unused-argument
    params = dict(seq_len=5, batch=2, input_size=3, hidden=4, with_bias=True)
    params.update(kwargs)
    direction = params.pop('direction', 'forward')
    model, _ = _make_gru_model(direction, **params)

    with pytest.raises(NotImplementedError):
        convert(model)


def test_gru_default_linear_before_reset_is_rejected() -> None:
    """ONNX defaults linear_before_reset to 0, which torch.gru cannot express.

    Omitting the attribute entirely must be rejected for the same reason as
    setting it to 0 — otherwise the default case silently miscomputes.
    """
    np.random.seed(0)
    hidden, input_size, seq_len, batch = 4, 3, 5, 2
    x = np.random.randn(seq_len, batch, input_size).astype(np.float32)
    node = onnx.helper.make_node(
        op_type='GRU',
        inputs=['X', 'W', 'R'],
        outputs=['Y', 'Y_h'],
        hidden_size=hidden,
    )
    model = make_model_from_nodes(
        nodes=node,
        initializers={
            'W': np.random.randn(1, 3 * hidden, input_size).astype(np.float32),
            'R': np.random.randn(1, 3 * hidden, hidden).astype(np.float32),
        },
        inputs_example={'X': x},
        opset_version=13,
    )
    with pytest.raises(NotImplementedError, match='linear_before_reset'):
        convert(model)
