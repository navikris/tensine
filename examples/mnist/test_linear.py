import torch
import pytest
import numpy as np
import tensine as ts
import torch.nn.functional as F

from layers import TsLinear


@pytest.mark.parametrize(
    "batch_size, in_features, out_features",
    (
        [1, 196, 10],
    )
)
@pytest.mark.parametrize("dtype",
    (
        np.float32,
        np.float64,
    )
)
def test_linear(
    batch_size,
    in_features,
    out_features,
    dtype
):
    # Creating the numpy random tenor
    numpy_input = np.random.rand(batch_size, in_features).astype(dtype)
    numpy_weight = np.random.rand(out_features, in_features).astype(dtype)
    numpy_bias = np.random.rand(out_features).astype(dtype)

    # Reference output
    torch_input = torch.from_numpy(numpy_input)
    torch_weight = torch.from_numpy(numpy_weight)
    torch_bias = torch.from_numpy(numpy_bias)
    torch_output = F.linear(torch_input, torch_weight, torch_bias)

    # Tensine output
    ts_linear = TsLinear(numpy_weight, numpy_bias)
    tensine_input = ts.Tensor(numpy_input)
    tensine_output = ts_linear(tensine_input)

    # Compare results
    torch_numpy_out = torch_output.numpy()
    tensine_numpy_out = tensine_output.to_numpy()
    assert(np.allclose(torch_numpy_out,tensine_numpy_out))
