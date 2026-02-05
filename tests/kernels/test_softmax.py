import torch
import pytest
import numpy as np
import tensine as ts
import torch.nn.functional as F


@pytest.mark.parametrize(
    "input_shape, dim",
    (
        [(1, 2, 3, 4), 3],
        [    (22, 53), 1],
        [      (100,), 0],
        [(32, 10, 64), 2],
    )
)
@pytest.mark.parametrize("dtype",
    (
        np.float32,
        np.float64,
    )
)
def test_softmax(
    input_shape,
    dim,
    dtype
):
    # Creating the numpy random tenor
    numpy_input = np.random.rand(*input_shape).astype(dtype)

    # Reference output
    torch_input = torch.from_numpy(numpy_input)
    torch_output = F.softmax(torch_input, dim=dim)

    # Tensine output
    tensine_input = ts.Tensor(numpy_input)
    tensine_output = ts.softmax(tensine_input, dim=dim)

    # Compare results
    torch_numpy_out = torch_output.numpy()
    tensine_numpy_out = tensine_output.to_numpy()
    assert(np.allclose(torch_numpy_out,tensine_numpy_out))
