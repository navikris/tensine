import torch
import pytest
import numpy as np
import tensine as ts


@pytest.mark.parametrize(
    "input_shape",
    (
        (6,),
        (9, 7),
        (11, 2, 4),
        (10, 32, 32, 512),
    )
)
@pytest.mark.parametrize("dtype",
    (
        np.float32,
        np.float64,
    )
)
def test_eltwise_add(
    input_shape,
    dtype,
):
    # Creating the numpy random tenor
    numpy_input_1 = np.random.rand(*input_shape).astype(dtype)
    numpy_input_2 = np.random.rand(*input_shape).astype(dtype)

    # Reference output
    torch_input_1 = torch.from_numpy(numpy_input_1)
    torch_input_2 = torch.from_numpy(numpy_input_2)
    torch_output = torch.add(torch_input_1, torch_input_2)

    # Tensine output
    tensine_input_1 = ts.Tensor(numpy_input_1)
    tensine_input_2 = ts.Tensor(numpy_input_2)
    tensine_output = ts.add(tensine_input_1, tensine_input_2)

    # Compare results
    torch_numpy_out = torch_output.numpy()
    tensine_numpy_out = tensine_output.to_numpy()
    assert(np.allclose(torch_numpy_out,tensine_numpy_out))
