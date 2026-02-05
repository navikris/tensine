import torch
import pytest
import numpy as np
import tensine as ts
import torch.nn.functional as F


@pytest.mark.parametrize(
    "input_shape, kernel_size",
    (
        [   (1, 16, 16, 64), 3],
        [    (1, 2, 2, 512), 2],
        [(32, 32, 32, 1024), 7],
    )
)
@pytest.mark.parametrize("stride", [1, 2, 3])
@pytest.mark.parametrize("padding", [0, 1])
@pytest.mark.parametrize("dtype",
    (
        np.float32,
        np.float64,
    )
)
def test_maxpool2d(
    input_shape,
    kernel_size,
    stride,
    padding,
    dtype,
):
    # Creating the numpy random tenor
    numpy_input = np.random.rand(*input_shape).astype(dtype)

    # Reference output
    torch_input = torch.from_numpy(numpy_input)
    torch_input = torch_input.permute(0, 3, 1, 2) # NHWC -> NCHW
    torch_output = F.max_pool2d(
        torch_input,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding
    )
    torch_output = torch_output.permute(0, 2, 3, 1) # NCHW -> NHWC

    # Tensine output
    tensine_input = ts.Tensor(numpy_input)
    tensine_output = ts.maxpool2d(
        tensine_input,
        kernel_size=(kernel_size, kernel_size),
        stride=(stride, stride),
        padding=(padding, padding)
    )

    # Compare results
    torch_numpy_out = torch_output.numpy()
    tensine_numpy_out = tensine_output.to_numpy()
    assert(np.allclose(torch_numpy_out,tensine_numpy_out))
