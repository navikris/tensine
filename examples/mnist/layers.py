import torch
import numpy as np
import tensine as ts

from typing import Optional


class TsLinear:
    def __init__(
        self,
        weight_tensor: ts.Tensor | np.ndarray | torch.Tensor,
        bias_tensor: Optional[ts.Tensor | np.ndarray | torch.Tensor]=None,
    ):
        if isinstance(weight_tensor, np.ndarray):
            self.weights = ts.Tensor(weight_tensor)
        elif isinstance(weight_tensor, torch.Tensor):
            self.weights = ts.Tensor(weight_tensor.detach().numpy())
        elif isinstance(weight_tensor, ts.Tensor):
            self.weights = weight_tensor
        else:
            raise TypeError("Unsupported weight tesnor format")
        self.weights = self.weights.transpose(0, 1) # (out_feat, in_feat) -> (in_feat, out_feat)

        self.bias = None
        if bias_tensor is not None:
            if isinstance(bias_tensor, np.ndarray):
                self.bias = ts.Tensor(bias_tensor)
            elif isinstance(bias_tensor, torch.Tensor):
                self.bias = ts.Tensor(bias_tensor.detach().numpy())
            elif isinstance(bias_tensor, ts.Tensor):
                self.bias = bias_tensor
            else:
                raise TypeError("Unsupported bias tesnor format")
            # TODO: Add some sort of broadcasting since add kernels dont do broadcast internally
            self.bias = self.bias.reshape((1, self.bias.shape()[0]))

    def __call__(self, x: ts.Tensor):
        out = ts.matmul(x, self.weights)
        if self.bias is not None:
            out = ts.add(out, self.bias)
        return out


class TsMaxpool2d:
    def __init__(
        self,
        kernel_size: int | tuple[int, int],
        stride: Optional[int | tuple[int, int]]=None,
        padding: Optional[int | tuple[int, int]]=0
    ):
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size, kernel_size]
        if isinstance(stride, int):
            stride = [stride, stride]
        if isinstance(padding, int):
            padding = [padding, padding]

        if stride is None:
            stride = kernel_size

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def __call__(self, x):
        return ts.maxpool2d(x, self.kernel_size, self.stride, self.padding)
