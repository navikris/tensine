import torch
import tensine as ts

from pathlib import Path
from collections import OrderedDict


def preprocess_weights(d):
    if isinstance(d, (dict, OrderedDict)):
        parameters = {}
        for k, v in d.items():
            parameters[k] = preprocess_weights(v)
        return parameters
    elif isinstance(d, torch.Tensor):
        return ts.Tensor(d.detach().numpy())
    return


def extract_params(ckpt_path: Path | str | None=None):
    if ckpt_path is None:
        ckpt_path = Path("examples/mnist/weights/mnist_cnn_pytorch.ckpt")
    else:
        ckpt_path = Path(ckpt_path)

    state_dict = torch.load(ckpt_path, map_location="cpu")
    return preprocess_weights(state_dict)
