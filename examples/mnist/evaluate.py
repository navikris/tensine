import torch
import numpy as np
import tensine as ts

from tqdm import tqdm
from torchvision import datasets, transforms

from model import TinyMNIST
from weight_processing import extract_params


WEIGHTS_PATH = "examples/mnist/weights/mnist_pytorch.ckpt"


def evaluate_model(model, data_loader):
    correct = 0
    for data, target in tqdm(data_loader, desc="Model-Evaluation"):
        # Convert to numpy
        np_input = data.detach().cpu().numpy()

        # NCHW -> NHWC
        ts_input = ts.Tensor(np_input).permute((0, 2, 3, 1))

        output = model(ts_input)
        output_numpy = output.to_numpy()

        # Get per-sample predictions
        pred = np.argmax(output_numpy, axis=1)

        correct += (pred == target.numpy()).sum()

    total = len(data_loader.dataset)
    accuracy = 100. * correct / total
    print('\nTest set: Accuracy: {}/{} ({:.2f}%)\n'.format(
        correct, total, accuracy))


def main():
    # The scaled mean and standard deviation of the MNIST dataset (precalculated)
    data_mean = 0.1307
    data_std = 0.3081

    # Convert input images to tensors and normalize
    transform=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((data_mean,), (data_std,))
        ]
    )

    # Get the MNIST data from torchvision
    dataset = datasets.MNIST(
        '../data',
        train=False,
        transform=transform,
        download=True
    )

    # Define the data loader that will handle fetching of data
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64
    )

    # Test the model
    state_dict = extract_params(WEIGHTS_PATH)
    model = TinyMNIST(state_dict)
    evaluate_model(model, data_loader)


if __name__ == "__main__":
    main()