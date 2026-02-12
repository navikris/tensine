import tensine as ts
from layers import TsLinear, TsMaxpool2d


class TinyMNIST:
    def __init__(self, parameters):
        self.pool = TsMaxpool2d(kernel_size=2, stride=2, padding=0)
        self.fc = TsLinear(parameters["fc.weight"], parameters["fc.bias"])

    def __call__(self, x):
        x = self.pool(x)
        shape = x.shape()
        x = x.reshape((shape[0], shape[1] * shape[2]))
        x = self.fc(x)
        return ts.softmax(x, dim=1)
