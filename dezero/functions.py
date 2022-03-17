import numpy as np
import dezero
from dezero import utils, cuda
from dezero.core import Function, Variable, as_variable, as_array


class Sin(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.sin(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy * cos(x)
        return gx


def sin(x):
    return Sin()(x)


class Cos(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.cos(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy * -sin(x)
        return gx

def cos(x):
    return Cos()(x)


class Tanh(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.tanh(x)
        return y

    def backward(self, gy):
        y = self.outputs[0]() # weakref
        gx = gy * (1 - y * y)
        return gx


def tanh(x):
    return Tanh()(x)


class Exp(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.exp(x)
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * y
        return gx


def exp(x):
    return Exp()(x)


class Log(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.log(x)
        return y
    
    def backward(self, gy):
        x, = self.inputs
        gx = gy / x
        return gx


def log(x):
    return Log()(x)


# =======================
# sum
# =======================
class Sum(Function):
    def __init__(self, axis, keepdims):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        self.x_shape = x.shape
        y = x.sum(axis=self.axis, keepdims=self.keepdims)
        return y
    
    def backward(self, gy):
        gy = utils.reshape_sum_backward()