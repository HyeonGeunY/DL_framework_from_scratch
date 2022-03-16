import numpy as np
gpu_enable = False
try:
    import cupy as cp
    cupy = cp
except ImportError:
    gpu_enable = False
from dezero import Variable


def get_array_module(x):
    """
    Parameters
    --------------
    x: dezero.Variable or numpy.ndarray, cupy.ndarray
        Numpy 또는 Cupy 중 어떤 것을 사용할지 결정
    
    Returns
    -------------
    module: Cupy 또는 numpy
    """

    if isinstance(x, Variable):
        x = x.data
    
    if not gpu_enable:
        return np
    xp = cp.get_array_module(x)
    return xp


def as_numpy(x):
    """
    Convert to numpy.ndarray

    Parameters
    -------------
    x: numpy.ndarray, cupy.ndarray

    Returns
    -------------
    numpy.ndarray
    """
    if isinstance(x, Variable):
        x = x.data

    if np.isscalar(x):
        return np.array(x)
    elif isinstance(x, np.ndarray):
        return x
    return cp.asnumpy(x)


def as_cupy(x):
    """
    Convert to cupy.ndarray

    Parameters
    -------------
    x: numpy.ndarray, cupy.ndarray
        cupy.ndarray로 변환 가능한 임의의 객체
    
    Returns
    --------------
    cupy.ndarray
        변환된 array
            
    """

    if isinstance(x, Variable):
        x = x.data
    
    if not gpu_enable:
        raise Exception('Cupy cannot be loaded. Install CuPy')
    
    return cp.asarray(x)
