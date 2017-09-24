# coding: utf-8
"""
Main Interface of the package
"""

import ctypes

from .libpath import find_lib_path

class PycassoError(Exception):
    """Error thrown by pycasso solver."""
    pass

def _load_lib():
    """Load picasso Library."""
    lib_path = find_lib_path()
    if not lib_path:
        PycassoError("Can not find picasso Library. Please install pycasso correctly.")
    lib = ctypes.cdll.LoadLibrary(lib_path[0])
    return lib

class Solver:
    """
    The picasso solver for GLM.
    """
    def __init__(self,x, y, lambdas = (), nlambda = 100, lambda_min_ratio = 0.05,
                 lambda_min = (), family = "gaussian", method = "l1",
                 type_gaussian = "naive", gamma = 3, df = (), standardize = True,
                 prec = 1e-7, max_ite = 1000,  verbose = False):
        """
        :param x: An n*m design matrix where n is the sample size and d is the data dimension.
        :param y: The n dimensional response vector.
        :param lambdas: A sequence of decresing positive values to control the regularization.
        :param nlambda: The number of values used in lambdas. Default value is 100.
        :param lambda_min_ratio: The smallest value for lambdas, as a fraction of the upper-bound of the regularization parameter.
        :param lambda_min:
        :param family:
        :param method:
        :param type_gaussian:
        :param gamma:
        :param df:
        :param standardize:
        :param prec:
        :param max_ite:
        :param verbose:
        """
        pass

    def __del__(self):
        pass

    def lalala(self):
        """lalala?wtf"""
        return 1