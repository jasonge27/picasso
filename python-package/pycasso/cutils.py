# coding: utf-8
"""
Implemented some utilities for calling C functions from library via ``ctypes``
"""
import ctypes


class DoubleArrayType:
    """
    Define a special type for the ``double *`` argument.
    It is used for converting ``param`` to a proper ctype object.
    """
    def from_param(self, param):
        """
        Provide a unified interface for different types for converting ``param`` to a proper ctype object.
        :param param: A python array.
        :type param: array.array, lists, tuples or numpy
        :return: A proper ctype object which can be feed to C function
        """
        typename = type(param).__name__
        if hasattr(self, 'from_' + typename):
            return getattr(self, 'from_' + typename)(param)
        elif isinstance(param, ctypes.Array):
            return param
        else:
            raise TypeError("Can't convert %s" % typename)

    # Cast from array.array objects
    def from_array(self, param):
        if param.typecode != 'd':
            raise TypeError('must be an array of doubles')
        ptr, _ = param.buffer_info()
        return ctypes.cast(ptr, ctypes.POINTER(ctypes.c_double))

    # Cast from lists/tuples
    def from_list(self, param):
        val = ((ctypes.c_double)*len(param))(*param)
        return val

    from_tuple = from_list

    # Cast from a numpy array
    def from_ndarray(self, param):
        return param.ctypes.data_as(ctypes.POINTER(ctypes.c_double))