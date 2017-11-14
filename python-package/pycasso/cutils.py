# coding: utf-8
"""
Implemented some utilities for calling C functions from library via ``ctypes``
"""
import ctypes
from numpy.ctypeslib import ndpointer

class DoubleArrayCType:
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
        print("Warning: turning lists/tuples to ctype object, which is an individual copy the array.")
        val = ((ctypes.c_double)*len(param))(*param)
        return val

    from_tuple = from_list

    # Cast from a numpy array
    def from_ndarray(self, param):
        if param.dtype != 'double':
            raise TypeError('must be an array of doubles')
        if not param.flags['C_CONTIGUOUS']:
            raise TypeError('Must be stored in contiguous space. The data set might be too big')
        # return ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")
        return param.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

CDoubleArray = DoubleArrayCType()


class IntArrayCType:
    """
    Define a special type for the ``int32 *`` argument.
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
    # def from_array(self, param):
    #     if param.typecode != 'i':
    #         raise TypeError('must be an array of ints')
    #     ptr, _ = param.buffer_info()
    #     return ctypes.cast(ptr, ctypes.POINTER(ctypes.c_int))

    # Cast from lists/tuples
    def from_list(self, param):
        print("Warning: turning lists/tuples to ctype object, which is an individual copy the array.")
        val = ((ctypes.c_int)*len(param))(*param)
        return val

    from_tuple = from_list

    # Cast from a numpy array
    def from_ndarray(self, param):
        if param.dtype != 'int32':
            raise TypeError('must be an array of int32s')
        if not param.flags['C_CONTIGUOUS']:
            raise TypeError('Must be stored in contiguous space. The data set might be too big')
        # return ndpointer(ctypes.c_int, flags="C_CONTIGUOUS")
        return param.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

CIntArray = IntArrayCType()