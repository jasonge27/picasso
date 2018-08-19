# coding: utf-8
"""Find the path to picasso dynamic library files."""

import os
import platform
import sys

class PicassoLibraryNotFound(Exception):
    """Error thrown by when picasso is not found"""
    pass

def find_lib_path():
    """Find the path to picasso dynamic library files.

    :return: List of all found library path to picasso
    :rtype: list(string)
    """
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    dll_path = [os.path.join(curr_path, './lib/')]

    if sys.platform == 'win32':
        dll_path = [os.path.join(p, 'picasso.dll') for p in dll_path] \
                    +[os.path.join(p, 'libpicasso.so') for p in dll_path]
    elif sys.platform.startswith('linux'):
        dll_path = [os.path.join(p, 'libpicasso.so') for p in dll_path]
    elif sys.platform == 'darwin':
        dll_path = [os.path.join(p, 'libpicasso.so') for p in dll_path] \
                    +[os.path.join(p, 'libpicasso.dylib') for p in dll_path]

    lib_path = [p for p in dll_path if os.path.exists(p) and os.path.isfile(p)]
    
    if not lib_path:
        print('Library file does not exist. Need to be updated!')
        return lib_path
    # From github issues, most of installation errors come from machines w/o compilers
    if not lib_path and not os.environ.get('PICASSO_BUILD_DOC', False):
        raise PicassoLibraryNotFound(
            'Cannot find Picasso Library in the candidate path, ' +
            'did you install compilers and make the project in root path?\n'
            'List of candidates:\n' + ('\n'.join(dll_path)))

    return lib_path
