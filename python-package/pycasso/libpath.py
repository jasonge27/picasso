# coding: utf-8
"""Find the path to picasso dynamic library files."""

import os
import platform
import sys

class PicassoLibraryNotFound(Exception):
    """Error thrown by when picasso is not found"""
    pass

def find_lib_path(update = False):
    """Find the path to picasso dynamic library files.

    :return: List of all found library path to picasso
    :rtype: list(string)
    """
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    # make pythonpack hack: copy this directory one level upper for setup.py
    if update:
        dll_path = [curr_path, os.path.join(curr_path, '../../lib/'),
                    os.path.join(curr_path, './lib/'),
                    os.path.join(sys.prefix, 'picasso')]
        if sys.platform == 'win32':
            if platform.architecture()[0] == '64bit':
                dll_path.append(os.path.join(curr_path, '../../windows/x64/Release/'))
                # hack for pip installation when copy all parent source directory here
                dll_path.append(os.path.join(curr_path, './windows/x64/Release/'))
            else:
                dll_path.append(os.path.join(curr_path, '../../windows/Release/'))
                # hack for pip installation when copy all parent source directory here
                dll_path.append(os.path.join(curr_path, './windows/Release/'))
    else:
        dll_path = [curr_path, os.path.join(curr_path, '../../lib/'),
                    os.path.join(curr_path, '../lib/'),
                    os.path.join(curr_path, './lib/'),
                    os.path.join(sys.prefix, 'picasso')]

    if sys.platform == 'win32':
        dll_path = [os.path.join(p, 'picasso.dll') for p in dll_path]
    elif sys.platform.startswith('linux'):
        dll_path = [os.path.join(p, 'libpicasso.so') for p in dll_path]
    elif sys.platform == 'darwin':
        dll_path = [os.path.join(p, 'libpicasso.dylib') for p in dll_path]

    lib_path = [p for p in dll_path if os.path.exists(p) and os.path.isfile(p)]

    if not lib_path and not update:
        print('Library file does not exist. There is no need to be updated!')
        return lib_path
    # From github issues, most of installation errors come from machines w/o compilers
    if not lib_path and not os.environ.get('PICASSO_BUILD_DOC', False):
        raise PicassoLibraryNotFound(
            'Cannot find Picasso Library in the candidate path, ' +
            'did you install compilers and make the project in root path?\n'
            'List of candidates:\n' + ('\n'.join(dll_path)))

    return lib_path
