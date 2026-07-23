# coding: utf-8
"""Python interface for PICASSO sparse regularization-path solvers.

:Author: Jason Ge, Xingguo Li, Haoming Jiang, Tuo Zhao
:Maintainer: Tuo Zhao <tourzhao@gatech.edu>
"""

from __future__ import absolute_import

import os

def test():
    """Show welcome information."""
    current_file = os.path.dirname(__file__)
    print(r"Picasso has been successfully imported!")
    print(r"Version: "+open(os.path.join(current_file, r'./VERSION')).read().strip())

from .core import PycassoError, Solver

__all__ = ["Solver", "PycassoError", "test"]
