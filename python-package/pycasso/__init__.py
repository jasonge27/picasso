# coding: utf-8
"""
PICASSO: Penalized generalized linear model solver.

:Authors: Jason Ge, Haoming Jiang, Xingguo Li, Tuo Zhao

"""

from __future__ import absolute_import

import os

def test():
    """Show welcome information."""
    current_file = os.path.dirname(__file__)
    print(r"Picasso has been successfully imported!")
    print(r"Version: "+open(os.path.join(current_file, r'./VERSION')).read().strip())

__all__ = ["core"]

from .core import Solver
