# coding: utf-8
"""
PICASSO: Penalized Generalized Linear Model Solver - Unleash the Power of Non-convex Penalty

:Author: Jason Ge, Haoming Jiang
:Maintainer: Haoming Jiang <jianghm@gatech.edu>

"""

from __future__ import absolute_import

import os

def test():
    """Show welcome information."""
    current_file = os.path.dirname(__file__)
    print(r"Picasso has been successfully imported!")
    print(r"Version: "+open(os.path.join(current_file, r'./VERSION')).read().strip())

__all__ = ["core"]
