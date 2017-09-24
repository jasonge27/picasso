# coding: utf-8
"""
PICASSO: Penalized Generalized Linear Model Solver - Unleash the Power of Non-convex Penalty

Author: Jason(Jian) Ge, Haoming Jiang
Package Maintainer: Haoming Jiang

Main Interface of the package
"""

import os

def test():
    current_file = os.path.dirname(__file__)
    print(r"Picasso has been successfully imported!")
    print(r"Version: "+open(os.path.join(current_file, r'./VERSION')).read().strip())

