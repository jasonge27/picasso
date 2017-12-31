#TODO: add more concrete examples
from pycasso import *
x = [[1, 2, 3, 4, 5, 0], [3, 4, 1, 7, 0, 1], [5, 6, 2, 1, 4, 0]]
y = [3.1, 6.9, 11.3]
s = core.Solver(x, y)
s.train()
s.predict()