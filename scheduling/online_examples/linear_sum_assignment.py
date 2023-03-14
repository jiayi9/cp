import numpy as np

x, y = np.meshgrid(np.arange(2), np.arange(3))

x.ravel()

xx = np.array([[[[[[1,2], [6,7]], [[8,9], [9,0]]]]]])
xx
xx.ravel()
from ortools.graph.python import linear_sum_assignment