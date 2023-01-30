from ortools.algorithms import pywrapknapsack_solver

from ortools.linear_solver import pywraplp

solver = pywraplp.Solver.CreateSolver('SCIP')

from ortools.graph.python import max_flow