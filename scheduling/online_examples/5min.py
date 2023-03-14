import collections
from ortools.sat.python import cp_model

class VarArraySolutionPrinter(cp_model.CpSolverSolutionCallback):
    """Print intermediate solutions."""

    def __init__(self, variables):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__variables = variables
        self.__solution_count = 0

    def on_solution_callback(self):
        self.__solution_count += 1
        for v in self.__variables:
            print('%s=%i' % (v, self.Value(v)), end=' ')
        print()

    def solution_count(self):
        return self.__solution_count

##########################################################
model = cp_model.CpModel()
x1 = model.NewIntVar(0, 10, 'x1')
y = model.NewIntVar(0, 20, 'y')
model.Add(y==x1)
model.Maximize(y)
solver = cp_model.CpSolver()
solution_printer = VarArraySolutionPrinter([x1, y])
status = solver.Solve(model, solution_printer)

##########################################################

model = cp_model.CpModel()
# x1 = model.NewIntVarFromDomain(
#     cp_model.Domain.FromValues([1, 3, 4, 6]), 'x1'
# )
domain = cp_model.Domain.FromValues([1, 3, 4, 6])
x1 = model.NewIntVarFromDomain(domain,'x1')
y = model.NewIntVar(0, 20, 'y')
model.Add(y==x1)
model.Maximize(y)
solver = cp_model.CpSolver()
solution_printer = VarArraySolutionPrinter([x1, y])
status = solver.Solve(model, solution_printer)



##########################################################

model = cp_model.CpModel()

x1 = model.NewBoolVar('x1')
x2 = model.NewBoolVar('x2')
y = model.NewIntVar(0,20, 'y')
model.AddBoolOr([])
model.Add(y == x1 + x2)
model.Maximize(y)
solver = cp_model.CpSolver()
solution_printer = VarArraySolutionPrinter([x1, x2, y])
status = solver.Solve(model, solution_printer)


##########################################################

#A channeling constraint
# Channeling is usually implemented using half-reified linear constraints:
# one constraint implies another (a → b),
# but not necessarily the other way around (a ← b).

# if x < 0, y = 0
# else, y = 10 -x

#  b ->     y = 10 -x
# !b ->     y = 0

# Create the CP-SAT model.

if True:
    model = cp_model.CpModel()

    # Declare our two primary variables.
    x = model.NewIntVar(0, 10, 'x')
    y = model.NewIntVar(0, 10, 'y')

    # Declare our intermediate boolean variable.
    b = model.NewBoolVar('b')

    # Implement b == (x >= 5).
    model.Add(x >= 5).OnlyEnforceIf(b)
    model.Add(x < 5).OnlyEnforceIf(b.Not())

    # Create our two half-reified constraints.
    # First, b implies (y == 10 - x).
    model.Add(y == 10 - x).OnlyEnforceIf(b)
    # Second, not(b) implies y == 0.
    model.Add(y == 0).OnlyEnforceIf(b.Not())

    # Search for x values in increasing order.
    model.AddDecisionStrategy([x], cp_model.CHOOSE_FIRST,
                              cp_model.SELECT_MIN_VALUE)

    # Create a solver and solve with a fixed search.
    solver = cp_model.CpSolver()

    # Force the solver to follow the decision strategy exactly.
    solver.parameters.search_branching = cp_model.FIXED_SEARCH
    # Enumerate all solutions.
    solver.parameters.enumerate_all_solutions = True

    # Search and print out all solutions.
    solution_printer = VarArraySolutionPrinter([x, y, b])
    solver.Solve(model, solution_printer)



from ortools.sat.python import cp_model
solver = cp_model.CpSolver()


#https://developers.google.com/optimization/cp/channeling

model = cp_model.CpModel()
x = model.NewIntVar(0, 10, 'x')
b = model.NewBoolVar('b')
y = model.NewIntVar(0, 10, 'y')
solution_printer = VarArraySolutionPrinter([x, b, y])
model.Add(x >= 5).OnlyEnforceIf(b)
model.Add(x < 5).OnlyEnforceIf(b.Not())
model.Add(y == b*5 - x)
model.Maximize(y)
status = solver.Solve(model, solution_printer)



model = cp_model.CpModel()
x = model.NewIntVar(0, 10, 'x')
b = model.NewBoolVar('b')
y = model.NewIntVar(0, 10, 'y')
solution_printer = VarArraySolutionPrinter([x, b, y])
model.Add(x >= 5).OnlyEnforceIf(b)
model.Add(x < 5).OnlyEnforceIf(b.Not())
model.Add(y == b*5 - x)
model.Maximize(y)
status = solver.Solve(model, solution_printer)



model = cp_model.CpModel()
a = model.NewBoolVar('a')
b = model.NewBoolVar('b')
c = model.NewBoolVar('c')
x = model.NewIntVar(0, 10, 'x')
y = model.NewIntVar(0, 10, 'y')
z = model.NewIntVar(0, 10, 'z')

solution_printer = VarArraySolutionPrinter([x,y,z,a,b,c])

model.AddBoolOr(a,b,c)
model.AddBoolAnd(a,b,c)
#model.AddBoolAnd(x,b,c)
#TypeError: TypeError: x is not a boolean variable


model.Maximize(c)
status = solver.Solve(model, solution_printer)

# if x < 0, y = 0
# else, y = 10 -x

class Student:
    def __init__(self, name, age):
        self.name = name
        self.age = age
        self.__doc__='yyyy'

s1 = Student('Mike', 12)
s2 = s1
print(s2.name)
s1.name = 'Lucy'
print(s2.name)
from copy import deepcopy
s3 = deepcopy(s1)
print(s3.name)
s1.name = 'OOO'
print(s3.name)



vars(Student)

vars(list)

x = 1
y = x
print(y)
x = 2
print(y)

