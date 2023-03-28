from ortools.sat.python import cp_model
model = cp_model.CpModel()
def get(x):
    return solver.Value(x)

#
x = model.NewBoolVar('x')
y = model.NewBoolVar('y')
model.AddBoolOr(x, y)
model.Minimize(x+y)
solver = cp_model.CpSolver()
status = solver.Solve(model=model)
print(get(x), get(y))

#
x = model.NewBoolVar('x')
y = model.NewBoolVar('y')
model.AddBoolAnd(x, y)
model.Minimize(x+y)
solver = cp_model.CpSolver()
status = solver.Solve(model=model)
print(get(x), get(y))

#
x = model.NewBoolVar('x')
y = model.NewBoolVar('y')
model.AddBoolXOr(x, y)
model.Minimize(x+y)
solver = cp_model.CpSolver()
status = solver.Solve(model=model)
print(get(x), get(y))


#
x = model.NewBoolVar('x')
y = model.NewBoolVar('y')
model.Add(x+y == 2)
model.Minimize(x+y)
solver = cp_model.CpSolver()
status = solver.Solve(model=model)
print(get(x), get(y))

#
x = model.NewBoolVar('x')
y = model.NewBoolVar('y')
model.Add(x+y == 1)
model.Minimize(x+y)
solver = cp_model.CpSolver()
status = solver.Solve(model=model)
print(get(x), get(y))

#
x = model.NewBoolVar('x')
y = model.NewBoolVar('y')
model.Add(x+y == 0)
model.Minimize(x+y)
solver = cp_model.CpSolver()
status = solver.Solve(model=model)
print(get(x), get(y))


#
model = cp_model.CpModel()
x = model.NewBoolVar('x')
y = model.NewBoolVar('y')
model.Add(x==1).OnlyEnforceIf(x.Not())
#model.Add(x==0).OnlyEnforceIf(x.Not())
# model.Add(x==1).OnlyEnforceIf(x)
model.Add(x==0).OnlyEnforceIf(x)
#model.Add(y==1).OnlyEnforceIf(x)
model.Minimize(x)
solver = cp_model.CpSolver()
status = solver.Solve(model=model)
print(get(x))


#

model = cp_model.CpModel()
x_is_between_5_and_10 = model.NewBoolVar('x_is_between_5_and_10')
x = model.NewIntVar(0, 100, 'x')
model.Add(x == 7)
model.Add(x_is_between_5_and_10 == 1).OnlyEnforceIf(5 <= x).OnlyEnforceIf(x <= 10)
solver = cp_model.CpSolver()
status = solver.Solve(model=model)
print('x', get(x))
print('x_is_between_5_and_10', get(x_is_between_5_and_10))









model = cp_model.CpModel()
x_is_between_5_and_10 = model.NewBoolVar('x_is_between_5_and_10')
x_is_no_less_than_5 = model.NewBoolVar('x_is_no_less_than_5')
x_is_no_more_than_10 = model.NewBoolVar('x_is_no_more_than_10')
x = model.NewIntVar(0, 100, 'x')
model.Add(x == 7)

model.Add(x_is_no_less_than_5 == x >= 5)


# model.Add(x_is_no_less_than_5 == 1).OnlyEnforceIf(x>=5)
# model.Add(x_is_no_more_than_10 == 1).OnlyEnforceIf(x <= 10)

model.Add(x_is_between_5_and_10 == 1).OnlyEnforceIf(5 <= x).OnlyEnforceIf(x <= 10)
solver = cp_model.CpSolver()
status = solver.Solve(model=model)
print('x', get(x))
print('x_is_between_5_and_10', get(x_is_between_5_and_10))





##########################################

from ortools.sat.python import cp_model
model = cp_model.CpModel()
x_is_greater_than_5 = model.NewBoolVar('x_is_greater_than_5')
x = model.NewIntVar(0, 100, 'x')
model.Add(x == 7)
model.Add(x >= 5).OnlyEnforceIf(x_is_greater_than_5)
model.Add(x < 5).OnlyEnforceIf(x_is_greater_than_5.Not())
solver = cp_model.CpSolver()
status = solver.Solve(model=model)
print('x', solver.Value(x))
print('x_is_greater_than_5', solver.Value(x_is_greater_than_5))



from ortools.sat.python import cp_model
model = cp_model.CpModel()
x = model.NewIntVar(0, 100, 'x')
x_is_between_5_and_10 = model.NewBoolVar('x_is_between_5_and_10')
model.Add(x >= 5).OnlyEnforceIf(x_is_between_5_and_10)
#model.Add(x <= 10).OnlyEnforceIf(x_is_between_5_and_10)
model.Add(x < 10).OnlyEnforceIf(x_is_between_5_and_10.Not())
#model.Add(x >10).OnlyEnforceIf(x_is_greater_than_5.Not())
# This gives invalid
model.Add(x == 3)
model.Add(x_is_between_5_and_10 == 1)
solver = cp_model.CpSolver()
status = solver.Solve(model=model)
print(status)
if status == 1 or status == 4:
    print('x', solver.Value(x))
    print('x_is_greater_than_5', solver.Value(x_is_between_5_and_10))




from ortools.sat.python import cp_model
model = cp_model.CpModel()
x = model.NewIntVar(0, 100, 'x')
x_is_between_5_and_10 = model.NewBoolVar('x_is_between_5_and_10')
model.Add(x >= 5).OnlyEnforceIf(x_is_between_5_and_10)
#model.Add(x <= 10).OnlyEnforceIf(x_is_between_5_and_10)
model.Add(x < 10).OnlyEnforceIf(x_is_between_5_and_10.Not())
#model.Add(x >10).OnlyEnforceIf(x_is_greater_than_5.Not())
#model.Add(x == 3)
model.Add(x_is_between_5_and_10 == 1)
solver = cp_model.CpSolver()
status = solver.Solve(model=model)
print(status)
if status == 1 or status == 4:
    print('x', solver.Value(x))
    print('x_is_greater_than_5', solver.Value(x_is_between_5_and_10))




from ortools.sat.python import cp_model
model = cp_model.CpModel()
x = model.NewIntVar(0, 100, 'x')
x_is_between_5_and_10 = model.NewBoolVar('x_is_between_5_and_10')
model.Add(x >= 5).OnlyEnforceIf(x_is_between_5_and_10)
model.Add(x <= 10).OnlyEnforceIf(x_is_between_5_and_10)

model.Add(x < 5).OnlyEnforceIf(x_is_between_5_and_10.Not())
model.Add(x >10).OnlyEnforceIf(x_is_between_5_and_10.Not())

model.Add(x == 3)
# model.Add(x_is_between_5_and_10 == 0)

solver = cp_model.CpSolver()
status = solver.Solve(model=model)
print(status)
if status == 1 or status == 4:
    print('x', solver.Value(x))
    print('x_is_greater_than_5', solver.Value(x_is_between_5_and_10))





from ortools.sat.python import cp_model
model = cp_model.CpModel()
x = model.NewIntVar(0, 100, 'x')
x_is_between_5_and_10 = model.NewBoolVar('5<x<10')
x_greater_than_5 = model.NewBoolVar('5<x')
x_less_than_10 = model.NewBoolVar('x<10')


model.Add(x > 5).OnlyEnforceIf(x_greater_than_5)
model.Add(x <= 5).OnlyEnforceIf(x_greater_than_5.Not())

model.Add(x < 10).OnlyEnforceIf(x_less_than_10)
model.Add(x >= 10).OnlyEnforceIf(x_less_than_10.Not())

model.Add(x_is_between_5_and_10==x_greater_than_5*x_less_than_10)
model.AddMultiplicationEquality(x_is_between_5_and_10, )

model.Add(x == 3)
# model.Add(x_is_between_5_and_10 == 0)

solver = cp_model.CpSolver()
status = solver.Solve(model=model)
print(status)
if status == 1 or status == 4:
    print('x', solver.Value(x))
    print('x_is_greater_than_5', solver.Value(x_is_between_5_and_10))









from ortools.sat.python import cp_model
model = cp_model.CpModel()
x_is_between_5_and_10 = model.NewBoolVar('5<x<10')
x = model.NewIntVar(0, 100, 'x')

model.AddLinearConstraint(x, 5, 10).OnlyEnforceIf(x_is_between_5_and_10)
model.AddLinearExpressionInDomain(
    x,
    cp_model.Domain.FromIntervals([[0, 4], [11, 100]])
).OnlyEnforceIf(x_is_between_5_and_10.Not())

model.Add(x == 3)
solver = cp_model.CpSolver()
status = solver.Solve(model=model)
print(status)
if status == 1 or status == 4:
    print('x', solver.Value(x))
    print('x_is_greater_than_5', solver.Value(x_is_between_5_and_10))





from ortools.sat.python import cp_model
model = cp_model.CpModel()
x = model.NewIntVar(0, 100, 'x')
x_is_between_5_and_10 = model.NewBoolVar('5<x<10')
x_greater_than_5 = model.NewBoolVar('5<x')
x_less_than_10 = model.NewBoolVar('x<10')

model.Add(x > 5).OnlyEnforceIf(x_greater_than_5)
model.Add(x <= 5).OnlyEnforceIf(x_greater_than_5.Not())

model.Add(x < 10).OnlyEnforceIf(x_less_than_10)
model.Add(x >= 10).OnlyEnforceIf(x_less_than_10.Not())

model.AddMultiplicationEquality(x_is_between_5_and_10, [x_greater_than_5, x_less_than_10])

model.Add(x_is_between_5_and_10 == 1)

solver = cp_model.CpSolver()
status = solver.Solve(model=model)
print(status)
if status == 1 or status == 4:
    print('x', solver.Value(x))
    print('x_is_greater_than_5', solver.Value(x_is_between_5_and_10))
