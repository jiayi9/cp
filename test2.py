
from ortools.sat.python import cp_model

model = cp_model.CpModel()

x = model.NewIntVar(0, 2, 'ooo')
y = model.NewIntVar(0, 2, 'ooo')

model.Add(x != 0)
model.Add(x != 1)
model.Add(y >= x)

li = [x,y]

model.Add(sum(li) == 4)



solver = cp_model.CpSolver()

status = solver.Solve(model)

if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    print('x = %i' % solver.Value(x))
    print('y = %i' % solver.Value(y))

else:
    print('No solution found.')
