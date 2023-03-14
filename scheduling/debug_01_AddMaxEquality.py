# https://stackoverflow.com/questions/75588142
# The answer is No. They are not compatible.

from ortools.sat.python import cp_model

model = cp_model.CpModel()

x = model.NewBoolVar('x')
y = model.NewBoolVar('y')
z = model.NewBoolVar('z')

model.AddMaxEquality(z, [0, y]).OnlyEnforceIf(x)
model.Minimize(1)

solver = cp_model.CpSolver()
status = solver.Solve(model=model)

if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    print(f'x {solver.Value(x)}, y {solver.Value(y)}, z {solver.Value(z)}')
elif status == cp_model.INFEASIBLE:
    print("Infeasible")
elif status == cp_model.MODEL_INVALID:
    print("Model invalid")
else:
    print(status)


###################################################################################

from ortools.sat.python import cp_model

model = cp_model.CpModel()

x = model.NewBoolVar('x')
y = model.NewBoolVar('y')
z = model.NewBoolVar('z')

Max = model.NewBoolVar('max')
Min = model.NewBoolVar('min')

model.AddMaxEquality(Max, [0, y])
model.AddMinEquality(Min, [0, y])

model.Add(z == Max).OnlyEnforceIf(z)
model.Add(z == Min).OnlyEnforceIf(z.Not())

model.Minimize(1)

solver = cp_model.CpSolver()
status = solver.Solve(model=model)

if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    print(f'x {solver.Value(x)}, y {solver.Value(y)}, z {solver.Value(z)}')
elif status == cp_model.INFEASIBLE:
    print("Infeasible")
elif status == cp_model.MODEL_INVALID:
    print("Model invalid")
else:
    print(status)
