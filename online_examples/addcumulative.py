from ortools.sat.python import cp_model

model = cp_model.CpModel()

a1 = model.NewIntVar(0, 100, 'a1')
a2 = model.NewIntVar(0, 100, 'a2')
a3 = model.NewIntVar(0, 100, 'a3')

x1 = model.NewIntervalVar(a1, 10, a1+10, 'x1')
x2 = model.NewIntervalVar(a2, 10, a2+10, 'x2')
x3 = model.NewIntervalVar(a3, 10, a3+10, 'x3')

# x1 = model.NewIntervalVar(start = a1, size = 3, end = a1+4, name = 'x1')
# x2 = model.NewIntervalVar(a2, 3, a2+3, 'x2')
# x3 = model.NewIntervalVar(a3, 3, a3+3, 'x3')


#model.AddNoOverlap([x1, x2, x3])

#model.AddCumulative([x1, x2, x3], [1,1,1], 1)

#model.AddCumulative([x1, x2, x3], [1,1,1], 2)

model.AddCumulative([x1, x2, x3], [1,1,1], 1)


solver = cp_model.CpSolver()
status = solver.Solve(model)
solver.parameters.log_search_progress = True
print(solver.Value(a1), solver.Value(a2), solver.Value(a3))

