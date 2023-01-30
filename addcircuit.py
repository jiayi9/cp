from ortools.sat.python import cp_model

model = cp_model.CpModel()

a1 = model.NewBoolVar("1")
a2 = model.NewBoolVar("2")
a3 = model.NewBoolVar("3")
a4 = model.NewBoolVar("4")
a5 = model.NewBoolVar("5")
a6 = model.NewBoolVar("6")
a7 = model.NewBoolVar("7")
a8 = model.NewBoolVar("8")
a9 = model.NewBoolVar("9")
a10 = model.NewBoolVar("10")


# arc1 = (0, 1, a1)
# arc2 = (1, 2, a2)
# arc3 = (2, 0, a3)
# model.AddCircuit([arc1, arc2, arc3])
# solver = cp_model.CpSolver()
# status = solver.Solve(model)
# print(solver.Value(a1), solver.Value(a2), solver.Value(a3))


arc1 = (0, 1, a1)
arc2 = (0, 2, a2)
arc3 = (1, 0, a3)
arc4 = (1, 2, a4)
arc5 = (2, 0, a5)
arc6 = (2, 1, a6)
arc7 = (0, 0, a7)
arc8 = (1, 1, a8)
arc9 = (2, 2, a9)
arc10 = (2, 2, a10)


#model.AddCircuit([arc1, arc2, arc3, arc4, arc5, arc6,  arc7, arc8, arc9, arc10])
#model.AddCircuit([arc1, arc2, arc3, arc4, arc5, arc6,  arc7, arc8])# arc9, arc10])

model.AddCircuit([arc1, arc2, arc3, arc4, arc5, arc6,  arc7, arc9, arc10])

#model.Add(a1 == 1)
# model.Add(a9 == 1)
# model.Add(a1 == 1)


solver = cp_model.CpSolver()
status = solver.Solve(model)

print(solver.Value(a1), solver.Value(a2), solver.Value(a3),
      solver.Value(a4), solver.Value(a5), solver.Value(a6),
      solver.Value(a7), solver.Value(a8), solver.Value(a9),
      )



from ortools.sat.python import cp_model
model = cp_model.CpModel()
a1 = model.NewBoolVar("1")
a2 = model.NewBoolVar("2")
a3 = model.NewBoolVar("3")
a4 = model.NewBoolVar("4")
a5 = model.NewBoolVar("5")
a6 = model.NewBoolVar("6")
arc1 = (0, 1, a1)
arc2 = (20, 20, a2)
arc3 = (1, 0, a3)
model.AddCircuit([arc1, arc2, arc3])
solver = cp_model.CpSolver()
status = solver.Solve(model)
print(status)
print(solver.Value(a1), solver.Value(a2), solver.Value(a3),
      solver.Value(a4), solver.Value(a5), solver.Value(a6))

