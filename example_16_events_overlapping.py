from ortools.sat.python import cp_model

model = cp_model.CpModel()
max_time = 10
tasks = {1, 2}

# 1. Data

var_task_starts = {
    task: model.NewIntVar(0, max_time, f"task_{task}_start") for task in tasks
}

var_task_ends = {
    task: model.NewIntVar(0, max_time, f"task_{task}_end") for task in tasks
}

# overlap
# model.Add(var_task_starts[1] == 1)
# model.Add(var_task_ends[1] == 5)
# model.Add(var_task_starts[2] == 3)
# model.Add(var_task_ends[2] == 7)

# No overlap
model.Add(var_task_starts[1] == 1)
model.Add(var_task_ends[1] == 3)
model.Add(var_task_starts[2] == 5)
model.Add(var_task_ends[2] == 7)



var_overlap = {
    (t1, t2): model.NewBoolVar('')
    for t1 in tasks
    for t2 in tasks
    if t1 != t2
}

var_overlap_duration = {
    (t1, t2): model.NewIntVar(0, max_time, '')
    for t1 in tasks
    for t2 in tasks
    if t1 != t2
}

var_start_earlier_than_start = {
    (t1, t2): model.NewBoolVar('')
    for t1 in tasks
    for t2 in tasks
    if t1 != t2
}

var_end_later_than_start = {
    (t1, t2): model.NewBoolVar('')
    for t1 in tasks
    for t2 in tasks
    if t1 != t2
}

for t1 in tasks:
    for t2 in tasks:
        if t1 == t2:
            continue
        model.Add(var_task_starts[t1] <= var_task_starts[t2]).OnlyEnforceIf(var_start_earlier_than_start[t1, t2])
        model.Add(var_task_starts[t1] > var_task_starts[t2]).OnlyEnforceIf(var_start_earlier_than_start[t1, t2].Not())

        model.Add(var_task_ends[t1] > var_task_starts[t2]).OnlyEnforceIf(var_end_later_than_start[t1, t2])
        model.Add(var_task_ends[t1] <= var_task_starts[t2]).OnlyEnforceIf(var_end_later_than_start[t1, t2].Not())

        model.AddMultiplicationEquality(
            var_overlap[t1, t2],
            [var_start_earlier_than_start[t1, t2], var_end_later_than_start[t1, t2]]
        )

        model.Add(var_overlap_duration[t1, t2] == var_task_ends[t1] - var_task_starts[t2]).OnlyEnforceIf(var_overlap[t1, t2])
        model.Add(var_overlap_duration[t1, t2] == 0).OnlyEnforceIf(var_overlap[t1, t2].Not())




# 3. Objectives
make_span = model.NewIntVar(0, max_time, "make_span")
model.AddMaxEquality(
    make_span,
    [var_task_ends[task] for task in tasks]
)
model.Minimize(make_span)


# 4. Solve
solver = cp_model.CpSolver()
status = solver.Solve(model=model)


# 5. Results
if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:

    count = 0
    for t1 in tasks:
        for t2 in tasks:
            if t1 == t2:
                continue
            if solver.Value(var_overlap[t1,t2]):
                count = count + 1
                print(f'Task{t1} starts earlier and overlap Task{t2} for a duration of '
                      f'{solver.Value(var_overlap_duration[t1, t2])} units')
    if count == 0:
        print("No overlapped tasks at all")

    print('===========================  TASKS SUMMARY  ===========================')
    for task in tasks:
        print(f'Task {task} ',
              solver.Value(var_task_starts[task]), solver.Value(var_task_ends[task]),
              )

    print('Make-span:', solver.Value(make_span))

elif status == cp_model.INFEASIBLE:
    print("Infeasible")
elif status == cp_model.MODEL_INVALID:
    print("Model invalid")
else:
    print(status)
