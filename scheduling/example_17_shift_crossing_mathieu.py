from ortools.sat.python import cp_model

# Initiate
model = cp_model.CpModel()

tasks = {1}
processing_times = {1: 3}
max_time = 10
breaks = {(0, 2)}
shifts = {1, 2}
shift_starts = {1:0, 2:4}
shift_ends = {1:4, 2:8}


var_task_starts = {
    task: model.NewIntVar(0, max_time, f"task_{task}_start") for task in tasks
}
var_task_ends = {
    task: model.NewIntVar(0, max_time, f"task_{task}_end") for task in tasks
}

var_shift_task_presence = {
    (shift, task): model.NewBoolVar(f'task_{task}_is_in_shift_{shift}')
    for shift in shifts for task in tasks
}

for task in tasks:
    model.Add(sum(var_shift_task_presence[shift, task] for shift in shifts) == 1 )

    for shift in shifts:
        model.Add(var_task_starts[task] >= shift_starts[shift]).OnlyEnforceIf(
            var_shift_task_presence[shift, task]
        )
        model.Add(var_task_ends[task] <= shift_ends[shift]).OnlyEnforceIf(
            var_shift_task_presence[shift, task]
        )


var_task_intervals = {
    task: model.NewIntervalVar(
        var_task_starts[task], processing_times[task], var_task_ends[task], name=f"interval_t{task}"
    ) for task in tasks
}

# Add break time
var_break_intervals = {
    (start, end): model.NewFixedSizeIntervalVar(start=start, size=end-start, name='a_break')
    for (start, end) in breaks
}


intervals = list(var_task_intervals.values()) + list(var_break_intervals.values())

demands = [1]*len(tasks) + [1]*len(breaks)

model.AddCumulative(intervals=intervals, demands=demands, capacity=1)


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
