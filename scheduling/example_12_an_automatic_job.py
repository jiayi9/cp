from ortools.sat.python import cp_model

# Initiate
model = cp_model.CpModel()

'''
task    product     type
1       A           TYPE_3
'''

# 1. Data

tasks = {1}
products = {'A', 'B'}
task_to_product = {1: 'A'}
task_to_type = {1: 'TYPE_3'}
processing_time = {'A': 3, 'B': 1}
max_time = 10
breaks = {(0, 1), (2, 10)}


# 2. Decision Variables
var_task_starts = {
    task: model.NewIntVar(0, max_time, f"task_{task}_start") for task in tasks
}

var_task_ends = {
    task: model.NewIntVar(0, max_time, f"task_{task}_end") for task in tasks
}

var_task_intervals = {
    task: model.NewIntervalVar(
        var_task_starts[task],
        processing_time[task_to_product[task]],
        var_task_ends[task],
        name=f"interval_t{task}"
    )
    for task in tasks
}

var_task_intervals_autojobs = {
    task: model.NewIntervalVar(
        var_task_starts[task],
        1,
        var_task_starts[task] + 1,
        name=f"interval_t{task}"
    )
    for task in tasks
    if task_to_type[task] == 'TYPE_3'
}



# Add break time
variables_breaks = {
    (start, end): model.NewFixedSizeIntervalVar(start=start, size=end-start, name='a_break')
    for (start, end) in breaks
}

intervals = list(var_task_intervals_autojobs.values()) + list(variables_breaks.values())

# task, resource reduction for breaks
demands = [1] + [1]*len(breaks)

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
