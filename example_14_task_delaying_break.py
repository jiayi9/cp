from ortools.sat.python import cp_model

# Initiate
model = cp_model.CpModel()

'''
task    product     type
1       A           TYPE_4
2       B           TYPE_4
'''

# 1. Data

tasks = {1, 2}
products = {'A'}
task_to_product = {1: 'A'}
task_to_type = {1: 'TYPE_4'}
processing_time = {'A': 3}
max_time = 10
breaks = {(2, 3)}
is_break = {i: 1 if 2<=i<=3 else 0 for i in range(max_time)}

# 2. Decision Variables

var_task_starts = {
    task: model.NewIntVar(0, max_time, f"task_{task}_start") for task in tasks
}

var_task_ends = {
    task: model.NewIntVar(0, max_time, f"task_{task}_end") for task in tasks
}

var_task_time = {
    (task, i): model.NewBoolVar(f'task {task} uses interval {i}')
    for task in tasks
    for i in range(max_time)
}

var_task_delay = {
    task: model.NewIntVar(0, max_time, f"task_{task}_delay") for task in tasks
}

for task in tasks:
    for i in range(max_time):
        model.Add(var_task_time[task, i] == 1).OnlyEnforceIf(
            var_task_starts[task] <=
        ).OnlyEnforceIf(
            i <= var_task_ends[task]
        )



for task in tasks:
    model.Add(
            var_task_delay[task] == sum(is_break[i]* for i in range(max_time))
    )


var_task_intervals = {
    task: model.NewIntervalVar(
        var_task_starts[task],
        processing_time[task_to_product[task]],
        var_task_ends[task],
        name=f"interval_t{task}"
    )
    for task in tasks
}

