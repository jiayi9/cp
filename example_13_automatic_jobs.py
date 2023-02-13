from ortools.sat.python import cp_model

# Initiate
model = cp_model.CpModel()

'''
task    product     type
1       A           3
2       B           3
'''

# 1. Data

tasks = {1, 2}
products = {'A'}
task_to_product = {1: 'A', 2: 'B'}
task_to_type = {1: 'TYPE_3', 2: 'TYPE_3'}
processing_time = {'A': 4, 'B': 3}
max_time = 10
breaks = {(0, 1), (2, 3), (4, 6), (7, 10)}


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

var_task_intervals_auto = {
    task: model.NewIntervalVar(
        var_task_starts[task],
        1,
        var_task_starts[task] + 1,
        name=f"interval_auto_t{task}"
    )
    for task in tasks
    if task_to_type[task] == 'TYPE_3'
}

var_task_seq = {
    (t1, t2): model.NewBoolVar(f"task {t1} --> task {t2}")
    for t1 in tasks
    for t2 in tasks
    if t1 != t2
}


arcs = []
for t1 in tasks:
    tmp_1 = model.NewBoolVar(f'first_to_{t1}')
    arcs.append([0, t1, tmp_1])

    tmp_2 = model.NewBoolVar(f'{t1}_to_last')
    arcs.append([t1, 0, tmp_2])

    for t2 in tasks:
        if t1 == t2:
            continue

        tmp_3 = model.NewBoolVar(f'{t1}_to_{t2}')
        arcs.append([t1, t2, var_task_seq[t1, t2]])

        model.Add(
            var_task_ends[t1] <= var_task_starts[t2]
        ).OnlyEnforceIf(var_task_seq[t1, t2])

model.AddCircuit(arcs)


# Add break time
variables_breaks = {
    (start, end): model.NewFixedSizeIntervalVar(start=start, size=end-start, name='a_break')
    for (start, end) in breaks
}

intervals = list(var_task_intervals_auto.values()) + list(variables_breaks.values())

# task, resource reduction for breaks
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
    print('=======================  ALLOCATION & SEQUENCE  =======================')
    if True:
        for t1 in tasks:
            for t2 in tasks:
                if t1 != t2:
                    value = solver.Value(var_task_seq[(t1, t2)])
                    print(f'{t1} --> {t2}  {value}')
                    # if value == 1 and t2 != 0:
                    #     print(f'{t1} --> {t2}   {task_to_product[t1]} >> {task_to_product[t2]}')#  cost: {m_cost[m, t1, t2]}')
                    # if value == 1 and t2 == 0:
                    #     print(f'{t1} --> {t2}   Closing')

elif status == cp_model.INFEASIBLE:
    print("Infeasible")
elif status == cp_model.MODEL_INVALID:
    print("Model invalid")
else:
    print(status)
