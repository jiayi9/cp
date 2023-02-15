from ortools.sat.python import cp_model

# Initiate
model = cp_model.CpModel()

M = 99999

# 1. Data
'''
task   product
1       A
2       B
3       A
'''
tasks = [0, 1, 2, 3]
task_to_product = {0: 'dummy', 1: 'A', 2: 'B', 3: 'A'}
processing_time = {'dummy': 0, 'A': 1, 'B': 1}
changeover_time = {'dummy': 0, 'A': 1, 'B': 1}
starting_product = 'A'

m = {
    (t1, t2)
    for t1 in tasks
    for t2 in tasks
    if t1 != t2
}

# A -> A, B --> B: 0
# dummy A -> A: 0
# dummy A -> B: 1
# A -> B, B -> A: 1

m_cost = {
    (t1, t2): 0
    if task_to_product[t1] == task_to_product[t2] or (
            task_to_product[t1] == 'dummy' and task_to_product[t2] == starting_product
    )
    else changeover_time[task_to_product[t2]]
    for (t1, t2) in m
}








# 2. Decision variables
max_time = 8

variables_task_ends = {
    task: model.NewIntVar(0, max_time, f"task_{task}_end") for task in tasks
}

variables_task_starts = {
    task: model.NewIntVar(0, max_time, f"task_{task}_end") for task in tasks
}

variables_sequence = {
    (t1, t2): model.NewBoolVar(f"task {t1} --> task {t2}")
    for (t1, t2) in m
}

# 3. Objectives

total_changeover_time = model.NewIntVar(0, max_time, "total_changeover_time")

total_changeover_time = sum(
    [variables_sequence[(t1, t2)]*m_cost[(t1, t2)] for (t1, t2) in m]
)

model.Minimize(total_changeover_time)


# 4. Constraints

# Add duration

for task in tasks:
    model.Add(
        variables_task_ends[task] - variables_task_starts[task] == processing_time[task_to_product[task]]
    )

# AddCircuits

arcs = list()

for (from_task, to_task) in m:
    arcs.append(
        [
            from_task,
            to_task,
            variables_sequence[(from_task, to_task)]
        ]
    )

    if from_task != 0 and to_task != 0:
        model.Add(
            variables_task_ends[from_task] <= variables_task_starts[to_task]
        ).OnlyEnforceIf(variables_sequence[(from_task, to_task)])

model.AddCircuit(arcs)


# Solve

solver = cp_model.CpSolver()
status = solver.Solve(model=model)

# Post-process

if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    for task in tasks:
        print(f'Task {task} ',
              solver.Value(variables_task_starts[task]), solver.Value(variables_task_ends[task])
              )

    for (t1, t2) in m:
        print(f'{t1} --> {t2}:   {solver.Value(variables_sequence[(t1, t2)])}')

elif status == cp_model.INFEASIBLE:
    print("Infeasible")
