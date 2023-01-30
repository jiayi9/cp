from ortools.sat.python import cp_model

# Initiate
model = cp_model.CpModel()

M = 99999

'''
task   product
1       A
2       B
3       A
4       B
'''

# 1. Data

tasks = [0, 1, 2, 3, 4]
task_to_product = {0: 'dummy', 1: 'A', 2: 'B', 3: 'A', 4: 'B'}
processing_time = {'dummy': 0, 'A': 1, 'B': 1}
changeover_time = {'dummy': 0, 'A': 1, 'B': 1}
machines = [0, 1]
machines_starting_products = {0: 'A', 1: 'B'}

m = {
    (t1, t2)
    for t1 in tasks
    for t2 in tasks
    if t1 != t2
}

m_cost = dict()

for machine in machines:
    m_cost[machine] = {
        (t1, t2): 0
        if task_to_product[t1] == task_to_product[t2] or (
                task_to_product[t1] == 'dummy' and task_to_product[t2] == machines_starting_products[machine]
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

variables_machine_task_starts = sorted({
    (m, t): model.NewIntVar(0, max_time, f"start_{m}_{t}")
    for t in tasks
    for m in machines
})
variables_machine_task_ends = sorted({
    (m, t): model.NewIntVar(0, max_time, f"start_{m}_{t}")
    for t in tasks
    for m in machines
})
variables_machine_task_presences = sorted({
    (m, t): model.NewBoolVar(f"presence_{m}_{t}")
    for t in tasks
    for m in machines
})

