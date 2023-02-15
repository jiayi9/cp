# circuit arcs
# tuples
# add arc : arcs.append([job1_index, job2_index, binary])                             |
# add link: Use OnlyEnforceIf for  binary <--->  job1.start_time <= job2.end_time     | -> three things connected

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

m = {
    (t1, t2)
    for t1 in tasks
    for t2 in tasks
    if t1 != t2
}

m_cost = {
    (t1, t2): 0
    if task_to_product[t1] == task_to_product[t2] or task_to_product[t1] == 'dummy' or task_to_product[t2] == 'dummy'
    else changeover_time[task_to_product[t2]]
    for (t1, t2) in m
}


# 2. Decision variables
'''
(1, 2): 0/1
(1, 3)
(2, 1)
(2, 3)
(3, 1)
(3, 2)
'''

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
#model.Maximize(total_changeover_time)


# 4. Constraints

# Add duration

for task in tasks:
    model.Add(
        variables_task_ends[task] - variables_task_starts[task] == processing_time[task_to_product[task]]
    )

# AddCircuits

arcs = list()
'''
arcs.append([0, 1, model.NewBoolVar("dummy0" + "_to_1")])
arcs.append([0, 2, model.NewBoolVar("dummy0" + "_to_2")])
arcs.append([0, 3, model.NewBoolVar("dummy0" + "_to_3")])
arcs.append([1, 0, model.NewBoolVar("1_to_" + "dummy0")])
arcs.append([2, 0, model.NewBoolVar("2_to_" + "dummy0")])
arcs.append([3, 0, model.NewBoolVar("3_to_" + "dummy0")])
'''
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

#model.Add(variables_task_starts[0] == 8)

# Solve

# https://github.com/d-krupke/cpsat-primer
#model.AddDecisionStrategy(variables_task_starts, cp_model.CHOOSE_FIRST, cp_model.SELECT_MIN_VALUE)
#model.AddDecisionStrategy([x[1] for x in variables_task_starts.items()], cp_model.CHOOSE_FIRST, cp_model.SELECT_MAX_VALUE)
#model.AddDecisionStrategy([x[1] for x in variables_task_starts.items()], cp_model.CHOOSE_FIRST, cp_model.SELECT_MIN_VALUE)
#model.AddDecisionStrategy([x[1] for x in variables_task_starts.items()], cp_model.CHOOSE_FIRST, cp_model.SELECT_MAX_VALUE)


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