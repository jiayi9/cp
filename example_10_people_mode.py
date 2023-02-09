from ortools.sat.python import cp_model

# Initiate
model = cp_model.CpModel()

'''
task   product
1       A
2       A
'''

### 1. Data

tasks = {1, 2}
products = {'A'}
task_to_product = {1: 'A', 2: 'A'}
machines = {1, 2}
resource_modes = {1, 2}

# product -> people mode -> time duration
processing_time = {
    ('A', 1): 3,
    ('A', 2): 2
}
resource_requirement = {
    ('A', 1): 2,
    ('A', 2): 3
}


max_time = 20

# (task, people_mode)

# 2. Decision variables

variables_task_ends = {
    task: model.NewIntVar(0, max_time, f"task_{task}_end") for task in tasks
}

variables_task_starts = {
    task: model.NewIntVar(0, max_time, f"task_{task}_start") for task in tasks
}

variables_task_resource_mode_matrix = {
    (task, resource_mode): model.NewBoolVar(f"task {task} using resource mode {resource_mode}")
    for task in tasks for resource_mode in resource_modes
}

variables_task_resource_modes = {
    task: model.NewIntVar(0, len(resource_modes), f"the resource mode of {task}")
    for task in tasks
}

# variables_task_sequence = {
#     (t1, t2): model.NewBoolVar(f"task {t1} --> task {t2}")
#     for t1 in tasks
#     for t2 in tasks
#     if t1 != t2
# }


variables_machine_task_starts = {
    (m, t): model.NewIntVar(0, max_time, f"start_{m}_{t}")
    for t in tasks
    for m in machines
}

variables_machine_task_ends = {
    (m, t): model.NewIntVar(0, max_time, f"start_{m}_{t}")
    for t in tasks
    for m in machines
}

variables_machine_task_presences = {
    (m, t): model.NewBoolVar(f"presence_{m}_{t}")
    for t in tasks
    for m in machines
}

variables_machine_task_sequence = {
    (m, t1, t2): model.NewBoolVar(f"Machine {m} task {t1} --> task {t2}")
    for t1 in tasks
    for t2 in tasks
    for m in machines
    if t1 != t2
}

# model.NewOptionalIntervalVar(1,1,2,1,'')
# model.NewOptionalIntervalVar(
#     model.NewIntVar(0,1,''),
#     model.NewIntVar(0, 1, ''),
#     model.NewIntVar(0,1,''),
#     model.NewIntVar(0, 1, ''),
#     '')


variables_machine_task_intervals = {
    (m, task): model.NewOptionalIntervalVar(
        variables_machine_task_starts[m, task],

        processing_time[task_to_product[task], variables_task_resource_modes[task, 1]],

        variables_machine_task_ends[m, task],
        variables_machine_task_presences[m, task],
        name=f"t_interval_{m}_{task}"
    )
    for task in tasks
    for m in machines
}



# 3. Objectives

make_span = model.NewIntVar(0, max_time, "make_span")

model.AddMaxEquality(
    make_span,
    [variables_task_ends[task] for task in tasks]
)

model.Minimize(make_span)
