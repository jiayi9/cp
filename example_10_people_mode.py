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
machines = {1,2}
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

variables_machine_task_presences = {
    (m, t): model.NewBoolVar(f"presence_{m}_{t}")
    for t in tasks
    for m in machines
}

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

# One task to one machine.
for task in tasks:
    task_candidate_machines = machines
    tmp = [
        variables_machine_task_presences[m, task]
        for m in task_candidate_machines
    ]
    # this task is only present in one machine
    model.AddExactlyOne(tmp)

# task level link to machine-task level
for task in tasks:
    task_candidate_machines = machines
    for m in task_candidate_machines:
        model.Add(
            variables_task_starts[task] == variables_machine_task_starts[m, task]
        ).OnlyEnforceIf(variables_machine_task_presences[m, task])

        model.Add(
            variables_task_ends[task] == variables_machine_task_ends[m, task]
        ).OnlyEnforceIf(variables_machine_task_presences[m, task])


# for sequence control
variables_machine_task_sequence = {
    (m, t1, t2): model.NewBoolVar(f"Machine {m} task {t1} --> task {t2}")
    for t1 in tasks
    for t2 in tasks
    for m in machines
    if t1 != t2
}

# Sequence
for m in machines:
    arcs = list()
    for node1, t1 in enumerate(tasks):
        tmp_1 = model.NewBoolVar(f'm_{m}_first_to_{t1}')
        arcs.append([0, t1, tmp_1])

        tmp_2 = model.NewBoolVar(f'm_{m}_{t1}_to_last')
        arcs.append([t1, 0, tmp_2])

        arcs.append([t1, t1, variables_machine_task_presences[m, t1].Not()])

        for node_2, t2 in enumerate(tasks):
            # arcs
            if t1 == t2:
                continue

            arcs.append([
                t1,
                t2,
                variables_machine_task_sequence[(m, t1, t2)]
            ])

            distance = 0

            # cannot require the time index of task 0 to represent the first and the last position
            model.Add(
                variables_task_ends[t1] + distance <= variables_task_starts[t2]
            ).OnlyEnforceIf(variables_machine_task_sequence[(m, t1, t2)])

    model.AddCircuit(arcs)

#
variables_task_resource_mode = {
    (task, resource_mode): model.NewBoolVar(f"task {task} using resource mode {resource_mode}")
    for task in tasks for resource_mode in resource_modes
}

# variables_task_resource_modes = {
#     task: model.NewIntVar(0, len(resource_modes), f"the resource mode of {task}")
#     for task in tasks
# }

# one task has to pick one resource mode
for task in tasks:
    tmp = sum(variables_task_resource_mode[task, resource_mode] for resource_mode in resource_modes)
    model.Add(tmp == 1)

# link resource id with decision matrix
# for task in tasks:
#     for resource_mode in resource_modes:
#         model.Add(
#             variables_task_resource_modes[task] == resource_mode
#         ).OnlyEnforceIf(
#             variables_task_resource_mode_matrix[task, resource_mode]
#         )

variables_task_processing_time = {
    task: model.NewIntVar(0, 99, f"processing time for {task}")
    for task in tasks
}

for task in tasks:
    model.Add(
        variables_task_processing_time[task] == sum(
            processing_time[task_to_product[task], resource_mode]*variables_task_resource_mode[task, resource_mode]
            for resource_mode in resource_modes
        )
    )

# model.NewOptionalIntervalVar(1,1,2,1,'')
# model.NewOptionalIntervalVar(
#     model.NewIntVar(0,1,''),
#     model.NewIntVar(0, 1, ''),
#     model.NewIntVar(0,1,''),
#     model.NewIntVar(0, 1, ''),
#     '')

# variables_machine_task_intervals = {
#     (m, task): model.NewOptionalIntervalVar(
#         variables_machine_task_starts[m, task],
#         #2,
#         variables_task_processing_time[task],
#         #processing_time[task_to_product[task], variables_task_resource_modes[task]],
#         variables_machine_task_ends[m, task],
#         variables_machine_task_presences[m, task],
#         name=f"t_interval_{m}_{task}"
#     )
#     for task in tasks
#     for m in machines
# }

variables_machine_task_resource_mode_presence = {
    (m, task, resource_mode): model.NewBoolVar(f'm{m}_task_{task}_resource_mode_{resource_mode}_presence')
    for m in machines
    for task in tasks
    for resource_mode in resource_modes
}

for task in tasks:
    for resource_mode in resource_modes:
        model.Add(
            sum(variables_machine_task_resource_mode_presence[m, task, resource_mode] for m in machines) == 1
        ).OnlyEnforceIf(
            variables_task_resource_mode[task, resource_mode]
        )

for task in tasks:
    for m in machines:
        model.Add(
            sum(variables_machine_task_resource_mode_presence[m, task, resource_mode] for resource_mode in resource_modes) == 1
        ).OnlyEnforceIf(
            variables_machine_task_presences[m, task]
        )

for task in tasks:
    model.Add(
        sum(variables_machine_task_resource_mode_presence[m, task, resource_mode]
            for resource_mode in resource_modes
            for m in machines
            ) == 1
    )


variables_machine_task_mode_intervals = {
    (m, task, resource_mode): model.NewOptionalIntervalVar(
        variables_machine_task_starts[m, task],
        #2,
        variables_task_processing_time[task],
        #processing_time[task_to_product[task], variables_task_resource_modes[task]],
        variables_machine_task_ends[m, task],
        variables_machine_task_resource_mode_presence[m, task, resource_mode],
        #variables_machine_task_presences[m, task]*variables_task_resource_mode[task, resource_mode],
        name=f"int_m{m}_t{task}_mode{resource_mode}"
    )
    for task in tasks
    for m in machines
    for resource_mode in resource_modes
}




intervals = [x for x in variables_machine_task_mode_intervals.values()]
demands = [
    resource_requirement[task_to_product[task], resource_mode]
    for machine, task, resource_mode in variables_machine_task_mode_intervals.keys()
]
model.AddCumulative(intervals=intervals, demands=demands, capacity=7)

# 3. Objectives

make_span = model.NewIntVar(0, max_time, "make_span")

model.AddMaxEquality(
    make_span,
    [variables_task_ends[task] for task in tasks]
)

model.Minimize(make_span)


# Solve

solver = cp_model.CpSolver()
status = solver.Solve(model=model)


# for m in machines:
#     for task in tasks:
#         for resource_mode in resource_modes:
#             print(f'{m} {task} {resource_mode}')
#             print(solver.Value(variables_machine_task_mode_intervals[m, task, resource_mode]))

# Post-process

if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:

    for m in machines:
        for task in tasks:
            for resource_mode in resource_modes:
                print(f'M {m} T {task} R {resource_mode}' ,end = ' ')
                print(f'  {solver.Value(variables_machine_task_resource_mode_presence[m, task, resource_mode])}')


    print('===========================  TASKS SUMMARY  ===========================')
    for task in tasks:
        print(f'Task {task} ',
              solver.Value(variables_task_starts[task]), solver.Value(variables_task_ends[task]), end='   '
              )
        for resource_mode in resource_modes:
            value = solver.Value(variables_task_resource_mode[task, resource_mode])
            if value == 1:
                print(f'Using resource mode {resource_mode}')
    print('Make-span:', solver.Value(make_span))
    print('=======================  ALLOCATION & SEQUENCE  =======================')
    for m in machines:

        print(f'------------\nMachine {m}')
        for t1 in tasks:
            value = solver.Value(variables_machine_task_presences[(m, t1)])
            if value == 1:
                print(f'task_{t1}_on machine_{m}')
            for t2 in tasks:
                if t1 != t2:
                    value = solver.Value(variables_machine_task_sequence[(m, t1, t2)])
                    if value == 1 and t2 != 0:
                        print(f'{t1} --> {t2}   {task_to_product[t1]} >> {task_to_product[t2]}')#  cost: {m_cost[m, t1, t2]}')
                    if value == 1 and t2 == 0:
                        print(f'{t1} --> {t2}   Closing')


elif status == cp_model.INFEASIBLE:
    print("Infeasible")
