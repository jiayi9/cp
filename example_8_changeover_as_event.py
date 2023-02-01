from ortools.sat.python import cp_model

# Initiate
model = cp_model.CpModel()

'''
task   product
1       A
2       B
'''

# 1. Data

tasks = {1, 2}
tasks_0 = tasks.union({0})
task_to_product = {0: 'dummy', 1: 'A', 2: 'B'}
processing_time = {'dummy': 0, 'A': 1, 'B': 1}
changeover_time = {'dummy': 0, 'A': 2, 'B': 2}
machines = {0}
machines_starting_products = {0: 'A'}

X = {
    (m, t1, t2)
    for t1 in tasks_0
    for t2 in tasks_0
    for m in machines
    if t1 != t2
}

# This is not yet used
m_cost = {
    (m, t1, t2): 0
    if task_to_product[t1] == task_to_product[t2] or (
            task_to_product[t1] == 'dummy' and task_to_product[t2] == machines_starting_products[m]
    )
    else changeover_time[task_to_product[t2]]
    for (m, t1, t2) in X
}


# 2. Decision variables
max_time = 8

variables_task_ends = {
    task: model.NewIntVar(0, max_time, f"task_{task}_end") for task in tasks_0
}

variables_task_starts = {
    task: model.NewIntVar(0, max_time, f"task_{task}_end") for task in tasks_0
}

variables_machine_task_starts = {
    (m, t): model.NewIntVar(0, max_time, f"start_{m}_{t}")
    for t in tasks_0
    for m in machines
}

variables_machine_task_ends = {
    (m, t): model.NewIntVar(0, max_time, f"start_{m}_{t}")
    for t in tasks_0
    for m in machines
}

variables_machine_task_presences = {
    (m, t): model.NewBoolVar(f"presence_{m}_{t}")
    for t in tasks_0
    for m in machines
}

variables_machine_task_sequence = {
    (m, t1, t2): model.NewBoolVar(f"Machine {m} task {t1} --> task {t2}")
    for (m, t1, t2) in X
}

# intervals
# ! This can replace the end - start = duration constrain
variables_machine_task_intervals = {
    (m, task): model.NewOptionalIntervalVar(
        variables_machine_task_starts[m, task],
        processing_time[task_to_product[task]],
        variables_machine_task_ends[m, task],
        variables_machine_task_presences[m, task],
        name=f"t_interval_{m}_{task}"
    )
    for task in tasks_0
    for m in machines
}


# Add change over-related variables !!!

variables_co_starts = {
    (t1, t2): model.NewIntVar(0, max_time, f"co_t{t1}_to_t{t2}_start") for t1 in tasks_0 for t2 in tasks_0 if t1 != t2
}
variables_co_ends = {
    (t1, t2): model.NewIntVar(0, max_time, f"co_t{t1}_to_t{t2}_end") for t1 in tasks_0 for t2 in tasks_0 if t1 != t2
}
variables_machine_co_starts = {
    (m, t1, t2): model.NewIntVar(0, max_time, f"m{m}_co_t{t1}_to_t{t2}_start")
    for t1 in tasks_0 for t2 in tasks_0 for m in machines if t1 != t2
}
variables_machine_co_ends = {
    (m, t1, t2): model.NewIntVar(0, max_time, f"m{m}_co_t{t1}_to_t{t2}_end")
    for t1 in tasks_0 for t2 in tasks_0 for m in machines if t1 != t2
}
variables_machine_co_presences = {
    (m, t1, t2): model.NewBoolVar(f"co_presence_m{m}_t{t1}_t{t2}")
    for t1 in tasks_0
    for t2 in tasks_0
    for m in machines
    if t1 != t2
}


variables_machine_co_intervals = {
    (m, t1, t2): model.NewOptionalIntervalVar(
        variables_machine_co_starts[m, t1, t2],
        changeover_time[task_to_product[t2]],
        variables_machine_co_ends[m, t1, t2],
        variables_machine_co_presences[m, t1, t2],
        name=f"co_interval_m{m}_t{t1}_t{t2}"
    )
    for t1 in tasks_0
    for t2 in tasks_0
    for m in machines
    if t1 != t2
}







# 3. Objectives

make_span = model.NewIntVar(0, max_time, "make_span")

model.AddMaxEquality(
    make_span,
    [variables_task_ends[task] for task in tasks]
)

model.Minimize(make_span)


# 4. Constraints

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
for task in tasks_0:
    task_candidate_machines = machines
    for m in task_candidate_machines:
        model.Add(
            variables_task_starts[task] == variables_machine_task_starts[m, task]
        ).OnlyEnforceIf(variables_machine_task_presences[m, task])

        model.Add(
            variables_task_ends[task] == variables_machine_task_ends[m, task]
        ).OnlyEnforceIf(variables_machine_task_presences[m, task])


# co level link to machine-co level
for t1 in tasks_0:
    for t2 in tasks_0:
        if t1 != t2:

            for m in machines:
                model.Add(
                    variables_co_starts[t1, t2] == variables_machine_co_starts[m, t1, t2]
                ).OnlyEnforceIf(variables_machine_co_presences[m, t1, t2])

                model.Add(
                    variables_co_ends[t1, t2] == variables_machine_co_ends[m, t1, t2]
                ).OnlyEnforceIf(variables_machine_co_presences[m, t1, t2])


# for dummies: Force task 0 (dummy) starts at 0 and is present on all machines
model.Add(variables_task_starts[0] == 0)
for m in machines:
    model.Add(variables_machine_task_presences[m, 0] == 1)


# Sequence
for m in machines:
    arcs = list()
    for t1 in tasks_0:
        for t2 in tasks_0:
            # arcs
            if t1 != t2:
                arcs.append([
                        t1,
                        t2,
                        variables_machine_task_sequence[(m, t1, t2)]
                ])
                distance = m_cost[m, t1, t2]
                # cannot require the time index of task 0 to represent the first and the last position
                if t2 != 0:
                    # to schedule tasks and c/o
                    model.Add(
                        variables_task_ends[t1] <= variables_co_starts[t1, t2]
                    ).OnlyEnforceIf(variables_machine_task_sequence[(m, t1, t2)])

                    model.Add(
                        variables_co_ends[t1, t2] <= variables_task_starts[t2]
                    ).OnlyEnforceIf(variables_machine_task_sequence[(m, t1, t2)])

                    model.Add(
                        variables_co_ends[t1, t2] - variables_co_starts[t1, t2] == distance
                    ).OnlyEnforceIf(variables_machine_task_sequence[(m, t1, t2)])

                    # ensure intervals are consistent so we can apply resource constraints later
                    model.Add(
                        variables_machine_co_presences[m, t1, t2] == 1
                    ).OnlyEnforceIf(variables_machine_task_sequence[(m, t1, t2)])

                    model.Add(
                        variables_machine_co_presences[m, t1, t2] == 0
                    ).OnlyEnforceIf(variables_machine_task_sequence[(m, t1, t2)].Not())


    for task in tasks:
        arcs.append([
            task, task, variables_machine_task_presences[(m, task)].Not()
        ])
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

    print('Make-span:', solver.Value(make_span))

    for m in machines:

        print(f'------------\nMachine {m}')
        print(f'Starting dummy product: {machines_starting_products[m]}')
        for t1 in tasks_0:
            for t2 in tasks_0:
                if t1 != t2:
                    value = solver.Value(variables_machine_task_sequence[(m, t1, t2)])
                    if value == 1 and t2 != 0:
                        print(f'{t1} --> {t2}   {task_to_product[t1]} >> {task_to_product[t2]}  cost: {m_cost[m, t1, t2]}')
                        print('variables_machine_task_sequence[t1, t2]', solver.Value(variables_machine_task_sequence[m, t1, t2]))
                        print('variables_co_starts[t1, t2]', solver.Value(variables_co_starts[t1, t2]))
                        print('variables_co_ends[t1, t2]', solver.Value(variables_co_ends[t1, t2]))
                        print('variables_machine_co_presences[m, t1, t2]', solver.Value(variables_machine_co_presences[m, t1, t2]))
                        print('variables_machine_co_starts[m, t1, t2]', solver.Value(variables_machine_co_starts[m, t1, t2]))
                        print('variables_machine_co_ends[m, t1, t2]', solver.Value(variables_machine_co_ends[m, t1, t2]))
                        #print('variables_machine_co_intervals[m, t1, t2]', variables_machine_co_intervals[m, t1, t2])

                    if value == 1 and t2 == 0:
                        print(f'{t1} --> {t2}   Closing')


elif status == cp_model.INFEASIBLE:
    print("Infeasible")
elif status == cp_model.MODEL_INVALID:
    print("Model invalid")
else:
    print(status)
