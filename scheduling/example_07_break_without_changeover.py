from ortools.sat.python import cp_model

# Initiate
model = cp_model.CpModel()

'''
task   product
1       A
2       A
3       A
4       A
'''

# 1. Data

tasks = {1, 2, 3, 4}
tasks_0 = tasks.union({0})
task_to_product = {0: 'dummy', 1: 'A', 2: 'A', 3: 'A', 4: 'A'}
processing_time = {'dummy': 0, 'A': 1}
changeover_time = {'dummy': 0, 'A': 1}
machines = {0}
machines_starting_products = {0: 'A'}
breaks = {(2, 3)}

X = {
    (m, t1, t2)
    for t1 in tasks_0
    for t2 in tasks_0
    for m in machines
    if t1 != t2
}

# Now this used in constraints, not in objective function anymore
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
variables_machine_task_intervals = {
    (m, task): model.NewOptionalIntervalVar(
        variables_machine_task_starts[m, task],
        processing_time[task_to_product[task]],
        variables_machine_task_ends[m, task],
        variables_machine_task_presences[m, task],
        name=f"interval_{m}_{task}"
    )
    for task in tasks_0
    for m in machines
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
    # For this task
    # get all allowed machines
    task_candidate_machines = machines
    # find the subset in presence matrix related to this task
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


# for dummies: Force task 0 (dummy) starts at 0 and is present on all machines
model.Add(variables_task_starts[0] == 0)
for m in machines:
    model.Add(variables_machine_task_presences[m, 0] == 1)


# Sequence
for m in machines:
    arcs = list()
    for from_task in tasks_0:
        for to_task in tasks_0:
            # arcs
            if from_task != to_task:
                arcs.append([
                        from_task,
                        to_task,
                        variables_machine_task_sequence[(m, from_task, to_task)]
                ])
                distance = m_cost[m, from_task, to_task]
                # cannot require the time index of task 0 to represent the first and the last position
                if to_task != 0:
                    model.Add(
                        variables_task_ends[from_task] + distance <= variables_task_starts[to_task]
                    ).OnlyEnforceIf(variables_machine_task_sequence[(m, from_task, to_task)])
    for task in tasks:
        arcs.append([
            task, task, variables_machine_task_presences[(m, task)].Not()
        ])
    model.AddCircuit(arcs)



# Add break time
variables_breaks = {
    (start, end): model.NewFixedSizeIntervalVar(start=start, size=1, name='a_break') for (start, end) in breaks
}

# Add resource control with break
intervals = list(variables_machine_task_intervals.values()) + list(variables_breaks.values())
model.AddCumulative(intervals, [1]*len(intervals), 1)



# Solve


# https://github.com/d-krupke/cpsat-primer

# default
# 0 1 4 2 3 0

#model.AddDecisionStrategy(variables_task_starts.values(), cp_model.CHOOSE_FIRST, cp_model.SELECT_MAX_VALUE)
# 0 4 3 2 1 0

#model.AddDecisionStrategy(variables_task_starts.values(), cp_model.CHOOSE_FIRST, cp_model.SELECT_MIN_VALUE)
# 0 1 4 2 3 0

# strange
#model.AddDecisionStrategy(variables_machine_task_sequence.values(), cp_model.CHOOSE_FIRST, cp_model.SELECT_MIN_VALUE)
# 0 1 4 2 3 0

# Need the following both to have the right sequence
model.AddDecisionStrategy(variables_task_starts.values(), cp_model.CHOOSE_FIRST, cp_model.SELECT_MIN_VALUE)
model.AddDecisionStrategy(variables_machine_task_sequence.values(), cp_model.CHOOSE_FIRST, cp_model.SELECT_MIN_VALUE)


# 0 1 2 3 4 0


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
                    if value == 1 and t2 == 0:
                        print(f'{t1} --> {t2}   Closing')


elif status == cp_model.INFEASIBLE:
    print("Infeasible")
elif status == cp_model.MODEL_INVALID:
    print("Model invalid")
else:
    print(status)
