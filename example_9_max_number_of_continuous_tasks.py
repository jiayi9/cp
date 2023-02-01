from ortools.sat.python import cp_model

# Initiate
model = cp_model.CpModel()

'''
task   product
1       A
2       A
3       A
'''

# 1. Data

tasks = {1, 2, 3}
task_to_product = {1: 'A', 2: 'A', 3: 'B'}
processing_time = {'A': 1, 'B': 1}
changeover_time = 2
max_conti_task_num = 2
max_time = 20

m_cost = {
    (t1, t2): 0
    if task_to_product[t1] == task_to_product[t2] else changeover_time
    for t1 in tasks for t2 in tasks
    if t1 != t2
}

# Need changeover if we switch from a product to a different product
# We can continue to do tasks of the same product type but there is
# a max number of continuous production of tasks with the same product type
# For A A A, we expect A -> A -> Changeover -> A
# For A A A A A, we expect A -> A -> Changeover -> A A -> Changeover -> A
# For A B, we expect A -> Changeover -> B
# For A A B, we expect A -> A -> Changeover -> B


# 2. Decision variables

variables_task_ends = {
    task: model.NewIntVar(0, max_time, f"task_{task}_end") for task in tasks
}

variables_task_starts = {
    task: model.NewIntVar(0, max_time, f"task_{task}_end") for task in tasks
}

variables_task_sequence = {
    (t1, t2): model.NewBoolVar(f"Machine {m} task {t1} --> task {t2}")
    for t1 in tasks
    for t2 in tasks
    if t1 != t2
}

variables_co_starts = {
    (t1, t2): model.NewIntVar(0, max_time, f"co_t{t1}_to_t{t2}_start")
    for t1 in tasks
    for t2 in tasks
    if t1 != t2
}

variables_co_ends = {
    (t1, t2): model.NewIntVar(0, max_time, f"co_t{t1}_to_t{t2}_end")
    for t1 in tasks
    for t2 in tasks
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

# Duration
for task in tasks:
    model.Add(
        variables_task_ends[task] - variables_task_starts[task] == processing_time[task_to_product[task]]
    )

# Sequence
arcs = list()
for t1 in tasks:
    for t2 in tasks:
        # arcs
        if t1 != t2:
            arcs.append([
                t1,
                t2,
                variables_task_sequence[(t1, t2)]
            ])
            distance = m_cost[t1, t2]
            # cannot require the time index of task 0 to represent the first and the last position
            if t2 != 0:
                # to schedule tasks and c/o
                model.Add(
                    variables_task_ends[t1] <= variables_co_starts[t1, t2]
                ).OnlyEnforceIf(variables_task_sequence[(t1, t2)])

                model.Add(
                    variables_co_ends[t1, t2] <= variables_task_starts[t2]
                ).OnlyEnforceIf(variables_task_sequence[(t1, t2)])

                model.Add(
                    variables_co_ends[t1, t2] - variables_co_starts[t1, t2] == distance
                ).OnlyEnforceIf(variables_task_sequence[(t1, t2)])

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
    print('-------------------------------------------------')
    print('Make-span:', solver.Value(make_span))
    for t1 in tasks:
        for t2 in tasks:
            if t1 != t2:
                value = solver.Value(variables_task_sequence[t1, t2])
                if value == 1 and t2 != 0:
                    print(f'{t1} --> {t2}   {task_to_product[t1]} >> {task_to_product[t2]}  cost: {m_cost[t1, t2]}')
                    print('variables_task_sequence[t1, t2]',
                          solver.Value(variables_task_sequence[t1, t2]))
                    print('variables_co_starts[t1, t2]', solver.Value(variables_co_starts[t1, t2]))
                    print('variables_co_ends[t1, t2]', solver.Value(variables_co_ends[t1, t2]))

                if value == 1 and t2 == 0:
                    print(f'{t1} --> {t2}   Closing')

elif status == cp_model.INFEASIBLE:
    print("Infeasible")
elif status == cp_model.MODEL_INVALID:
    print("Model invalid")
else:
    print(status)
