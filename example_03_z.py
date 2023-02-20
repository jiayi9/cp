from ortools.sat.python import cp_model
from time import time
from matplotlib import pyplot as plt


def generate_data(num_tasks):
    tasks = {i+1 for i in range(num_tasks)}
    tasks_0 = tasks.union({0})
    task_to_product = {0: 'dummy'}
    task_to_product.update({x+1: 'A' for x in range(int(num_tasks/2))})
    task_to_product.update({x+1: 'B' for x in range(int(num_tasks/2), int(num_tasks))})
    return tasks, tasks_0, task_to_product


def model(num_tasks):

    model = cp_model.CpModel()

    # 1. Data
    tasks, tasks_0, task_to_product = generate_data(num_tasks)
    max_time = num_tasks
    processing_time = {'dummy': 0, 'A': 1, 'B': 1}
    changeover_time = {'dummy': 0, 'A': 1, 'B': 1}
    machines = {0, 1}
    machines_starting_products = {0: 'A', 1: 'A'}

    X = {
        (m, t1, t2)
        for t1 in tasks_0
        for t2 in tasks_0
        for m in machines
        if t1 != t2
    }

    m_cost = {
        (m, t1, t2): 0
        if task_to_product[t1] == task_to_product[t2] or (
                task_to_product[t1] == 'dummy' and task_to_product[t2] == machines_starting_products[m]
        )
        else changeover_time[task_to_product[t2]]
        for (m, t1, t2) in X
    }

    # 2. Decision variables
    variables_task_ends = {
        task: model.NewIntVar(0, max_time, f"task_{task}_end") for task in tasks
    }

    variables_task_starts = {
        task: model.NewIntVar(0, max_time, f"task_{task}_end") for task in tasks
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
    variables_machine_task_presences = {
        (m, t): model.NewBoolVar(f"presence_{m}_{t}")
        for t in tasks
        for m in machines
    }

    variables_machine_task_sequence = {
        (m, t1, t2): model.NewBoolVar(f"Machine {m} task {t1} --> task {t2}")
        for (m, t1, t2) in X
    }

    # 3. Objectives

    total_changeover_time = model.NewIntVar(0, max_time, "total_changeover_time")

    total_changeover_time = sum(
        [variables_machine_task_sequence[(m, t1, t2)]*m_cost[(m, t1, t2)] for (m, t1, t2) in X]
    )

    make_span = model.NewIntVar(0, max_time, "make_span")

    model.AddMaxEquality(
        make_span,
        [variables_task_ends[task] for task in tasks]
    )
    model.Minimize(make_span + total_changeover_time)

    # 4. Constraints
    for task in tasks:
        task_candidate_machines = machines
        # find the subset in presence matrix related to this task
        tmp = [
            variables_machine_task_presences[m, task]
            for m in task_candidate_machines
        ]

        # this task is only present in one machine
        model.AddExactlyOne(tmp)

        # task level link to machine-task level
        for m in task_candidate_machines:
            model.Add(
                variables_task_starts[task] == variables_machine_task_starts[m, task]
            ).OnlyEnforceIf(variables_machine_task_presences[m, task])

            model.Add(
                variables_task_ends[task] == variables_machine_task_ends[m, task]
            ).OnlyEnforceIf(variables_machine_task_presences[m, task])

    for task in tasks:
        model.Add(
            variables_task_ends[task] - variables_task_starts[task] == processing_time[task_to_product[task]]
        )

    # AddCircuits
    for machine in machines:
        arcs = list()
        tmp = [x for x in X if x[0] == machine]
        for (m, from_task, to_task) in tmp:
            arcs.append(
                [
                    from_task,
                    to_task,
                    variables_machine_task_sequence[(m, from_task, to_task)]
                ]
            )
            if from_task != 0 and to_task != 0:
                model.Add(
                    variables_task_ends[from_task] <= variables_task_starts[to_task]
                ).OnlyEnforceIf(variables_machine_task_sequence[(m, from_task, to_task)])
        for task in tasks:
            arcs.append([
                task, task, variables_machine_task_presences[(machine, task)].Not()
            ])
        model.AddCircuit(arcs)

    # Solve
    solver = cp_model.CpSolver()

    start = time()
    status = solver.Solve(model=model)
    total_time = time() - start

    return total_time


if __name__ == '__main__':
    num_tasks = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    seconds = []
    for i in num_tasks:
        print(i)
        processing_time = model(i)
        seconds.append(processing_time)

    plt.plot(num_tasks, seconds)
    plt.show()
