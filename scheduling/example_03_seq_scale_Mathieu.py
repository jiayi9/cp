# USE INTERVAL !!!

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
    variables_intervals = {
        (m, t): model.NewOptionalIntervalVar(
            start=variables_machine_task_starts[m, t],
            size=processing_time[task_to_product[t]],
            end=variables_machine_task_ends[m, t],
            is_present=variables_machine_task_presences[m, t],
            name=f"interval_{t}_{m}",
        )
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

    for machine in machines:
        tmp = {(m, t) for (m, t) in variables_intervals if m == machine}
        intervals = [variables_intervals[x] for x in tmp]
        model.AddNoOverlap(intervals)

    # Create circuit constraints
    add_circuit_constraints(
        model,
        machines,
        tasks,
        variables_task_starts,
        variables_task_ends,
        variables_machine_task_presences,
    )

    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 8
    start = time()
    status = solver.Solve(model=model)
    total_time = time() - start

    return total_time


def add_circuit_constraints(
    model,
    machines,
    tasks,
    variables_task_starts,
    variables_task_ends,
    variables_machine_task_presences,
):
    for machine in machines:
        arcs = (
            []
        )  # List of all feasible arcs within a machine. Arcs are boolean to specify circuit from node to node
        machine_tasks = tasks

        for node_1, task_1 in enumerate(machine_tasks):
            mt_1 = str(task_1) + "_" + str(machine)
            # Initial arc from the dummy node (0) to a task.
            arcs.append(
                [0, node_1 + 1, model.NewBoolVar("first" + "_" + mt_1)]
            )  # if mt_1 follows dummy node 0
            # Final arc from an arc to the dummy node (0).
            arcs.append(
                [node_1 + 1, 0, model.NewBoolVar("last" + "_" + mt_1)]
            )  # if dummy node 0 follows mt_1

            # For optional task on machine (i.e other machine choice)
            # Self-looping arc on the node that corresponds to this arc.
            arcs.append(
                [
                    node_1 + 1,
                    node_1 + 1,
                    variables_machine_task_presences[(machine, task_1)].Not(),
                ]
            )

            for node_2, task_2 in enumerate(machine_tasks):
                if node_1 == node_2:
                    continue
                mt_2 = str(task_2) + "_" + str(machine)
                # Add sequential boolean constraint: mt_2 follows mt_1
                mt2_after_mt1 = model.NewBoolVar(f"{mt_2} follows {mt_1}")
                arcs.append([node_1 + 1, node_2 + 1, mt2_after_mt1])

                # We add the reified precedence to link the literal with the
                # times of the two tasks.
                min_distance = 0
                (
                    model.Add(
                        variables_task_starts[task_2] >= variables_task_ends[task_1] + min_distance
                    ).OnlyEnforceIf(mt2_after_mt1)
                )
        model.AddCircuit(arcs)



if __name__ == '__main__':

    num_tasks = [x+2 for x in range(80)]

    seconds = []

    for i in num_tasks:
        print(i)
        processing_time = model(i)
        seconds.append(processing_time)

    plt.plot(num_tasks, seconds)
    plt.show()
