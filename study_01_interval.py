# This script benchmark
# Constraint: End - Start = Duration V.S. Using NewIntervalVar

from ortools.sat.python import cp_model
from time import time
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator


def add_circuit_constraints(model, tasks, var_task_starts, var_task_ends, literals):
    arcs = []
    for t1 in tasks:
        arcs.append([0, t1, model.NewBoolVar(f"first_to_{t1}")])
        arcs.append([t1, 0, model.NewBoolVar(f"{t1}_to_last")])
        for t2 in tasks:
            if t1 == t2:
                continue
            arcs.append([t1, t2, literals[t1, t2]])
            model.Add(var_task_ends[t1] <= var_task_starts[t2]).OnlyEnforceIf(literals[t1, t2])
    model.AddCircuit(arcs)


def run_model_1(num_tasks):
    # Using Constraint: End - Start = Duration
    model = cp_model.CpModel()
    max_time = num_tasks
    tasks = {i+1 for i in range(num_tasks)}
    processing_time = 1
    var_task_starts = {
        task: model.NewIntVar(0, max_time, f"task_{task}_start") for task in tasks
    }
    var_task_ends = {
        task: model.NewIntVar(0, max_time, f"task_{task}_end") for task in tasks
    }
    for task in tasks:
        model.Add(var_task_ends[task] - var_task_starts[task] == processing_time)
    # Sequence optimizing
    literals = {(t1, t2): model.NewBoolVar(f"{t1} -arc-> {t2}") for t1 in tasks for t2 in tasks if t1!=t2}
    add_circuit_constraints(model, tasks, var_task_starts, var_task_ends, literals)
    # Objective
    make_span = model.NewIntVar(0, max_time, "make_span")
    model.AddMaxEquality(make_span,[var_task_ends[task] for task in tasks])
    model.Minimize(make_span)
    solver = cp_model.CpSolver()
    start = time()
    status = solver.Solve(model=model)
    total_time = time() - start
    return total_time


def run_model_2(num_tasks):
    # Using NewIntervalVar
    model = cp_model.CpModel()
    max_time = num_tasks
    tasks = {i+1 for i in range(num_tasks)}
    processing_time = 1
    # 2. Decision variables
    var_task_starts = {
        task: model.NewIntVar(0, max_time, f"task_{task}_start") for task in tasks
    }
    var_task_ends = {
        task: model.NewIntVar(0, max_time, f"task_{task}_end") for task in tasks
    }
    var_intervals = {
        task: model.NewIntervalVar(
            var_task_starts[task], processing_time, var_task_ends[task], f"interval_{task}"
        )
        for task in tasks
    }
    # Sequence optimizing
    literals = {(t1, t2): model.NewBoolVar(f"{t1} -arc-> {t2}") for t1 in tasks for t2 in tasks if t1 != t2}
    add_circuit_constraints(model, tasks, var_task_starts, var_task_ends, literals)
    # Objective
    make_span = model.NewIntVar(0, max_time, "make_span")
    model.AddMaxEquality(make_span, [var_task_ends[task] for task in tasks])
    model.Minimize(make_span)
    solver = cp_model.CpSolver()
    start = time()
    status = solver.Solve(model=model)
    total_time = time() - start
    return total_time


if __name__ == '__main__':

    sizes = [2, 3, 4, 5, 6, 7, 8]
    model_1_times = []
    model_2_times = []

    print("Constraint: End - Start = Duration")
    for i in sizes:
        print(i)
        model_1_times.append(run_model_1(i))

    print("With NewIntervalVar")
    for i in sizes + [9]:
        print(i)
        model_2_times.append(run_model_2(i))

    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.plot(sizes, model_1_times, marker='o', label='Constraint: End - Start = Duration')
    plt.plot(sizes, model_2_times, '-.', marker='o', label='With NewIntervalVar')
    plt.legend()
    plt.title('Performance benchmarking')
    plt.xlabel('The number of tasks')
    plt.ylabel('Seconds')
    plt.show()