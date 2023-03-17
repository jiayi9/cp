from ortools.sat.python import cp_model
from time import time
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from ortools.sat import cp_model_pb2


# method 1: No overlapping
def run_model_1(number_of_tasks):
    #number_of_tasks = 200
    tasks = {x for x in range(number_of_tasks)}

    # 2 tasks --> 8 time units

    processing_time = 2
    max_time = number_of_tasks * 4 + 100

    breaks = {(4 * i, 4 * i + 4) for i in range(int(number_of_tasks)) if i % 2 != 0}

    valid_start_times = [[0 + i * 8, 1 + i * 8, 2 + i * 8] for i in range(int(number_of_tasks / 2))]
    valid_start_times = [item for sublist in valid_start_times for item in sublist]

    model = cp_model.CpModel()

    var_task_starts = {task: model.NewIntVar(0, max_time, f"task_{task}_start") for task in tasks}
    var_task_ends = {task: model.NewIntVar(0, max_time, f"task_{task}_end") for task in tasks}
    var_task_intervals = {
        task: model.NewIntervalVar(
            var_task_starts[task],
            processing_time,
            var_task_ends[task],
            name=f"interval_{task}"
        )
        for task in tasks
    }
    for task in tasks:
        if task != 0:
            model.Add(var_task_starts[task-1] < var_task_starts[task])

    var_break_intervals = {
        break_id: model.NewFixedSizeIntervalVar(
            break_start, break_end-break_start, f"break_{break_id}"
        ) for break_id, (break_start, break_end) in enumerate(breaks)
    }

    all_intervals = [x for x in var_task_intervals.values()] + [x for x in var_break_intervals.values()]
    model.AddNoOverlap(all_intervals)

    # obj
    make_span = model.NewIntVar(0, max_time, "make_span")

    model.AddMaxEquality(
        make_span,
        [var_task_ends[task] for task in tasks]
    )

    model.Minimize(make_span)

    solver = cp_model.CpSolver()
    start = time()
    status = solver.Solve(model=model)
    total_time = time() - start

    # if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    #     for task in tasks:
    #         print(f'Task {task} ',
    #               solver.Value(var_task_starts[task]), solver.Value(var_task_ends[task])
    #               )
    #
    #     print('Make-span:', solver.Value(make_span))

    if status == cp_model.OPTIMAL:
        return total_time
    else:
        return -999


# method 2: Use linear domain
def run_model_2(number_of_tasks):

    #number_of_tasks = 4
    tasks = {x for x in range(number_of_tasks)}

    # 2 tasks --> 8 time units

    processing_time = 2
    max_time = number_of_tasks * 4 + 100

    breaks = {(4 * i, 4 * i + 4) for i in range(int(number_of_tasks)) if i % 2 != 0}

    valid_start_times = [[0 + i * 8, 1 + i * 8, 2 + i * 8] for i in range(int(number_of_tasks / 2))]
    valid_periods = [[0 + i * 8, 2 + i * 8] for i in range(int(number_of_tasks / 2))]

    valid_start_times = [item for sublist in valid_start_times for item in sublist]

    model = cp_model.CpModel()

    #model.NewIntVarFromDomain(cp_model.Domain.FromIntervals([[1, 2], [4, 6]]), 'x')

    var_task_starts = {task: model.NewIntVarFromDomain(
        cp_model.Domain.FromIntervals(valid_periods),
        f"task_{task}_start")
        for task in tasks}

    var_task_ends = {task: model.NewIntVar(0, max_time, f"task_{task}_end") for task in tasks}
    var_task_intervals = {
        task: model.NewIntervalVar(
            var_task_starts[task],
            processing_time,
            var_task_ends[task],
            name=f"interval_{task}"
        )
        for task in tasks
    }
    for task in tasks:
        if task != 0:
            model.Add(var_task_starts[task-1] < var_task_starts[task])

    all_intervals = [x for x in var_task_intervals.values()]
    model.AddNoOverlap(all_intervals)

    # obj
    make_span = model.NewIntVar(0, max_time, "make_span")

    model.AddMaxEquality(
        make_span,
        [var_task_ends[task] for task in tasks]
    )

    model.Minimize(make_span)

    solver = cp_model.CpSolver()
    start = time()
    status = solver.Solve(model=model)
    total_time = time() - start

    # if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    #     for task in tasks:
    #         print(f'Task {task} ',
    #               solver.Value(var_task_starts[task]), solver.Value(var_task_ends[task])
    #               )
    #
    #     print('Make-span:', solver.Value(make_span))

    if status == cp_model.OPTIMAL:
        return total_time
    else:
        return -999


if __name__ == '__main__':

    N = 200

    sizes_1 = list(range(2, N, 10))
    sizes_2 = list(range(2, N, 10))

    model_1_times = []
    model_2_times = []

    print("\nNoOverlap\n")
    for size in tqdm(sizes_1):
        model_1_times.append(run_model_1(size))

    print("\nLinear Domain\n")
    for size in tqdm(sizes_2):
        model_2_times.append(run_model_2(size))

    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.plot(sizes_1, model_1_times, marker='o', label='NoOverLapping')
    plt.plot(sizes_2, model_2_times, '-.', marker='o', label='Linear Domain')
    plt.legend()
    plt.title('Performance benchmarking')
    plt.xlabel('The number of tasks')
    plt.ylabel('Seconds')
    plt.show()
