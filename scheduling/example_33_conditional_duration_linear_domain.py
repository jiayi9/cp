from ortools.sat.python import cp_model
from time import time
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from ortools.sat import cp_model_pb2
import pandas as pd
import string


if __name__ == '__main__':

    num_of_tasks = 3
    max_time = num_of_tasks*3
    processing_time = 2
    break_offset = 0
    breaks = [(i*num_of_tasks + break_offset, i*num_of_tasks + break_offset + 1) for i in range(num_of_tasks)]
    tasks = {x for x in range(num_of_tasks)}

    model = cp_model.CpModel()

    var_task_starts = {task: model.NewIntVar(0, max_time, f"task_{task}_start") for task in tasks}
    var_task_ends = {task: model.NewIntVar(0, max_time, f"task_{task}_end") for task in tasks}
    var_task_durations = {task: model.NewIntVar(2, 3, f"task_{task}_end") for task in tasks}

    print("Heuristic 1: Apply the tasks sequence heuristics")
    for task in tasks:
        if task == 0:
            continue
        model.Add(var_task_ends[task-1] <= var_task_starts[task])

    # intervals
    var_intervals = {
        task: model.NewIntervalVar(
            var_task_starts[task],
            var_task_durations[task],
            var_task_ends[task],
            name=f"interval_{task}"
        )
        for task in tasks
    }

    model.AddNoOverlap(var_intervals)


    # Objectives
    make_span = model.NewIntVar(0, max_time, "make_span")

    model.AddMaxEquality(
        make_span,
        [var_task_ends[task] for task in tasks]
    )
    model.Minimize(make_span)
