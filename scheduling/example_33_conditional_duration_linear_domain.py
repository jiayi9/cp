from ortools.sat.python import cp_model
from time import time
import pandas as pd


if __name__ == '__main__':
    """
    Offset = 0:
        | x |   |   | x |   |   | x |   |   |
    Offset = 1:
        |   | x |   |   | x |   |   | x |   |    
    Offset = 2:
        |   |   | x |   |   | x |   |   | x |    
        
    where x represent a unit duration break period
    
    """
    break_offset = 0

    num_of_tasks = 3
    max_time = num_of_tasks*3
    processing_time = 2

    if break_offset == 0:
        help_text = "| x |   |   | x |   |   | x |   |   |"
    elif break_offset == 1:
        help_text = "|   | x |   |   | x |   |   | x |   |"
    elif break_offset == 2:
        help_text = "|   |   | x |   |   | x |   |   | x |"
    else:
        print("offset wrong")
        exit()

    breaks = [(i*num_of_tasks + break_offset, i*num_of_tasks + break_offset + 1) for i in range(num_of_tasks)]
    tasks = {x for x in range(num_of_tasks)}
    starts_no_break = [x*3+break_offset+1 for x in range(-1, num_of_tasks) if x*3+break_offset+1>= 0]
    starts_break = list(set(range(max_time)).difference(starts_no_break))
    domain_no_break = cp_model.Domain.FromIntervals([[x] for x in starts_no_break])
    domain_break = cp_model.Domain.FromIntervals([[x] for x in starts_break])

    model = cp_model.CpModel()

    var_task_starts = {task: model.NewIntVar(0, max_time, f"task_{task}_start") for task in tasks}
    var_task_ends = {task: model.NewIntVar(0, max_time, f"task_{task}_end") for task in tasks}
    var_task_durations = {task: model.NewIntVar(2, 3, f"task_{task}_end") for task in tasks}
    var_task_overlap_break = {task: model.NewBoolVar(f"task_{task}_overlap_a_break") for task in tasks}

    # print("Heuristic 1: Apply the tasks sequence heuristics")
    for task in tasks:
        if task == 0:
            continue
        model.Add(var_task_ends[task-1] <= var_task_starts[task])

    for task in tasks:

        model.AddLinearExpressionInDomain(var_task_starts[task], domain_break).OnlyEnforceIf(
            var_task_overlap_break[task]
        )

        model.AddLinearExpressionInDomain(var_task_starts[task], domain_no_break).OnlyEnforceIf(
            var_task_overlap_break[task].Not()
        )

        model.Add(var_task_durations[task] == processing_time+1).OnlyEnforceIf(
            var_task_overlap_break[task]
        )

        model.Add(var_task_durations[task] == processing_time).OnlyEnforceIf(
            var_task_overlap_break[task].Not()
        )

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

    model.AddNoOverlap(var_intervals.values())

    # Objectives
    make_span = model.NewIntVar(0, max_time, "make_span")
    model.AddMaxEquality(
        make_span,
        [var_task_ends[task] for task in tasks]
    )

    model.Minimize(make_span + sum(var_task_durations))
    # model.Minimize(sum(var_task_durations))

    solver = cp_model.CpSolver()
    start = time()
    status = solver.Solve(model=model)
    total_time = time() - start

    print_result = True
    # show the result if getting the optimal one
    if print_result:
        print("-----------------------------------------")
        print(help_text)
        print("breaks periods:", breaks)
        print("break starts:", starts_break)
        print("no break starts:", starts_no_break)

        if status == cp_model.OPTIMAL:
            big_list = []
            for task in tasks:
                tmp = [
                    f"task {task}",
                    solver.Value(var_task_starts[task]),
                    solver.Value(var_task_ends[task]),
                    solver.Value(var_task_overlap_break[task]),
                    solver.Value(var_task_durations[task]),
                ]
                big_list.append(tmp)
            df = pd.DataFrame(big_list)
            df.columns = ['task', 'start', 'end', 'overlap_break', 'duration']
            df = df.sort_values(['start'])
            print(df)
            print('Make-span:', solver.Value(make_span))
        elif status == cp_model.INFEASIBLE:
            print("Infeasible")
        elif status == cp_model.MODEL_INVALID:
            print("Model invalid")
        else:
            print(status)
