from ortools.sat.python import cp_model
from time import time
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Set
import copy
import logging
logger = logging.getLogger(__name__)


@dataclass
class Variables:
    """
    All variables of the optimization problem, used for decisions and result values
    Type `Any` shorthand for a solver variable object, or a result float/int
    """
    var_task_starts: Dict[str, any] = None
    var_task_ends: Dict[str, any] = None
    var_task_durations: Dict[str, any] = None
    var_task_overlap_break: Dict[str, any] = None
    var_intervals: Dict[str, any] = None
    total_duration: Dict[str, any] = None
    make_span: Dict[int, any] = None


def extract_solution(solver: cp_model.CpSolver, variables):
    """Extracts the solution from the variables
    Three kinds of variables have been considered
    Integer, Boolean and Interval Variable
    Incase of other types of variables,
    this function would require modification
    """

    # Initializing solution class
    solution = copy.deepcopy(variables)
    for varname, vardict in variables.__dict__.items():
        setattr(solution, varname, None)

    # Assigning variable values
    for varname, vardict in variables.__dict__.items():
        if vardict is not None:
            setattr(
                solution,
                varname,
                {
                    k: solver.Value(v) if type(v) not in [cp_model.IntervalVar] else v
                    for k, v in vardict.items()
                },
            )
        else:
            logger.warning(f"Variable '{varname}' is defined but not used in model")
    return solution



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
    variables = Variables()
    break_offset = 0
    num_of_tasks = 1
    max_time = num_of_tasks * 3
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

    breaks = [(i * num_of_tasks + break_offset, i * num_of_tasks + break_offset + 1) for i in range(num_of_tasks)]
    tasks = {x for x in range(num_of_tasks)}
    starts_no_break = [x * 3 + break_offset + 1 for x in range(-1, num_of_tasks) if x * 3 + break_offset + 1 >= 0]
    starts_break = list(set(range(max_time)).difference(starts_no_break))
    domain_no_break = cp_model.Domain.FromValues([x for x in starts_no_break])
    domain_break = cp_model.Domain.FromValues([x for x in starts_break])

    model = cp_model.CpModel()

    variables.var_task_starts = {task: model.NewIntVar(0, max_time, f"task_{task}_start") for task in tasks}
    variables.var_task_ends = {task: model.NewIntVar(0, max_time, f"task_{task}_end") for task in tasks}
    variables.var_task_durations = {task: model.NewIntVar(2, 3, f"task_{task}_end") for task in tasks}
    variables.var_task_overlap_break = {task: model.NewBoolVar(f"task_{task}_overlap_a_break") for task in tasks}

    for task in tasks:
        if task == 0:
            continue
        model.Add(variables.var_task_ends[task - 1] <= variables.var_task_starts[task])

    for task in tasks:
        model.AddLinearExpressionInDomain(variables.var_task_starts[task], domain_break).OnlyEnforceIf(
            variables.var_task_overlap_break[task]
        )

        model.AddLinearExpressionInDomain(variables.var_task_starts[task], domain_no_break).OnlyEnforceIf(
            variables.var_task_overlap_break[task].Not()
        )
        # model.Add(variables.var_task_durations[task] == processing_time + variables.var_task_overlap_break[task] * 1)

        model.Add(variables.var_task_durations[task] == processing_time + 1).OnlyEnforceIf(
            variables.var_task_overlap_break[task]
        )

        model.Add(variables.var_task_durations[task] == processing_time).OnlyEnforceIf(
            variables.var_task_overlap_break[task].Not()
        )

    # intervals
    variables.var_intervals = {
        task: model.NewIntervalVar(
            variables.var_task_starts[task],
            variables.var_task_durations[task],
            variables.var_task_ends[task],
            name=f"interval_{task}"
        )
        for task in tasks
    }


    tracking_method = 1

    if tracking_method == 0:
        # native  cumulative

        variables.resource_needs = {'all': model.NewIntVar(0, 10, 'resource_needs')}
        resource_needs = {task: 1 for task in tasks}

        intervals, headcounts = [], []
        # 2.1 Add tasks requirements
        # 2.1.1 Add demand-based intervals
        for task in tasks:
            intervals.append(variables.var_intervals[task])
            headcounts.append(resource_needs[task])
        # 2.2 Create cumulative constraints
        model.AddCumulative(intervals, headcounts, variables.resource_needs['all'])
    elif tracking_method == 1:
        # cumulative_with_start_time

        max_r = 10
        lb = [variables.var_task_starts[i].Proto().domain[0] for i in tasks]
        ub = [variables.var_task_starts[i].Proto().domain[1] for i in tasks]

        times_min = min(lb)
        times_max = max(ub)
        time_range = range(times_min, times_max + 1)
        variables.resource_needs = {t: model.NewIntVar(0, max_r, f'resource_{t}')
                                    for t in time_range
                                    }
        variables.task_resource_needs = {(task, t): model.NewIntVar(0, max_r, f'resource_{task}_{t}')
                                         for task in tasks
                                         for t in time_range
                                         }
        variables.duration_task_resource_needs = {(task, t, duration): model.NewIntVar(0, max_r, f'resource_{task}_{t}')
                                                  for task in tasks
                                                  for t in time_range
                                                  for duration in [2, 3]
                                                  }
        variables.var_task_starts_presence = {
            (task, t): model.NewBoolVar(f'start_{task}_{t}')
            for task in tasks
            for t in range(times_min, times_max + 1)
        }
        for task in tasks:
            model.AddExactlyOne([variables.var_task_starts_presence[task, t] for t in time_range])
            # Define min & max duration
            min_duration = variables.var_task_durations[task].Proto().domain[0]
            max_duration = variables.var_task_durations[task].Proto().domain[1]
            for t in time_range:
                # Capture start time
                model.Add(variables.var_task_starts[task] == t).OnlyEnforceIf(
                    variables.var_task_starts_presence[task, t])
                # Capture based on duration
                for duration in [min_duration, max_duration]:
                    model.AddMaxEquality(variables.duration_task_resource_needs[task, t, duration],
                                         [variables.var_task_starts_presence[task, t_start]
                                          for t_start in range(t - duration + 1, t + 1)
                                          if (task, t_start) in variables.var_task_starts_presence
                                          ]
                                         )
                # Capture the actual task duration and the needs
                model.Add(variables.task_resource_needs[task, t]
                          == variables.duration_task_resource_needs[task, t, 2]
                          ).OnlyEnforceIf(variables.var_task_overlap_break[task].Not())
                model.Add(variables.task_resource_needs[task, t]
                          == variables.duration_task_resource_needs[task, t, 3]
                          ).OnlyEnforceIf(variables.var_task_overlap_break[task])

        # Capture total resource needs
        for t in range(times_min, times_max + 1):
            model.Add(variables.resource_needs[t] == sum([variables.task_resource_needs[task, t] * 1
                                                          for task in tasks
                                                          ]
                                                         ))
    elif tracking_method == 2:
        """
        Cumulative with overlap
        """

        max_r = 10
        lb = [variables.var_task_starts[i].Proto().domain[0] for i in tasks]
        ub = [variables.var_task_starts[i].Proto().domain[1] for i in tasks]

        times_min = min(lb)
        times_max = max(ub)
        resource_needs = {task: 1 for task in tasks}
        variables.resource_needs = {t: model.NewIntVar(0, max_r, f'resource_{t}')
                                    for t in range(times_min, times_max + 1)
                                    }
        variables.task_resource_needs = {(task, t): model.NewIntVar(0, max_r, f'resource_{task}_{t}')
                                         for task in tasks
                                         for t in range(times_min, times_max + 1)
                                         }
        variables.duration_task_resource_needs = {(task, t, duration): model.NewIntVar(0, max_r, f'resource_{task}_{t}')
                                                  for task in tasks
                                                  for t in range(times_min, times_max + 1)
                                                  for duration in [2, 3]
                                                  }
        variables.var_task_starts_presence = {
            (task, t): model.NewBoolVar(f'start_{task}_{t}')
            for task in tasks
            for t in range(times_min, times_max + 1)
        }
        for task in tasks:
            for t in range(times_min, times_max + 1):
                # s[i] < t
                b1 = model.NewBoolVar("")
                model.Add(variables.var_task_starts[task] <= t).OnlyEnforceIf(b1)
                model.Add(variables.var_task_starts[task] > t).OnlyEnforceIf(b1.Not())

                # t < e[i]
                b2 = model.NewBoolVar("")
                model.Add(t < variables.var_task_ends[task]).OnlyEnforceIf(b2)
                model.Add(t >= variables.var_task_ends[task]).OnlyEnforceIf(b2.Not())

                # b1 and b2 (b1 * b2)
                b3 = model.NewBoolVar("")
                model.AddBoolAnd([b1, b2]).OnlyEnforceIf(b3)
                model.AddBoolOr([b1.Not(), b2.Not()]).OnlyEnforceIf(b3.Not())

                # b1 * b2 * r[i]
                model.AddMultiplicationEquality(variables.task_resource_needs[task, t], [b3, resource_needs[task]])

        # Capture total resource needs
        for t in range(times_min, times_max + 1):
            model.Add(variables.resource_needs[t] == sum([variables.task_resource_needs[task, t] * 1
                                                          for task in tasks
                                                          ]
                                                         ))


    # Objectives
    variables.make_span = {0: model.NewIntVar(0, max_time, "make_span")}
    model.AddMaxEquality(
        variables.make_span[0],
        [variables.var_task_ends[task] for task in tasks]
    )
    variables.total_duration = {0: model.NewIntVar(0, max_time, "make_span")}
    model.Add(variables.total_duration[0] == sum(variables.var_task_durations.values()))

    model.Minimize(variables.make_span[0] + variables.total_duration[0])

    solver = cp_model.CpSolver()
    start = time()
    status = solver.Solve(model=model)
    total_time = time() - start
    print("total time:", total_time)

    print_result = True
    solution: Variables = extract_solution(solver, variables)
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
                    solution.var_task_starts[task],
                    solution.var_task_ends[task],
                    solution.var_task_overlap_break[task],
                    solution.var_task_durations[task],
                ]
                big_list.append(tmp)
            df = pd.DataFrame(big_list)
            df.columns = ['task', 'start', 'end', 'overlap_break', 'duration']
            df = df.sort_values(['start'])
            print(df)
            if tracking_method == 0:
                print('resource_needs', solution.resource_needs)

            elif tracking_method == 1:
                print('resource_needs', solution.resource_needs)
                print('task_resource_needs', solution.task_resource_needs)
                print('duration_task_resource_needs', solution.duration_task_resource_needs)
                print('var_task_starts_presence', solution.var_task_starts_presence)

            elif tracking_method == 2:
                print('resource_needs', solution.resource_needs)
                print('task_resource_needs', solution.task_resource_needs)
                print('duration_task_resource_needs', solution.duration_task_resource_needs)
                print('var_task_starts_presence', solution.var_task_starts_presence)

            print('Make-span:', solution.make_span[0])
        elif status == cp_model.INFEASIBLE:
            print("Infeasible")
        elif status == cp_model.MODEL_INVALID:
            print("Model invalid")
        else:
            print(status)
