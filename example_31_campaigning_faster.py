# Inspired by https://stackoverflow.com/questions/75554536

from ortools.sat.python import cp_model
from time import time
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd

model = cp_model.CpModel()


def generate_one_product_data(num_tasks):
    """ Generate N tasks of product A """
    tasks = {i for i in range(num_tasks)}
    task_to_product = ({i: 'A' for i in range(int(num_tasks))})
    return tasks, task_to_product


def run_model(num_tasks, campaign_size, print_result=True):

    # if campaign size is 2, then we need cumul indicator to be 0, 1

    changeover_time = 2
    max_time = num_tasks*2
    processing_time = 1

    tasks, task_to_product = generate_one_product_data(num_tasks)
    var_task_starts = {task: model.NewIntVar(0, max_time, f"task_{task}_start") for task in tasks}
    var_task_ends = {task: model.NewIntVar(0, max_time, f"task_{task}_end") for task in tasks}
    var_task_cumul = {task: model.NewIntVar(0, campaign_size-1, f"task_{task}_cumul") for task in tasks}
    model.Add(var_task_cumul[0]==0)
    var_reach_campaign_end = {task: model.NewBoolVar(f"task_{task}_reach_max") for task in tasks}

    # Lock the sequence of the tasks (assume the deadlines are in this sequence !)
    # A relative later task shall not start earlier than a relative earlier task
    for task in tasks:
        if task != 0:
            model.Add(var_task_starts[task-1] <= var_task_starts[task])

    # ! SHALL REMOVE THE FOLLOWING ! LEAVE THE DECISION (OF STARTING A NEW CAMPAIGN OR NOT) BACK TO THE MODEL !
    # for task in tasks:
    #     model.Add(var_task_cumul[task] == campaign_size-1).OnlyEnforceIf(var_task_reach_max[task])
    #     model.Add(var_task_cumul[task] < campaign_size-1).OnlyEnforceIf(var_task_reach_max[task].Not())

    var_task_intervals = {
        t: model.NewIntervalVar(
            var_task_starts[t],
            processing_time,
            var_task_ends[t],
            f"task_{t}_interval"
        )
        for t in tasks
    }
    model.AddNoOverlap(var_task_intervals.values())

    make_span = model.NewIntVar(0, max_time, "make_span")
    model.AddMaxEquality(make_span, [var_task_ends[task] for task in tasks])
    model.Minimize(make_span)


    literals = {(t1, t2): model.NewBoolVar(f"{t1} -> {t2}") for t1 in tasks for t2 in tasks if t1 != t2}

    max_values = {(t1, t2): model.NewIntVar(0, max_time, f"{t1} -> {t2}") for t1 in tasks for t2 in tasks if t1 != t2}

    arcs = []
    for t1 in tasks:
        arcs.append([-1, t1, model.NewBoolVar(f"first_to_{t1}")])
        arcs.append([t1, -1, model.NewBoolVar(f"{t1}_to_last")])
        for t2 in tasks:
            if t1 == t2:
                continue
            arcs.append([t1, t2, literals[t1, t2]])

            # [ task1 ] -> [ C/O ] -> [ task 2]
            model.Add(
                var_task_ends[t1] + var_reach_campaign_end[t1]*changeover_time <= var_task_starts[t2]
            ).OnlyEnforceIf(
                literals[t1, t2]
            )

            if t1 > t2:
                model.Add(literals[t1, t2] == 0)

            # This is for fixed campaigning size. DO full-campaign or NOT DO.
            # model.Add(
            #    var_task_cumul[t2] == var_task_cumul[t1] + 1 - var_task_reach_max[t1]*campaign_size
            # ).OnlyEnforceIf(literals[t1, t2])

            # The creator has confirmed AddMaxEquality is not compatible with OnlyEnforceIf
            # model.AddMaxEquality(
            #     var_task_cumul[t2],
            #     [0,var_task_cumul[t1] + 1 - var_reach_campaign_end[t1]*campaign_size]
            # ).OnlyEnforceIf(literals[t1, t2])

            # ! HERE IS THE CHANGE RECOMMENDED FOR FLEXIBLE CAMPAIGNING. BUT IN TWO STEPS !
            # NOTE var_reach_campaign_end ARE NOW OPEN BOOL DECISION VARIABLES.
            #
            # If reaching limit (var_task_cumul +1 is equal to campaign_size), the expression is max (0,0)
            #   the model must do C/O because var_task_cumul[t2] has to be 0 (a new campaign starts).
            #
            # If not yet reaching limit (var_task_cumul +1 < campaign_size), the expression is max(0, -x)
            #   model can still choose to do C/O by having var_reach_campaign_end to be 1 (if that brings a better obj)
            model.AddMaxEquality(
                max_values[t1, t2],
                [0, var_task_cumul[t1] + 1 - var_reach_campaign_end[t1]*campaign_size]
            )
            model.Add(var_task_cumul[t2] == max_values[t1, t2]).OnlyEnforceIf(literals[t1, t2])

    model.AddCircuit(arcs)

    solver = cp_model.CpSolver()
    start = time()
    status = solver.Solve(model=model)
    total_time = time() - start

    if print_result:
        if status == cp_model.OPTIMAL:
            for task in tasks:
                print(f'Task {task} ',
                      solver.Value(var_task_starts[task]),
                      solver.Value(var_task_ends[task]),
                      solver.Value(var_task_cumul[task]),
                      solver.Value(var_reach_campaign_end[task])
                      )
            print('-------------------------------------------------')
            print('Make-span:', solver.Value(make_span))
        elif status == cp_model.INFEASIBLE:
            print("Infeasible")
        elif status == cp_model.MODEL_INVALID:
            print("Model invalid")
        else:
            print(status)

    if status == cp_model.OPTIMAL:
        return total_time
    else:
        return -999


def print_unit_test_result(x, y1,  title=''):
    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.plot(x, y1,  label='Campaign size: 5')
    # plt.plot(x, y2,  label='Campaign size: 5')
    plt.legend()
    plt.title(title)
    plt.xlabel('The number of tasks')
    plt.ylabel('Seconds')
    plt.show()


if __name__ == '__main__':

    # print('Point check for run_model(40, 10)\nExpect make-span = 40*1 + (40/10 - 1)*2 = 40 + 6 = 46')
    # run_model(40, 10)

    N = 60
    sizes = range(2, N+1, 4)
    model_times_campaign = []

    for num_task in sizes:
        print(num_task)
        model_times_campaign.append(run_model(num_task, campaign_size=5, print_result=False))

    df = pd.DataFrame({
        'num_tasks': sizes,
        'time': model_times_campaign,
    })
    print(df)

    print_unit_test_result(sizes,
                           model_times_campaign,
                           'Scalability of Campaigning with Cumulative Indicator')
