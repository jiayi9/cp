# Inspired by https://stackoverflow.com/questions/75554536

from ortools.sat.python import cp_model
from time import time
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

model = cp_model.CpModel()


def generate_one_product_data(num_tasks):
    """ Generate N tasks of product A """
    tasks = {i for i in range(num_tasks)}
    task_to_product = ({i: 'A' for i in range(int(num_tasks))})
    return tasks, task_to_product


def run_model(num_tasks, campaign_size, print_result = True):

    # if campaign size is 2, then we need cumul indicator to be 0, 1

    changeover_time = 2
    max_time = num_tasks*2
    processing_time = 1

    tasks, task_to_product = generate_one_product_data(num_tasks)
    var_task_starts = {task: model.NewIntVar(0, max_time, f"task_{task}_start") for task in tasks}
    var_task_ends = {task: model.NewIntVar(0, max_time, f"task_{task}_end") for task in tasks}
    var_task_cumul = {task: model.NewIntVar(0, campaign_size-1, f"task_{task}_cumul") for task in tasks}
    model.Add(var_task_cumul[0]==0)
    var_task_reach_max = {task: model.NewBoolVar(f"task_{task}_reach_max") for task in tasks}

    # Lock the sequence of the tasks (assume the deadlines are in this sequence !)
    for task in tasks:
        if task != 0:
            model.Add(var_task_ends[task-1] <= var_task_starts[task])


    for task in tasks:
        model.Add(var_task_cumul[task] == campaign_size-1).OnlyEnforceIf(var_task_reach_max[task])
        model.Add(var_task_cumul[task] < campaign_size-1).OnlyEnforceIf(var_task_reach_max[task].Not())

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

    arcs = []
    for t1 in tasks:
        arcs.append([-1, t1, model.NewBoolVar(f"first_to_{t1}")])
        arcs.append([t1, -1, model.NewBoolVar(f"{t1}_to_last")])
        for t2 in tasks:
            if t1 == t2:
                continue
            arcs.append([t1, t2, literals[t1, t2]])

            # if from task has not reached MAX, continue the campaign
            model.Add(var_task_ends[t1] <= var_task_starts[t2]).OnlyEnforceIf(
                literals[t1, t2]
            ).OnlyEnforceIf(var_task_reach_max[t1].Not())
            model.Add(var_task_cumul[t2] == var_task_cumul[t1] + 1).OnlyEnforceIf(
                literals[t1, t2]
            ).OnlyEnforceIf(var_task_reach_max[t1].Not())

            # if from task has reached MAX, apply changeover and reset its cumulative indicator
            model.Add(var_task_cumul[t2] == 0).OnlyEnforceIf(
                literals[t1, t2]
            ).OnlyEnforceIf(var_task_reach_max[t1])
            model.Add(var_task_ends[t1] + changeover_time <= var_task_starts[t2]).OnlyEnforceIf(
                literals[t1, t2]
            ).OnlyEnforceIf(var_task_reach_max[t1])

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
                      )
            print('-------------------------------------------------')
            print('Make-span:', solver.Value(make_span))
        elif status == cp_model.INFEASIBLE:
            print("Infeasible")
        elif status == cp_model.MODEL_INVALID:
            print("Model invalid")
        else:
            print(status)

    return total_time


def print_unit_test_result(x, y1, y2, title=''):
    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.plot(x, y1, marker='o', label = 'Campaign size: 2')
    plt.plot(x, y2, marker='o', label = 'Campaign size: 3')
    plt.legend()
    plt.title(title)
    plt.xlabel('The number of tasks')
    plt.ylabel('Seconds')
    plt.show()


if __name__ == '__main__':

    N = 25
    sizes = range(2, N+1)
    model_times_campaign_2 = []
    model_times_campaign_3 = []

    for num_task in sizes:
        print(num_task)
        model_times_campaign_2.append(run_model(num_task, campaign_size=2, print_result=False))
        model_times_campaign_3.append(run_model(num_task, campaign_size=3, print_result=False))

    print_unit_test_result(sizes, model_times_campaign_2, model_times_campaign_3,
                           'Scalability of Campaigning with Cumulative Indicator')