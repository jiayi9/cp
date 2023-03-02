from ortools.sat.python import cp_model
from time import time
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
import string
from math import ceil

model = cp_model.CpModel()


def generate_task_data(num_of_products, num_of_tasks_per_product):
    """ Generate tasks of products (no more than 26 products) """
    products = string.ascii_uppercase[0:num_of_products]
    total_num_of_tasks = num_of_tasks_per_product*num_of_products
    tasks = {x for x in range(total_num_of_tasks)}
    task_to_product = {}
    for product_idx, product in enumerate(products):
        task_to_product.update({
            product_idx*num_of_tasks_per_product+task_idx: product for task_idx in range(num_of_tasks_per_product)
        })
    return tasks, task_to_product


def run_model(number_of_products, num_of_tasks_per_product, campaign_size, print_result=True):
    """
    Do changeovers if either of the following occurs:
    1. Changeover between different products: [A] -> changeover -> [B]
    2. Previous campaign reaching  size limit: [A ... A]  -> changeover -> next any campaign
    """

    # number_of_products = 2
    # num_of_tasks_per_product = 4
    # campaign_size = 3
    # print_result = True

    changeover_time = 2
    max_time = num_of_tasks_per_product*number_of_products*2
    processing_time = 1

    tasks, task_to_product = generate_task_data(number_of_products, num_of_tasks_per_product)
    print(f'\nInput data: {task_to_product}\n')

    product_change_indicator = {
        (t1, t2): 0 if task_to_product[t1] == task_to_product[t2] else 1 for t1 in tasks for t2 in tasks if t1 != t2
    }

    var_task_starts = {task: model.NewIntVar(0, max_time, f"task_{task}_start") for task in tasks}
    var_task_ends = {task: model.NewIntVar(0, max_time, f"task_{task}_end") for task in tasks}
    var_task_cumul = {task: model.NewIntVar(0, campaign_size-1, f"task_{task}_cumul") for task in tasks}
    for product_idx, product in enumerate(range(number_of_products)):
        model.Add(var_task_cumul[product_idx*num_of_tasks_per_product] == 0)

    var_reach_campaign_end = {task: model.NewBoolVar(f"task_{task}_reach_max") for task in tasks}
    var_product_change = {task: model.NewBoolVar(f"task_{task}_go_to_different_product") for task in tasks}

    # Lock the sequence of the tasks (assume the deadlines are in this sequence !)
    # A relative later task shall not start earlier than a relative earlier task
    # And make them
    for task in tasks:
        if task != 0:
            model.Add(var_task_starts[task-1] <= var_task_starts[task])

    var_task_intervals = {
        t: model.NewIntervalVar(var_task_starts[t], processing_time, var_task_ends[t], f"task_{t}_interval")
        for t in tasks
    }
    model.AddNoOverlap(var_task_intervals.values())

    # Set objective to minimize make-span
    make_span = model.NewIntVar(0, max_time, "make_span")
    model.AddMaxEquality(make_span, [var_task_ends[task] for task in tasks])
    model.Minimize(make_span)

    # the bool variables to indicator if t1 -> t2
    literals = {(t1, t2): model.NewBoolVar(f"{t1} -> {t2}") for t1 in tasks for t2 in tasks if t1 != t2}

    # the technical variables to allow flexible campaigning
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
            model.Add(var_product_change[t1] == product_change_indicator[t1, t2]).OnlyEnforceIf(
                literals[t1, t2]
            )

            model.Add(var_reach_campaign_end[t1] >= var_product_change[t1])


            model.Add(
                var_task_ends[t1] + var_reach_campaign_end[t1]*changeover_time <= var_task_starts[t2]
            ).OnlyEnforceIf(
                literals[t1, t2]
            )

            # allow flexible campaigning
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
                if task%num_of_tasks_per_product == 0:
                    print('--------- product divider ---------\n')
                print(f'Task {task} {task_to_product[task]}',
                      solver.Value(var_task_starts[task]),
                      solver.Value(var_task_ends[task]),
                      solver.Value(var_task_cumul[task]),
                      solver.Value(var_reach_campaign_end[task]),
                      solver.Value(var_product_change[task]),
                      )
                if solver.Value(var_reach_campaign_end[task]):
                    print('-- campaign ends --\n')
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


if __name__ == '__main__':

    _number_of_products = 3
    _num_of_tasks_per_product = 4
    _campaign_size = 4
    n = ceil(_num_of_tasks_per_product/_campaign_size)
    _make_span = (_num_of_tasks_per_product + 2*n)*_number_of_products-2
    run_model(_number_of_products, _num_of_tasks_per_product, _campaign_size)

    print(f'\nExpected make-span if all right:'
          f'\n = (_num_of_tasks_per_product + changeover_time*ceil(_num_of_tasks_per_product/_campaign_size))*'
          f'_number_of_products - changeover_time '
          f'\n = {_make_span}')

