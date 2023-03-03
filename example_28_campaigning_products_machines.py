from ortools.sat.python import cp_model
from time import time
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from ortools.sat import cp_model_pb2
import pandas as pd
import string

model = cp_model.CpModel()


def generate_task_data(num_of_products, num_of_tasks_per_product):
    """ Generate the same number of tasks for multiple products (no more than 26 products please) """
    products = string.ascii_uppercase[0:num_of_products]
    total_num_of_tasks = num_of_tasks_per_product*num_of_products
    tasks = {x for x in range(total_num_of_tasks)}
    task_to_product = {}
    for product_idx, product in enumerate(products):
        task_to_product.update({
            product_idx*num_of_tasks_per_product+task_idx: product for task_idx in range(num_of_tasks_per_product)
        })
    return tasks, task_to_product


def run_model(number_of_products, num_of_tasks_per_product, campaign_size, number_of_machines, print_result=True):
    """
    Allocate to tasks to multiple machines
    And do changeover if either of the following occurs:
    1. Switch between different products: [A campaign] -> changeover -> [B campaign]
    2. Previous campaign reaching the size limit: [A FULL campaign]  -> changeover -> next any campaign
    """

    # number_of_products = 2
    # num_of_tasks_per_product = 4
    # campaign_size = 2
    # number_of_machines = 2
    # print_result = True

    changeover_time = 2
    max_time = num_of_tasks_per_product*number_of_products*3
    processing_time = 1
    machines = {x for x in range(number_of_machines)}

    tasks, task_to_product = generate_task_data(number_of_products, num_of_tasks_per_product)
    print('Input data:\nTasks:', tasks, task_to_product, '\n')

    product_change_indicator = {
        (t1, t2): 0 if task_to_product[t1] == task_to_product[t2] else 1 for t1 in tasks for t2 in tasks if t1 != t2
    }

    var_task_starts = {task: model.NewIntVar(0, max_time, f"task_{task}_start") for task in tasks}
    var_task_ends = {task: model.NewIntVar(0, max_time, f"task_{task}_end") for task in tasks}

    var_machine_task_starts = {(m, t): model.NewIntVar(0, max_time, f"m{m}_t{t}_start") for t in tasks for m in machines}
    var_machine_task_ends = {(m, t): model.NewIntVar(0, max_time, f"m{m}_t{t}_end") for t in tasks for m in machines}
    var_machine_task_presences = {(m, t): model.NewBoolVar(f"pre_{m}_{t}") for t in tasks for m in machines}

    var_machine_task_rank = {(m, t): model.NewIntVar(0, campaign_size-1, f"t_{t}_cu") for t in tasks for m in machines}

    # No influence on the final result. Not need to lock the starting rank values of the first tasks per product to be 0
    # for product_idx, product in enumerate(range(number_of_products)):
    #     print(f"Lock the rank of task {product_idx*num_of_tasks_per_product} to zero on all machines")
    #     for m in machines:
    #         model.Add(var_machine_task_rank[m, product_idx*num_of_tasks_per_product] == 0)

    var_m_t_reach_campaign_end = {(m, t): model.NewBoolVar(f"t{t}_reach_max_on_m{m}") for t in tasks for m in machines}
    var_m_t_product_change = {(m, t): model.NewBoolVar(f"task_{t}_change_product_on_m{m}") for t in tasks for m in machines}

    # This is optional
    model.AddDecisionStrategy(
        var_m_t_product_change.values(),
        cp_model.CHOOSE_FIRST,
        cp_model.SELECT_MIN_VALUE
    )

    # Heuristic: Lock the sequence of the tasks (assume the deadlines are in the task order
    # AND a task with later deadline shall not start earlier than a task with a earlier deadline)
    print("\nApply the tasks sequence heuristics")
    # Option 1: Locking the sequence of tasks per product! This is slower (7.54s for 3, 4, 4)
    for product_idx, product in enumerate(range(number_of_products)):
        for task_id_in_product_group, task in enumerate(range(num_of_tasks_per_product)):
            _index = product_idx * num_of_tasks_per_product + task_id_in_product_group
            if task_id_in_product_group == 0:
                print(f"\nLock {_index}", end=" ")
            else:
                print(f" <= {_index}", end=" ")
                model.Add(var_task_ends[_index-1] <= var_task_starts[_index])
    print("\n")

    # These intervals is needed otherwise the duration is not constrained
    var_machine_task_intervals = {
        (m, t): model.NewOptionalIntervalVar(
            var_machine_task_starts[m, t],
            processing_time,
            var_machine_task_ends[m, t],
            var_machine_task_presences[m, t],
            f"task_{t}_interval_on_m_{m}")
        for t in tasks for m in machines
    }

    # each task is only present in one machine
    for task in tasks:
        task_candidate_machines = machines
        tmp = [
            var_machine_task_presences[m, task]
            for m in task_candidate_machines
        ]
        model.AddExactlyOne(tmp)

    # link task-level to machine-task level for start time & end time
    for task in tasks:
        task_candidate_machines = machines
        for m in task_candidate_machines:
            model.Add(
                var_task_starts[task] == var_machine_task_starts[m, task]
            ).OnlyEnforceIf(var_machine_task_presences[m, task])

            model.Add(
                var_task_ends[task] == var_machine_task_ends[m, task]
            ).OnlyEnforceIf(var_machine_task_presences[m, task])

    # Set objective to minimize make-span
    make_span = model.NewIntVar(0, max_time, "make_span")
    total_changeover_time = model.NewIntVar(0, max_time, "total_changeover_time")
    model.AddMaxEquality(make_span, [var_task_ends[task] for task in tasks])
    model.Add(total_changeover_time == sum(var_m_t_reach_campaign_end[m,t] for m in machines for t in tasks))
    model.Minimize(make_span)

    # the bool variables to indicator if t1 -> t2
    literals = {(m, t1, t2): model.NewBoolVar(f"{t1} -> {t2}")
                for m in machines for t1 in tasks for t2 in tasks if t1 != t2}

    # the technical variables to allow flexible campaigning
    max_values = {(m, t1, t2): model.NewIntVar(0, max_time, f"{t1} -> {t2}")
                  for m in machines for t1 in tasks for t2 in tasks if t1 != t2}

    # schedule the tasks
    for m in machines:
        arcs = []
        for t1 in tasks:
            arcs.append([-1, t1, model.NewBoolVar(f"first_to_{t1}")])
            arcs.append([t1, -1, model.NewBoolVar(f"{t1}_to_last")])
            arcs.append([t1, t1, var_machine_task_presences[(m, t1)].Not()])

            for t2 in tasks:
                if t1 == t2:
                    continue
                arcs.append([t1, t2, literals[m, t1, t2]])

                # If A -> B then var_m_t_product_change>=1  (can be 0 if the last task in a machine)
                model.Add(var_m_t_product_change[m, t1] >= product_change_indicator[t1, t2]).OnlyEnforceIf(
                    literals[m, t1, t2]
                )

                # If var_m_t_product_change=1 then the campaign must end
                model.Add(var_m_t_reach_campaign_end[m, t1] >= var_m_t_product_change[m, t1])

                # if the campaign ends then there must be changeover time
                # [ task1 ] -> [ C/O ] -> [ task 2]
                model.Add(
                    var_task_ends[t1] + var_m_t_reach_campaign_end[m, t1]*changeover_time <= var_task_starts[t2]
                ).OnlyEnforceIf(
                    literals[m, t1, t2]
                )

                # model could also decide to end the campaign before it reaches size limit, then reset the rank for t2
                # has to do this in two steps since AddMaxEquality is not compatible with OnlyEnforceIf
                model.AddMaxEquality(
                    max_values[m, t1, t2],
                    [0, var_machine_task_rank[m, t1] + 1 - var_m_t_reach_campaign_end[m, t1]*campaign_size]
                )
                model.Add(var_machine_task_rank[m, t2] == max_values[m, t1, t2]).OnlyEnforceIf(literals[m, t1, t2])

        model.AddCircuit(arcs)

    solver = cp_model.CpSolver()
    start = time()
    status = solver.Solve(model=model)
    total_time = time() - start

    # show the result if getting the optimal one
    if print_result:
        if status == cp_model.OPTIMAL:
            big_list = []
            for m in machines:
                for task in tasks:
                    if solver.Value(var_machine_task_presences[m, task]):
                        tmp = [
                            f"machine {m}",
                            f"task {task}",
                            task_to_product[task],
                            solver.Value(var_task_starts[task]),
                            solver.Value(var_task_ends[task]),
                            solver.Value(var_machine_task_rank[m, task]),
                            solver.Value(var_m_t_product_change[m, task]),
                            solver.Value(var_m_t_reach_campaign_end[m, task])
                        ]
                        big_list.append(tmp)
            df = pd.DataFrame(big_list)
            df.columns = ['machine', 'task', 'product', 'start', 'end', 'rank', 'product_change', 'need_changeover']
            df = df.sort_values(['machine', 'start'])
            for m in machines:
                print(f"\n======= Machine {m} =======")
                df_tmp = df[df['machine']==f"machine {m}"]
                print(df_tmp)
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

    # number_of_products, num_of_tasks_per_product, campaign_size, number_of_machines
    args = 2, 4, 3, 3

    runtime = run_model(*args)
    print(f"Solving time: {round(runtime, 2)}s")
