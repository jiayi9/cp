from ortools.sat.python import cp_model
from time import time
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from ortools.sat import cp_model_pb2
import pandas as pd
import numpy as np
import string


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


def create_model(number_of_products, num_of_tasks_per_product, campaign_size, number_of_machines, print_result=True):

    model = cp_model.CpModel()

    changeover_time = 2
    max_time = num_of_tasks_per_product*number_of_products*5
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
    var_m_t_reach_campaign_end = {(m, t): model.NewBoolVar(f"t{t}_reach_max_on_m{m}") for t in tasks for m in machines}
    var_m_t_product_change = {(m, t): model.NewBoolVar(f"task_{t}_change_product_on_m{m}") for t in tasks for m in machines}

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

    return model, make_span


def get_solutions(model, solver):
    vars_sol = {}
    for i, var in enumerate(model.Proto().variables):
        value = solver.ResponseProto().solution[i]
        vars_sol[var.name] = value
    return vars_sol


def add_hints(model, solution):
    for i, var in enumerate(model.Proto().variables):
        if var.name in solution:
            model.Proto().solution_hint.vars.append(i)
            model.Proto().solution_hint.values.append(solution[var.name])


def create_model_for_test():
    # model = create_model(5, 20, 3, 1)
    model, obj = create_model(4, 4, 3, 1)
    return model, obj


if __name__ == '__main__':

    time_n = 2
    times = [(i + 1) * 10 for i in range(time_n)]
    phases = [{'phase_id': i, 'max_time': (i+1)*10} for i in range(time_n)]

    # repetitions
    M = 1

    # no hints from last run
    print('-------------------------------------------------------------------------------------------------------')
    print('--------------------------------  No hints from last run/No obj  --------------------------------------')
    print('-------------------------------------------------------------------------------------------------------')

    L_1 = []

    for m in range(M):
        model, _ = create_model_for_test()

        obj_list = []
        for phase in phases:
            phase_id, max_time = phase['phase_id'], phase['max_time']
            print('----------------------------')
            model.ClearHints()
            if phase_id == 0:
                solver = cp_model.CpSolver()
            # if phase_id > 0 and 'solution' in locals():
            #     print("Add hints")
            #     add_hints(model, solution)
            print('number of variables in solution hints:', len(model.Proto().solution_hint.vars))
            solver.parameters.max_time_in_seconds = max_time
            start = time()
            status = solver.Solve(model=model)
            print('number of solutions:', solver.ResponseProto().solution)

            if status == 1 or status == 3:
                print(f'error status : {status}')
                break
            if status == 0:
                print(f'Cannot find a feasible solution in the given time {max_time}. status:{status}')
                obj_list.append(np.nan)
                continue

            obj_value = solver.ObjectiveValue()
            obj_list.append(obj_value)
            solution = get_solutions(model, solver)
            total_time = time() - start
            print(f"phase_id: {phase_id}, max_time: {max_time}, status: {status}, obj: {obj_value}. total time: {round(total_time,1)}")

            if status == 4:
                print('======================================================')
                print('Optimal Solution Achieved ! No need to continue')
                break

        L_1.append(obj_list)


    #
    print('-------------------------------------------------------------------------------------------------------')
    print('--------------------------------  Add hints from last run/No obj  -------------------------------------')
    print('-------------------------------------------------------------------------------------------------------')

    L_2 = []

    for m in range(M):
        model, _ = create_model_for_test()

        obj_list = []
        for phase in phases:
            phase_id, max_time = phase['phase_id'], phase['max_time']
            print('----------------------------')
            model.ClearHints()
            if phase_id == 0:
                solver = cp_model.CpSolver()
            if phase_id > 0 and 'solution' in locals():
                print("Add hints")
                add_hints(model, solution)
            print('number of variables in solution hints:', len(model.Proto().solution_hint.vars))
            solver.parameters.max_time_in_seconds = max_time
            start = time()
            status = solver.Solve(model=model)
            print('number of solutions:', solver.ResponseProto().solution)

            if status == 1 or status == 3:
                print(f'error status : {status}')
                break
            if status == 0:
                print(f'Cannot find a feasible solution in the given time {max_time}. status:{status}')
                obj_list.append(np.nan)
                continue

            obj_value = solver.ObjectiveValue()
            obj_list.append(obj_value)
            solution = get_solutions(model, solver)
            total_time = time() - start
            print(f"phase_id: {phase_id}, max_time: {max_time}, status: {status}, obj: {obj_value}. total time: {round(total_time,1)}")

            if status == 4:
                print('======================================================')
                print('Optimal Solution Achieved ! No need to continue')
                break

        L_2.append(obj_list)


    #
    print('-------------------------------------------------------------------------------------------------------')
    print('---------------------------   No hints from last run / obj from last run  -----------------------------')
    print('-------------------------------------------------------------------------------------------------------')

    L_3 = []

    for m in range(M):
        model, obj_var = create_model_for_test()

        target_variable = [x for x in model.Proto().variables if x.name == 'make_span'][0]

        obj_list = []

        for phase in phases:
            phase_id, max_time = phase['phase_id'], phase['max_time']
            print('----------------------------')
            model.ClearHints()
            if phase_id == 0:
                solver = cp_model.CpSolver()
            if phase_id > 0 and 'solution' in locals():
                # print("Add hints")
                # add_hints(model, solution)
                model.Add(obj_var <= int(obj_value))

            print('number of variables in solution hints:', len(model.Proto().solution_hint.vars))
            solver.parameters.max_time_in_seconds = max_time
            start = time()
            status = solver.Solve(model=model)
            print('number of solutions:', solver.ResponseProto().solution)

            if status == 1 or status == 3:
                print(f'error status : {status}')
                break
            if status == 0:
                print(f'Cannot find a feasible solution in the given time {max_time}. status:{status}')
                obj_list.append(np.nan)
                continue

            obj_value = solver.ObjectiveValue()
            obj_list.append(obj_value)
            solution = get_solutions(model, solver)
            total_time = time() - start
            print(f"phase_id: {phase_id}, max_time: {max_time}, status: {status}, obj: {obj_value}. total time: {round(total_time,1)}")

            if status == 4:
                print('======================================================')
                print('Optimal Solution Achieved ! No need to continue')
                break

        L_3.append(obj_list)

    #
    print('-------------------------------------------------------------------------------------------------------')
    print('---------------------------      hints from last run / obj from last run  -----------------------------')
    print('-------------------------------------------------------------------------------------------------------')
    L_4 = []

    for m in range(M):
        model, obj_var = create_model_for_test()

        target_variable = [x for x in model.Proto().variables if x.name == 'make_span'][0]

        obj_list = []

        for phase in phases:
            phase_id, max_time = phase['phase_id'], phase['max_time']
            print('----------------------------')
            model.ClearHints()
            if phase_id == 0:
                solver = cp_model.CpSolver()
            if phase_id > 0 and 'solution' in locals():
                print("Add hints")
                add_hints(model, solution)
                print("Using the obj achieved from last run as LB")
                model.Add(obj_var <= int(obj_value))

            print('number of variables in solution hints:', len(model.Proto().solution_hint.vars))
            solver.parameters.max_time_in_seconds = max_time
            start = time()
            status = solver.Solve(model=model)
            print('number of solutions:', solver.ResponseProto().solution)

            if status == 1 or status == 3:
                print(f'error status : {status}')
                break
            if status == 0:
                print(f'Cannot find a feasible solution in the given time {max_time}. status:{status}')
                obj_list.append(np.nan)
                continue

            obj_value = solver.ObjectiveValue()
            obj_list.append(obj_value)
            solution = get_solutions(model, solver)
            total_time = time() - start
            print(f"phase_id: {phase_id}, max_time: {max_time}, status: {status}, obj: {obj_value}. total time: {round(total_time,1)}")

            if status == 4:
                print('======================================================')
                print('Optimal Solution Achieved ! No need to continue')
                break

        L_4.append(obj_list)


    df_no_hint_no_obj = pd.DataFrame(L_1)
    df_no_hint_no_obj.columns = times
    df_no_hint_no_obj = df_no_hint_no_obj.assign(group='no hints & no obj')

    df_hint_no_obj = pd.DataFrame(L_2)
    df_hint_no_obj.columns = times
    df_hint_no_obj = df_hint_no_obj.assign(group='hints & no obj')

    df_no_hint_obj = pd.DataFrame(L_3)
    df_no_hint_obj.columns = times
    df_no_hint_obj = df_no_hint_obj.assign(group='no hints & obj')

    df_hint_obj = pd.DataFrame(L_4)
    df_hint_obj.columns = times
    df_hint_obj = df_hint_obj.assign(group='hints & obj')

    df = pd.concat([df_no_hint_no_obj, df_hint_no_obj, df_no_hint_obj, df_hint_obj], axis=0)
    df.to_csv('C:/Temp/hints_analysis.csv', index=False)


    # plot the results

    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    for idx, v in enumerate(L_1):
        if idx == 0:
            plt.plot(times, v, '-.',marker='o', color='blue', label='No hint & No Obj.')
        else:
            plt.plot(times, v, '-.',marker='o', color='blue')

    for idx, v in enumerate(L_2):
        if idx == 0:
            plt.plot(times, v, '-.', marker='o', color='orange', label='Hint & No Obj')
        else:
            plt.plot(times, v, '-.',  marker='o', color='orange')

    for idx, v in enumerate(L_3):
        if idx == 0:
            plt.plot(times, v, '-.', marker='o', color='red', label='No hint & Obj')
        else:
            plt.plot(times, v, '-.',  marker='o', color='red')

    for idx, v in enumerate(L_4):
        if idx == 0:
            plt.plot(times, v, '-.', marker='o', color='green', label='Hint & Obj')
        else:
            plt.plot(times, v, '-.',  marker='o', color='green')

    plt.legend()
    plt.title(f'time vs obj')
    plt.xlabel('running seconds')
    plt.ylabel('obj')
    plt.show()

