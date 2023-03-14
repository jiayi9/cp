from ortools.sat.python import cp_model
from time import time
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm
# Initiate
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
    var_task_reach_max = {task: model.NewBoolVar(f"task_{task}_reach_max") for task in tasks}

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


def run_model_orig(num_tasks, campaign_size, print_result=True):

    # num_tasks = 4
    # campaign_size = 2
    changeover_time = 2
    processing_time = 1
    max_time = num_tasks*2
    products = {'A'}
    tasks, task_to_product = generate_one_product_data(num_tasks)

    model = cp_model.CpModel()

    max_product_campaigns = {
        product: len([task for task in tasks if task_to_product[task] == product])
        for product in products
    }

    product_campaigns = {
        (product, f"{product}_{campaign}")
        for product in max_product_campaigns
        for campaign in list(range(0, max_product_campaigns[product]))
    }

    campaign_to_product = {
        campaign: product for product, campaign in product_campaigns
    }

    campaigns = {campaign for product, campaign in product_campaigns}

    product_to_campaigns = {
        product: [c for c in campaigns if campaign_to_product[c] == product] for product in products
    }

    task_to_campaigns = {
        task: [
            campaign for campaign in campaigns if
            campaign_to_product[campaign] == task_to_product[task]
        ]
        for task in tasks
    }

    campaign_to_tasks = {
        campaign:
            [
                task for task in tasks if campaign in task_to_campaigns[task]
            ]
        for campaign in campaigns
    }

    # Task level
    var_task_starts = {task: model.NewIntVar(0, max_time, f"task_{task}_start") for task in tasks}
    var_task_ends = {task: model.NewIntVar(0, max_time, f"task_{task}_end") for task in tasks}
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

    var_task_campaign_presences = {
        (t, c): model.NewBoolVar(f"task_{t}_presence_in_campaign_{c}") for t in tasks for c in task_to_campaigns[t]
    }

    # Campaigning related

    var_campaign_starts = {c: model.NewIntVar(0, max_time, f"start_{c}") for c in campaigns}

    var_campaign_ends = {c: model.NewIntVar(0, max_time, f"c_end_{c}") for c in campaigns}

    var_campaign_durations = {c: model.NewIntVar(0, max_time, f"c_duration_{c}") for c in campaigns}

    var_campaign_presences = {c: model.NewBoolVar(f"c_presence_{c}") for c in campaigns}

    var_campaign_intervals = {
        c: model.NewOptionalIntervalVar(
            var_campaign_starts[c],
            var_campaign_durations[c],
            var_campaign_ends[c],
            var_campaign_presences[c],
            f"c_interval_{c}"
        )
        for c in campaigns
    }

    model.AddNoOverlap(list(var_campaign_intervals.values()))

    # If a task in allocated to a campaign
    var_task_campaign_presences = {
        (t, c): model.NewBoolVar(f"task_{t}_presence_in_campaign_{c}") for t in tasks for c in task_to_campaigns[t]
    }

    for c in campaigns:
        #  1. Duration definition
        model.Add(
            var_campaign_durations[c] == sum(
                processing_time * var_task_campaign_presences[t, c]
                for t in campaign_to_tasks[c]
            )
        )
        # 2. Start-end definition
        for t in campaign_to_tasks[c]:

            model.Add(var_campaign_starts[c] <= var_task_starts[t]).OnlyEnforceIf(
                var_task_campaign_presences[t, c]
            )
            model.Add(var_task_ends[t] <= var_campaign_ends[c]).OnlyEnforceIf(
                var_task_campaign_presences[t, c]
            )
        # 3. Link c & tc presence: If 1 task is scheduled on a campaign -> presence[t, c] = 1 ==> presence[c] == 1
        # as long as there is one task in a campaign, this campaign must be present
        model.AddMaxEquality(
            var_campaign_presences[c], [var_task_campaign_presences[t, c] for t in campaign_to_tasks[c]]
        )

    # 2. Definition of the bool var: if a task belongs to a campaign
    for task in tasks:
        # One task belongs to at most 1 campaign
        # task_to_campaigns[task]
        model.Add(
            sum(var_task_campaign_presences[task, campaign]
                for campaign in campaigns
                if campaign in task_to_campaigns[task]
                ) == 1
            )
    for c in campaigns:
        model.Add(
            var_campaign_durations[c] <= campaign_size
        )

    # Add campaign circuit
    arcs = []
    var_campaign_sequence = {}
    for node_1, campaign_1 in enumerate(campaigns):

        tmp1 = model.NewBoolVar(f'first_to_{campaign_1}')
        arcs.append([
            -1, node_1, tmp1
        ])

        tmp2 = model.NewBoolVar(f'{campaign_1}_to_last')
        arcs.append([
            node_1, -1, tmp2
        ])

        arcs.append([node_1, node_1, var_campaign_presences[campaign_1].Not()])

        # for outputting
        var_campaign_sequence.update({(0, campaign_1): tmp1})
        var_campaign_sequence.update({(campaign_1, 0): tmp2})

        for node_2, campaign_2 in enumerate(campaigns):

            if node_1 == node_2:
                continue

            indicator_node_1_to_node_2 = model.NewBoolVar(f'{campaign_1}_to_{campaign_2}')

            var_campaign_sequence.update({(campaign_1, campaign_2): indicator_node_1_to_node_2})

            arcs.append([node_1, node_2, indicator_node_1_to_node_2])

            model.Add(
                var_campaign_ends[campaign_1] + changeover_time <= var_campaign_starts[campaign_2]
            ).OnlyEnforceIf(
                indicator_node_1_to_node_2
            ).OnlyEnforceIf(
                var_campaign_presences[campaign_1]
            ).OnlyEnforceIf(
                var_campaign_presences[campaign_2]
            )

    model.AddCircuit(arcs)

    # Solve
    solver = cp_model.CpSolver()
    start = time()

    status = solver.Solve(model=model)
    total_time = time() - start
    if print_result:
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            for task in tasks:
                print(f'Task {task} ',
                      solver.Value(var_task_starts[task]), solver.Value(var_task_ends[task])
                      )
            print('-------------------------------------------------')
            print('Make-span:', solver.Value(make_span))
            for c1 in list(campaigns) + [0]:
                for c2 in list(campaigns) + [0]:
                    if c1 == c2:
                        continue
                    value = solver.Value(var_campaign_sequence[c1, c2])

                    if value == 1 and c2 != 0 and c1 != 0:
                        c1_tasks = []
                        c2_tasks = []
                        for task in campaign_to_tasks[c1]:
                            if solver.Value(var_task_campaign_presences[task, c1]) == 1:
                                c1_tasks.append(task)

                        for task in campaign_to_tasks[c2]:
                            if solver.Value(var_task_campaign_presences[task, c2]) == 1:
                                c2_tasks.append(task)

                        print(
                            f'{c1} --> {c2}   {campaign_to_product[c1]} {c1_tasks}  >>  {campaign_to_product[c2]} {c2_tasks} changeover:{changeover_time}')

                    if value == 1 and c1 == 0:
                        print(f'{c1} --> {c2}   Starting')

                    if value == 1 and c2 == 0:
                        print(f'{c1} --> {c2}   Closing')

    if status == cp_model.OPTIMAL:
        return total_time
    else:
        return -999


if __name__ == '__main__':

    sizes_new = [2, 3, 4, 5, 6, 7]
    sizes_old = [2, 3, 4, 5, 6]

    model_times_old_campaign_2 = []
    model_times_old_campaign_3 = []

    model_times_new_campaign_2 = []
    model_times_new_campaign_3 = []

    for num_task in sizes_new:
        model_times_new_campaign_2.append(run_model(num_task, campaign_size=2, print_result=False))
        model_times_new_campaign_3.append(run_model(num_task, campaign_size=3, print_result=False))

    for num_task in sizes_old:
        model_times_old_campaign_2.append(run_model(num_task, campaign_size=2, print_result=False))
        model_times_old_campaign_3.append(run_model(num_task, campaign_size=3, print_result=False))

    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.plot(sizes_new, model_times_new_campaign_2, '-.', marker='o', color = 'blue', label = 'With Cumulative Counter and Campaign size: 2')
    plt.plot(sizes_new, model_times_new_campaign_3, marker='o', color = 'blue', label = 'With Cumulative Counter and Campaign size: 3')
    plt.plot(sizes_old, model_times_old_campaign_2, '-.', color = 'orange', marker='o', label = 'Original Formulation and Campaign size: 2')
    plt.plot(sizes_old, model_times_old_campaign_3, marker='o', color = 'orange', label = 'Original Formulation and Campaign size: 3')

    plt.legend()
    plt.title('Benchmarking')
    plt.xlabel('The number of tasks')
    plt.ylabel('Seconds')
    plt.show()
