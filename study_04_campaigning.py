# This script analyze the scalability of the campaigning concept
# This script is based on example_09_max_number_of_continuous_tasks.py

from ortools.sat.python import cp_model
from time import time
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm


def add_circuit_constraints(model, tasks, var_task_starts, var_task_ends, literals):
    arcs = []
    for t1 in tasks:
        arcs.append([0, t1, model.NewBoolVar(f"first_to_{t1}")])
        arcs.append([t1, 0, model.NewBoolVar(f"{t1}_to_last")])
        for t2 in tasks:
            if t1 == t2:
                continue
            arcs.append([t1, t2, literals[t1, t2]])
            model.Add(var_task_ends[t1] <= var_task_starts[t2]).OnlyEnforceIf(literals[t1, t2])
    model.AddCircuit(arcs)


def run_model(num_tasks):

    model = cp_model.CpModel()
    max_time = num_tasks*2
    tasks = {i+1 for i in range(num_tasks)}
    processing_time = 1

# Initiate
model = cp_model.CpModel()

# 1. Data

tasks = {1, 2, 3, 4}
products = {'A', 'B'}
task_to_product = {1: 'A', 2: 'A', 3: 'A', 4: 'B'}
processing_time = {'A': 1, 'B': 1}
changeover_time = 2
max_conti_task_num = 2
max_time = 100

m_cost = {
    (t1, t2): 0
    if task_to_product[t1] == task_to_product[t2] else changeover_time
    for t1 in tasks for t2 in tasks
    if t1 != t2
}



# Campaign related data pre=processing

max_product_campaigns = {
    product: len([task for task in tasks if task_to_product[task]==product]) for product in products
}

product_campaigns = {
    (product, f"{product}_{campaign}")
    for product in max_product_campaigns
    for campaign in list(range(0, max_product_campaigns[product]))
    #if max_product_campaigns[product] > 0
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
        campaign for campaign in campaigns if campaign_to_product[campaign] == task_to_product[task]
    ]
    for task in tasks
}

campaign_size = {campaign: max_conti_task_num for campaign in campaigns}

campaign_duration = {campaign: max_conti_task_num for campaign in campaigns}

campaign_to_tasks = {
    campaign:
        [
            task for task in tasks if campaign in task_to_campaigns[task]
        ]
    for campaign in campaigns
}

m_cost_campaign = {
    (c1, c2): 0
    if campaign_to_product[c1] == campaign_to_product[c2] else changeover_time
    for c1 in campaigns for c2 in campaigns
    if c1 != c2
}




# Need changeover if we switch from a product to a different product
# We can continue to do tasks of the same product type but there is
# a max number of continuous production of tasks with the same product type
# For A A A, we expect A -> A -> Changeover -> A
# For A A A A A, we expect A -> A -> Changeover -> A A -> Changeover -> A
# For A B, we expect A -> Changeover -> B
# For A A B, we expect A -> A -> Changeover -> B


# 2. Decision variables

variables_task_ends = {
    task: model.NewIntVar(0, max_time, f"task_{task}_end") for task in tasks
}

variables_task_starts = {
    task: model.NewIntVar(0, max_time, f"task_{task}_start") for task in tasks
}

# variables_task_sequence = {
#     (t1, t2): model.NewBoolVar(f"task {t1} --> task {t2}")
#     for t1 in tasks
#     for t2 in tasks
#     if t1 != t2
# }
#
# variables_co_starts = {
#     (t1, t2): model.NewIntVar(0, max_time, f"co_t{t1}_to_t{t2}_start")
#     for t1 in tasks
#     for t2 in tasks
#     if t1 != t2
# }
#
# variables_co_ends = {
#     (t1, t2): model.NewIntVar(0, max_time, f"co_t{t1}_to_t{t2}_end")
#     for t1 in tasks
#     for t2 in tasks
#     if t1 != t2
# }

# 3. Objectives

make_span = model.NewIntVar(0, max_time, "make_span")

model.AddMaxEquality(
    make_span,
    [variables_task_ends[task] for task in tasks]
)

model.Minimize(make_span)


# 4. Constraints

# Task Duration
# for task in tasks:
#     model.Add(
#         variables_task_ends[task] - variables_task_starts[task] == processing_time[task_to_product[task]]
#     )

var_task_intervals = {
    t: model.NewIntervalVar(
        variables_task_starts[t],
        1,
        variables_task_ends[t],
        f"task_{t}_interval"
    )
    for t in tasks
}



# Sequence
# arcs = list()
# for t1 in tasks:
#     for t2 in tasks:
#         # arcs
#         if t1 != t2:
#             arcs.append([
#                 t1,
#                 t2,
#                 variables_task_sequence[(t1, t2)]
#             ])
#             distance = m_cost[t1, t2]
#             # cannot require the time index of task 0 to represent the first and the last position
#             if t2 != 0:
#                 # to schedule tasks and c/o
#                 model.Add(
#                     variables_task_ends[t1] <= variables_co_starts[t1, t2]
#                 ).OnlyEnforceIf(variables_task_sequence[(t1, t2)])
#
#                 model.Add(
#                     variables_co_ends[t1, t2] <= variables_task_starts[t2]
#                 ).OnlyEnforceIf(variables_task_sequence[(t1, t2)])
#
#                 model.Add(
#                     variables_co_ends[t1, t2] - variables_co_starts[t1, t2] == distance
#                 ).OnlyEnforceIf(variables_task_sequence[(t1, t2)])
#
# model.AddCircuit(arcs)


# Campaigning related

var_campaign_starts = {
    c: model.NewIntVar(0, max_time, f"start_{c}") for c in campaigns
}

var_campaign_ends = {
    c: model.NewIntVar(0, max_time, f"c_end_{c}") for c in campaigns
}

var_campaign_durations = {
    c: model.NewIntVar(0, max_time, f"c_duration_{c}") for c in campaigns
}

var_campaign_presences = {
    c: model.NewBoolVar(f"c_presence_{c}") for c in campaigns
}

# Task Duration
# for c in campaigns:
#     model.Add(
#         var_campaign_ends[c] - var_campaign_starts[c] == var_campaign_durations[c]
#     )


# var_campaign_intervals = {
#     c: model.NewOptionalIntervalVar(
#         var_campaign_starts[c],  # campaign start
#         var_campaign_durations[c],  # campaign duration
#         var_campaign_ends[c],  # campaign end
#         var_campaign_presences[c],  # campaign presence
#         f"c_interval_{c}",
#     )
#     for c in campaigns
# }

# If a task in allocated to a campaign
var_task_campaign_presences = {
    (t, c): model.NewBoolVar(f"task_{t}_presence_in_campaign_{c}") for t in tasks for c in task_to_campaigns[t]
}

# add_campaign_definition_constraints
# 1. Campaign definition: Start & duration based on tasks that belongs to the campaign

for c in campaigns:
    #  1. Duration definition
    model.Add(
        var_campaign_durations[c] == sum(
            processing_time[task_to_product[t]] * var_task_campaign_presences[t, c]
            for t in campaign_to_tasks[c]
        )
    )
    # 2. Start-end definition
    # TODO: MinEquality ?
    for t in campaign_to_tasks[c]:
        # var_campaign_starts
        # var_campaign_ends

        model.Add(var_campaign_starts[c] <= variables_task_starts[t]).OnlyEnforceIf(
            var_task_campaign_presences[t, c]
        )
        model.Add(variables_task_ends[t] <= var_campaign_ends[c]).OnlyEnforceIf(
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
# 3. No campaign overlap
# Campaigns won't overlap anyway. But we need this for tasks within campaigns
model.AddNoOverlap([x for x in var_task_intervals.values()])

for c in campaigns:
    model.Add(
        var_campaign_durations[c] <= 2
    )


# Add campaign circuit

arcs = []

var_campaign_sequence = {}

for node_1, campaign_1 in enumerate(campaigns):

    tmp1 = model.NewBoolVar(f'first_to_{campaign_1}')
    arcs.append([
        0, node_1 + 1, tmp1
    ])

    tmp2 = model.NewBoolVar(f'{campaign_1}_to last')
    arcs.append([
        node_1 + 1, 0, tmp2
    ])

    arcs.append([node_1 + 1, node_1 + 1, var_campaign_presences[campaign_1].Not()])

    # for outputting
    var_campaign_sequence.update({(0, campaign_1): tmp1})
    var_campaign_sequence.update({(campaign_1, 0): tmp2})

    for node_2, campaign_2 in enumerate(campaigns):

        if node_1 == node_2:
            continue

        indicator_node_1_to_node_2 = model.NewBoolVar(f'{campaign_1}_to_{campaign_2}')

        var_campaign_sequence.update({(campaign_1, campaign_2): indicator_node_1_to_node_2})

        distance = m_cost_campaign[campaign_1, campaign_2]

        arcs.append([node_1 + 1, node_2 + 1, indicator_node_1_to_node_2])

        model.Add(
            var_campaign_ends[campaign_1] + distance <= var_campaign_starts[campaign_2]
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
status = solver.Solve(model=model)


# Post-process

if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    for task in tasks:
        print(f'Task {task} ',
              solver.Value(variables_task_starts[task]), solver.Value(variables_task_ends[task])
              )
    print('-------------------------------------------------')
    print('Make-span:', solver.Value(make_span))
    # for t1 in tasks:
    #     for t2 in tasks:
    #         if t1 != t2:
    #             value = solver.Value(var[t1, t2])
    #             if value == 1 and t2 != 0:
    #                 print(f'{t1} --> {t2}   {task_to_product[t1]} >> {task_to_product[t2]}  cost: {m_cost[t1, t2]}')
    #                 print('variables_task_sequence[t1, t2]',
    #                       solver.Value(variables_task_sequence[t1, t2]))
    #                 print('variables_co_starts[t1, t2]', solver.Value(variables_co_starts[t1, t2]))
    #                 print('variables_co_ends[t1, t2]', solver.Value(variables_co_ends[t1, t2]))
    #
    #             if value == 1 and t2 == 0:
    #                 print(f'{t1} --> {t2}   Closing')
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

                print(f'{c1} --> {c2}   {campaign_to_product[c1]} {c1_tasks}  >>  {campaign_to_product[c2]} {c2_tasks} cost: {m_cost_campaign[c1, c2]}')

            if value == 1 and c1 == 0:
                print(f'{c1} --> {c2}   Starting')

            if value == 1 and c2 == 0:
                print(f'{c1} --> {c2}   Closing')


elif status == cp_model.INFEASIBLE:
    print("Infeasible")
elif status == cp_model.MODEL_INVALID:
    print("Model invalid")
else:
    print(status)
