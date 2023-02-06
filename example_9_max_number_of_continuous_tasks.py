from ortools.sat.python import cp_model

# Initiate
model = cp_model.CpModel()

'''
task   product
1       A
2       A
3       A
4       B
'''

# 1. Data

tasks = {1, 2, 3, 4}
products = {'A', 'B'}
task_to_product = {1: 'A', 2: 'A', 3: 'A', 4: 'B'}
processing_time = {'A': 1, 'B': 1}
changeover_time = 2
max_conti_task_num = 2
max_time = 20
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
    task: model.NewIntVar(0, max_time, f"task_{task}_end") for task in tasks
}

variables_task_sequence = {
    (t1, t2): model.NewBoolVar(f"task {t1} --> task {t2}")
    for t1 in tasks
    for t2 in tasks
    if t1 != t2
}

variables_co_starts = {
    (t1, t2): model.NewIntVar(0, max_time, f"co_t{t1}_to_t{t2}_start")
    for t1 in tasks
    for t2 in tasks
    if t1 != t2
}

variables_co_ends = {
    (t1, t2): model.NewIntVar(0, max_time, f"co_t{t1}_to_t{t2}_end")
    for t1 in tasks
    for t2 in tasks
    if t1 != t2
}

# 3. Objectives

make_span = model.NewIntVar(0, max_time, "make_span")

model.AddMaxEquality(
    make_span,
    [variables_task_ends[task] for task in tasks]
)

model.Minimize(make_span)


# 4. Constraints

# Duration
for task in tasks:
    model.Add(
        variables_task_ends[task] - variables_task_starts[task] == processing_time[task_to_product[task]]
    )

# Sequence
arcs = list()
for t1 in tasks:
    for t2 in tasks:
        # arcs
        if t1 != t2:
            arcs.append([
                t1,
                t2,
                variables_task_sequence[(t1, t2)]
            ])
            distance = m_cost[t1, t2]
            # cannot require the time index of task 0 to represent the first and the last position
            if t2 != 0:
                # to schedule tasks and c/o
                model.Add(
                    variables_task_ends[t1] <= variables_co_starts[t1, t2]
                ).OnlyEnforceIf(variables_task_sequence[(t1, t2)])

                model.Add(
                    variables_co_ends[t1, t2] <= variables_task_starts[t2]
                ).OnlyEnforceIf(variables_task_sequence[(t1, t2)])

                model.Add(
                    variables_co_ends[t1, t2] - variables_co_starts[t1, t2] == distance
                ).OnlyEnforceIf(variables_task_sequence[(t1, t2)])

model.AddCircuit(arcs)


# Campaigning related

var_campaign_starts = {
    c: model.NewIntVar(0, max_time, f"start_{c}") for c in campaigns
}

var_campaign_durations = {
    c: model.NewIntVar(0, max_time, f"c_duration_{c}") for c in campaigns
}

var_campaign_ends = {
    c: model.NewIntVar(0, max_time, f"c_end_{c}") for c in campaigns
}

var_campaign_presences = {
    c: model.NewBoolVar(f"c_presence_{c}") for c in campaigns
}

var_campaign_intervals = {
    c: model.NewOptionalIntervalVar(
        var_campaign_starts[c],  # campaign start
        var_campaign_durations[c],  # campaign duration
        var_campaign_ends[c],  # campaign end
        var_campaign_presences[c],  # campaign presence
        f"c_interval_{c}",
    )
    for c in campaigns
}

var_task_campaign_presences = {
    (t, c): model.NewBoolVar(f"tc_presence_{t}_{c}") for t in tasks for c in task_to_campaigns[t]
}

# add_campaign_definition_constraints
# 1. Campaign definition: Start & duration based on tasks that belongs to the campaign

for c in campaigns:
    #  1. Duration definition
    model.Add(
        var_campaign_durations[c]
        == sum(
            # processing_times[t, c2m[c]] * tc_presences[t, c]
            processing_time[task_to_product[t]] * var_task_campaign_presences[t, c]
            for t in campaign_to_tasks[c]
        )
    )
    # 2. Start-end definition
    # TODO: MinEquality ?
    for t in campaign_to_tasks[c]:
        var_campaign_starts
        var_campaign_ends

        model.Add(var_campaign_starts[c] <= variables_task_starts[t]).OnlyEnforceIf(
            var_task_campaign_presences[t, c]
        )
        model.Add(var_campaign_ends[c] >= variables_task_ends[t]).OnlyEnforceIf(
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
    task_to_campaigns[task]
    model.Add(
        sum(var_task_campaign_presences[task, campaign]
            for campaign in campaigns
            if campaign in task_to_campaigns[task]
            ) == 1
    )
# 3. No campaign overlap
model.AddNoOverlap([x for x in var_campaign_intervals.values()])


# Add campaign circuit











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
    for t1 in tasks:
        for t2 in tasks:
            if t1 != t2:
                value = solver.Value(variables_task_sequence[t1, t2])
                if value == 1 and t2 != 0:
                    print(f'{t1} --> {t2}   {task_to_product[t1]} >> {task_to_product[t2]}  cost: {m_cost[t1, t2]}')
                    print('variables_task_sequence[t1, t2]',
                          solver.Value(variables_task_sequence[t1, t2]))
                    print('variables_co_starts[t1, t2]', solver.Value(variables_co_starts[t1, t2]))
                    print('variables_co_ends[t1, t2]', solver.Value(variables_co_ends[t1, t2]))

                if value == 1 and t2 == 0:
                    print(f'{t1} --> {t2}   Closing')

elif status == cp_model.INFEASIBLE:
    print("Infeasible")
elif status == cp_model.MODEL_INVALID:
    print("Model invalid")
else:
    print(status)
