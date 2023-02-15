from ortools.sat.python import cp_model

# Initiate
model = cp_model.CpModel()

'''
task    product     type
1       A           TYPE_4
2       B           TYPE_4
'''

# 1. Data

tasks = {1}
products = {'A'}
task_to_product = {1: 'A'}
task_to_type = {1: 'TYPE_4'}
processing_time = {'A': 3}
max_time = 10
breaks = {(2, 4)}
is_break = {i: 1 if 2<=i<4 else 0 for i in range(max_time)}

# 2. Decision Variables

var_task_starts = {
    task: model.NewIntVar(0, max_time, f"task_{task}_start") for task in tasks
}

var_task_ends = {
    task: model.NewIntVar(0, max_time, f"task_{task}_end") for task in tasks
}

var_task_new_duration = {
    task: model.NewIntVar(0, max_time, f"task_{task}_delay") for task in tasks
}

var_task_duration_timeslots = {
    (task, i): model.NewBoolVar(f'task {task} uses interval {i}')
    for task in tasks
    for i in range(max_time)
}

## four
# AND pair
var_task_start_earlier_than_start = {
    (task, i): model.NewBoolVar(f'')
    for task in tasks
    for i in range(max_time)
}
var_task_end_later_than_end = {
    (task, i): model.NewBoolVar(f'')
    for task in tasks
    for i in range(max_time)
}
# # OR pair
# var_task_start_later_than_end = {
#     (task, i): model.NewBoolVar(f'')
#     for task in tasks
#     for i in range(max_time)
# }
# var_task_end_earlier_than_start = {
#     (task, i): model.NewBoolVar(f'')
#     for task in tasks
#     for i in range(max_time)
# }


for task in tasks:
    for i in range(max_time):
        model.Add(var_task_starts[task] <= i).OnlyEnforceIf(var_task_start_earlier_than_start[task, i])
        model.Add(var_task_starts[task] > i).OnlyEnforceIf(var_task_start_earlier_than_start[task, i].Not())

        model.Add(var_task_ends[task] >= i + 1).OnlyEnforceIf(var_task_end_later_than_end[task, i])
        model.Add(var_task_ends[task] < i + 1).OnlyEnforceIf(var_task_end_later_than_end[task, i].Not())

        # model.Add(var_task_starts[task] >= i+1).OnlyEnforceIf(var_task_start_later_than_end[task, i])
        # model.Add(var_task_starts[task] < i+1).OnlyEnforceIf(var_task_start_later_than_end[task, i].Not())
        #
        # model.Add(var_task_ends[task] <= i).OnlyEnforceIf(var_task_end_earlier_than_start[task, i])
        # model.Add(var_task_ends[task] > i).OnlyEnforceIf(var_task_end_earlier_than_start[task, i].Not())

        # And
        model.AddMultiplicationEquality(
            var_task_duration_timeslots[task, i],
            [var_task_start_earlier_than_start[task, i], var_task_end_later_than_end[task, i]]
        )

        # model.AddMinEquality(
        #     var_task_duration_timeslots[task, i],
        #     [var_task_start_later_than_end[task, i], var_task_end_earlier_than_start[task, i]]
        # )

# for task in tasks:
#     for i in range(max_time):
#         start_index = i
#         end_index = i + 1
#
#         # TRUE
#         model.AddLinearConstraint(var_task_starts[task], 0, start_index).OnlyEnforceIf(var_task_timeslots[task, i])
#         model.AddLinearConstraint(var_task_ends[task], end_index, max_time).OnlyEnforceIf(var_task_timeslots[task, i])
#
#         # FALSE
#         if i == 0:
#             # first time slot
#             model.AddLinearExpressionInDomain(
#             var_task_starts[task],
#             cp_model.Domain.FromIntervals([[1, max_time]])
#             ).OnlyEnforceIf(var_task_timeslots[task, i].Not())
#
#         elif i == max_time - 1:
#             # last time slot
#             model.AddLinearExpressionInDomain(
#             var_task_ends[task],
#             cp_model.Domain.FromIntervals([[0, max_time-1]])
#             ).OnlyEnforceIf(var_task_timeslots[task, i].Not())
#         else:
#             model.AddLinearExpressionInDomain(
#             x,
#             cp_model.Domain.FromIntervals([[1, start_index], [end_index+1, max_time]])
#             ).OnlyEnforceIf(var_task_timeslots[task, i].Not())


#
# task_1_start  <  task_2_start <  task_1_end
# Variables:
# 1: Overlap
# 2: (task_2_start < task_1_end)
# 3: (task_1_start < task_2_start)
# Constraints:
# 1: (task_1_start < task_2_start) -> task_1_start
# 2:


for task in tasks:
    model.Add(
            var_task_new_duration[task] == processing_time[task_to_product[task]] +
            sum(is_break[i]*var_task_duration_timeslots[task, i] for i in range(max_time))
    )


var_task_intervals = {
    task: model.NewIntervalVar(
        var_task_starts[task],
        var_task_new_duration[task],
        #processing_time[task_to_product[task]],
        var_task_ends[task],
        name=f"interval_t{task}"
    )
    for task in tasks
}


var_task_intervals_auto = {
    task: model.NewIntervalVar(
        var_task_starts[task],
        1,
        var_task_starts[task] + 1,
        name=f"interval_auto_t{task}"
    )
    for task in tasks
    if task_to_type[task] == 'TYPE_4'
}

# Add break time
variables_breaks = {
    (start, end): model.NewFixedSizeIntervalVar(start=start, size=end-start, name='a_break')
    for (start, end) in breaks
}

intervals = list(var_task_intervals_auto.values()) + list(variables_breaks.values())

# task, resource reduction for breaks
demands = [1]*len(tasks) + [1]*len(breaks)

model.AddCumulative(intervals=intervals, demands=demands, capacity=1)

# 3. Objectives

make_span = model.NewIntVar(0, max_time, "make_span")

model.AddMaxEquality(
    make_span,
    [var_task_ends[task] for task in tasks]
)

model.Minimize(make_span)




# 4. Solve

solver = cp_model.CpSolver()
status = solver.Solve(model=model)


# 5. Results

if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:

    for task in tasks:
        print(f'task {task}')
        for i in range(max_time):
            print(f'{i} {solver.Value(var_task_duration_timeslots[task, i])}')

    print('===========================  TASKS SUMMARY  ===========================')
    for task in tasks:
        print(f'Task {task} ',
              solver.Value(var_task_starts[task]), solver.Value(var_task_ends[task]),
              )
        print(solver.Value(var_task_new_duration[task]))

    print('Make-span:', solver.Value(make_span))
    # print('=======================  ALLOCATION & SEQUENCE  =======================')
    # if True:
    #     for t1 in tasks:
    #         for t2 in tasks:
    #             if t1 != t2:
    #                 value = solver.Value(var_task_seq[(t1, t2)])
    #                 print(f'{t1} --> {t2}  {value}')
    #                 # if value == 1 and t2 != 0:
    #                 #     print(f'{t1} --> {t2}   {task_to_product[t1]} >> {task_to_product[t2]}')#  cost: {m_cost[m, t1, t2]}')
    #                 # if value == 1 and t2 == 0:
    #                 #     print(f'{t1} --> {t2}   Closing')

elif status == cp_model.INFEASIBLE:
    print("Infeasible")
elif status == cp_model.MODEL_INVALID:
    print("Model invalid")
else:
    print(status)
