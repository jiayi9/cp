import collections
from ortools.sat.python import cp_model

jobs_data = [  # task = (machine_id, processing_time).
    [(0, 3), (1, 2), (2, 2)],  # Job0
    [(0, 2), (2, 1), (1, 4)],  # Job1
    [(1, 4), (2, 3)]  # Job2
]

machines_count = 1 + max(task[0] for job in jobs_data for task in job)

all_machines = range(machines_count)

# Computes horizon dynamically as the sum of all durations.
horizon = sum(task[1] for job in jobs_data for task in job)

model = cp_model.CpModel()

# Named tuple to store information about created variables.
# task_type(start=, end=, interval=)
task_type = collections.namedtuple('task_type', 'start end interval')
# Named tuple to manipulate solution information.


#
# Point = collections.namedtuple('Point', ['x', 'y'])
# p = Point(11, y=22)
# Point(11, 22)
# Point(x=33, 22)
# p.x
# p.y

# Creates job intervals and add to the corresponding machine lists.
all_tasks = {}
machine_to_intervals = collections.defaultdict(list)
#
# machine_to_intervals[(1,1)] = [123,1231,2312,3]
#
# machine_to_intervals[1] = 1
# machine_to_intervals[2] = 1
# machine_to_intervals[3] = [1,2,3]
#
# machine_to_intervals[(1, 1)]
#
#
# s = 'mississippi'
# d = collections.defaultdict(int)
# for k in s:
#     print('-------')
#     print(k)
#     print(d[k])
#     d[k] += 1
#     print(d[k])
#
# s = 'mississippi'
# d = {}
# for k in s:
#     print(k)
#     d[k] += 1
#
# d = {}
# d['x'] = 1
#

for job_id, job in enumerate(jobs_data):
    for task_id, task in enumerate(job):

        machine = task[0]
        duration = task[1]
        print(f"job_id: {job_id}     job: {job}")
        print(f"task_id: {task_id}    task: {task}")
        print('---------------')
        suffix = '_%i_%i' % (job_id, task_id)
        machine_time = f"machine: {machine}     duration: {duration}"
        print(suffix, machine_time)

        # define int variables here
        start_var = model.NewIntVar(0, horizon, 'start' + suffix)
        end_var = model.NewIntVar(0, horizon, 'end' + suffix)

        # define interval variable
        interval_var = model.NewIntervalVar(start_var, duration, end_var,
                                            'interval' + suffix)

        # fill the dict
        all_tasks[job_id, task_id] = task_type(start=start_var,
                                               end=end_var,
                                               interval=interval_var)
        machine_to_intervals[machine].append(interval_var)

"""
machine_to_intervals
  - 0
    - interval_var_1: model.NewIntervalVar(
                        start_var: model.NewIntVar, 
                        duration: int, 
                        end_var: model.NewIntVar, 
                        'interval_0_1'
                        )
    - interval_var_2
    - interval_var_3
  - 1
    - interval_var_4
  - 2
    - interval_var_5
"""


# Create and add disjunctive constraints.
for machine in all_machines:
    print(machine)
    the_list_of_inverval_variables = machine_to_intervals[machine]
    print(the_list_of_inverval_variables)
    model.AddNoOverlap(machine_to_intervals[machine])

# Precedences inside a job.
for job_id, job in enumerate(jobs_data):
    for task_id in range(len(job) - 1):
        model.Add(all_tasks[job_id, task_id +
                            1].start >= all_tasks[job_id, task_id].end)


# Makespan objective.
obj_var = model.NewIntVar(0, horizon, 'makespan')

LIST = [all_tasks[job_id, len(job) - 1].end for job_id, job in enumerate(jobs_data)]

model.AddMaxEquality(
    obj_var,
    LIST
)

model.Minimize(obj_var)

solver = cp_model.CpSolver()
status = solver.Solve(model)

assigned_task_type = collections.namedtuple('assigned_task_type',
                                            'start job index duration')

if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    print('Solution:')
    # Create one list of assigned tasks per machine.
    assigned_jobs = collections.defaultdict(list)
    for job_id, job in enumerate(jobs_data):
        for task_id, task in enumerate(job):
            machine = task[0]
            assigned_jobs[machine].append(
                assigned_task_type(start=solver.Value(
                    all_tasks[job_id, task_id].start),
                                   job=job_id,
                                   index=task_id,
                                   duration=task[1]))

    # Create per machine output lines.
    output = ''
    for machine in all_machines:
        # Sort by starting time.
        assigned_jobs[machine].sort()
        sol_line_tasks = 'Machine ' + str(machine) + ': '
        sol_line = '           '

        for assigned_task in assigned_jobs[machine]:
            name = 'job_%i_task_%i' % (assigned_task.job,
                                       assigned_task.index)
            # Add spaces to output to align columns.
            sol_line_tasks += '%-20s' % name

            start = assigned_task.start
            duration = assigned_task.duration
            sol_tmp = '[%i,%i]' % (start, start + duration)
            # Add spaces to output to align columns.
            sol_line += '%-20s' % sol_tmp

        sol_line += '\n'
        sol_line_tasks += '\n'
        output += sol_line_tasks
        output += sol_line

    # Finally print the solution found.
    print(f'Optimal Schedule Length: {solver.ObjectiveValue()}')
    print(output)
else:
    print('No solution found.')