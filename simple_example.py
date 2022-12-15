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
assigned_task_type = collections.namedtuple('assigned_task_type',
                                            'start job index duration')

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
    model.AddNoOverlap(machine_to_intervals[machine])

# Precedences inside a job.
for job_id, job in enumerate(jobs_data):
    for task_id in range(len(job) - 1):
        model.Add(all_tasks[job_id, task_id +
                            1].start >= all_tasks[job_id, task_id].end)