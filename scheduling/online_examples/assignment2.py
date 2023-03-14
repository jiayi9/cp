from ortools.sat.python import cp_model

cost = [
    [90, 76, 75, 70, 50, 74, 12, 68],
    [35, 85, 55, 65, 48, 101, 70, 83],
    [125, 95, 90, 105, 59, 120, 36, 73],
    [45, 110, 95, 115, 104, 83, 37, 71],
    [60, 105, 80, 75, 59, 62, 93, 88],
    [45, 65, 110, 95, 47, 31, 81, 34],
    [38, 51, 107, 41, 69, 99, 115, 48],
    [47, 85, 57, 71, 92, 77, 109, 36],
    [39, 63, 97, 49, 118, 56, 92, 61],
    [47, 101, 71, 60, 88, 109, 52, 90]
    ]
# 10 row x 8 col
print(len(cost))
print([len(x) for x in cost])

# 8 tasks
sizes = [10, 7, 3, 12,
         15, 4, 11, 5]

total_size_max = 15 # for each people in the ten
num_workers = len(cost)
num_tasks = len(cost[1])
all_workers = range(num_workers)
all_tasks = range(num_tasks)

# 10 people. 8 tasks each?

model = cp_model.CpModel()

model.Proto()

total_cost = model.NewIntVar(0, 1000, 'total_cost')

model.Proto()

tmp = model.Proto()

tmp.name

tmp.constraints

x = []
for i in all_workers:
    print(f"workder {i} ")
    t = []
    for j in all_tasks:
        print(f'task {j}')
        t.append(model.NewBoolVar('x[%i,%i]' % (i, j)))
    x.append(t)

#   x    worker1   worker2   work3   work4    ...    worker10
# task1  boolean
# task2
# ...
# task8

# worker 0
x[0]

# task 0
[worker[0] for worker in x]

#[model.Add(sum(x[i][j] for i in all_workers) >= 1) for j in all_tasks]

# Each task is assigned to at least one worker.
for j in all_tasks:
    # x[worker_index][task_index]
    model.Add(sum(x[i][j] for i in all_workers) == 1)


for i in all_workers:
    # sum of task indicator * task size for a given people
    model.Add(sum(sizes[j] * x[i][j] for j in all_tasks) <= total_size_max)

model.Add(
    total_cost == sum(x[i][j] * cost[i][j] for j in all_tasks for i in all_workers)
)

model.Minimize(total_cost)

solver = cp_model.CpSolver()
status = solver.Solve(model)

if status == cp_model.OPTIMAL:
    print('Total cost = %i' % solver.ObjectiveValue())
    print()
    for i in all_workers:
        for j in all_tasks:
            if solver.Value(x[i][j]) == 1:
                print('Worker ', i, ' assigned to task ', j, '  Cost = ',
                      cost[i][j])

    print()

solver.Value(x[0][0])

print('Statistics')
print('  - conflicts : %i' % solver.NumConflicts())
print('  - branches  : %i' % solver.NumBranches())
print('  - wall time : %f s' % solver.WallTime())