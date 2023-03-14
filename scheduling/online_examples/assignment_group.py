"""Solve assignment problem for given group of workers."""
from ortools.sat.python import cp_model


def main():
    # Data
    costs = [
        [90, 76, 75, 70, 50, 74],
        [35, 85, 55, 65, 48, 101],
        [125, 95, 90, 105, 59, 120],
        [45, 110, 95, 115, 104, 83],
        [60, 105, 80, 75, 59, 62],
        [45, 65, 110, 95, 47, 31],
        [38, 51, 107, 41, 69, 99],
        [47, 85, 57, 71, 92, 77],
        [39, 63, 97, 49, 118, 56],
        [47, 101, 71, 60, 88, 109],
        [17, 39, 103, 64, 61, 92],
        [101, 45, 83, 59, 92, 27],
    ]
    num_workers = len(costs)
    num_tasks = len(costs[0])

    # Allowed groups of workers:
    group1 = [
        [0, 0, 1, 1],  # Workers 2, 3
        [0, 1, 0, 1],  # Workers 1, 3
        [0, 1, 1, 0],  # Workers 1, 2
        [1, 1, 0, 0],  # Workers 0, 1
        [1, 0, 1, 0],  # Workers 0, 2
    ]

    group2 = [
        [0, 0, 1, 1],  # Workers 6, 7
        [0, 1, 0, 1],  # Workers 5, 7
        [0, 1, 1, 0],  # Workers 5, 6
        [1, 1, 0, 0],  # Workers 4, 5
        [1, 0, 0, 1],  # Workers 4, 7
    ]

    group3 = [
        [0, 0, 1, 1],  # Workers 10, 11
        [0, 1, 0, 1],  # Workers 9, 11
        [0, 1, 1, 0],  # Workers 9, 10
        [1, 0, 1, 0],  # Workers 8, 10
        [1, 0, 0, 1],  # Workers 8, 11
    ]

    group1 = [
        [1, 1, 1, 1],
    ]
    group2 = [
        [0,0,0,0],
        [0, 0, 1, 1],  # Workers 6, 7
    ]
    group3 = [
        [0, 0, 1, 1],  # Workers 10, 11
    ]

    # Model
    model = cp_model.CpModel()

    # Variables
    x = {}
    for worker in range(num_workers):
        for task in range(num_tasks):
            x[worker, task] = model.NewBoolVar(f'x[{worker},{task}]')

    x[(0,0)]

    # Constraints
    # Each worker is assigned to at most one task.
    for worker in range(num_workers):
        model.AddAtMostOne(x[worker, task] for task in range(num_tasks))

    # Each task is assigned to exactly one worker.
    for task in range(num_tasks):
        model.AddExactlyOne(x[worker, task] for worker in range(num_workers))

    # Create variables for each worker, indicating whether they work on some task.
    work = {}
    for worker in range(num_workers):
        work[worker] = model.NewBoolVar(f'work[{worker}]')

    for worker in range(num_workers):
        for task in range(num_tasks):
            model.Add(work[worker] == sum(
                x[worker, task] for task in range(num_tasks)))

    for worker in range(num_workers):
        model.Add(work[worker] == sum(
            x[worker, task] for task in range(num_tasks)))

    # Define the allowed groups of worders
    model.AddAllowedAssignments([work[0], work[1], work[2], work[3]], group1)
    model.AddAllowedAssignments([work[4], work[5], work[6], work[7]], group2)
    model.AddAllowedAssignments([work[8], work[9], work[10], work[11]], group3)

    # Objective
    objective_terms = []
    for worker in range(num_workers):
        for task in range(num_tasks):
            objective_terms.append(costs[worker][task] * x[worker, task])
    model.Minimize(sum(objective_terms))

    # Solve
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    # Print solution.
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print(f'Total cost = {solver.ObjectiveValue()}\n')
        for worker in range(num_workers):
            for task in range(num_tasks):
                if solver.BooleanValue(x[worker, task]):
                    print(f'Worker {worker} assigned to task {task}.' +
                          f' Cost = {costs[worker][task]}')
    else:
        print('No solution found.')


if __name__ == '__main__':
    main()