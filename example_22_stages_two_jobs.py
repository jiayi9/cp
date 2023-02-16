from ortools.sat.python import cp_model

# Initiate
model = cp_model.CpModel()

jobs = {1, 2}
stages = {1, 2, 3}
tasks = {(job, stage) for job in jobs for stage in stages}

processing_time = 3
max_time = 20

# 1. Jobs
var_job_starts = {
    job: model.NewIntVar(0, max_time, f"job_{job}_start") for job in jobs
}

var_job_ends = {
    job: model.NewIntVar(0, max_time, f"job_{job}_end") for job in jobs
}

# 2. Tasks
var_task_starts = {
    (job, stage): model.NewIntVar(0, max_time, f"job_{job}_stage_{stage}_start") for (job, stage) in tasks
}

var_task_ends = {
    (job, stage): model.NewIntVar(0, max_time, f"job_{job}_stage_{stage}_end") for (job, stage) in tasks
}


for job in jobs:
    model.AddMinEquality(var_job_starts[job], [var_task_starts[job, stage] for stage in stages])
    model.AddMaxEquality(var_job_ends[job], [var_task_ends[job, stage] for stage in stages])

    for stage in stages:
        if stage == len(stages):
            continue
        model.Add(var_task_ends[job, stage] <= var_task_starts[job, stage + 1])


var_task_intervals = {
    (job, stage): model.NewIntervalVar(
        var_task_starts[job, stage],
        processing_time,
        var_task_ends[job, stage],
        name=f"interval_job_{job}_stage_{stage}"
    )
    for job in jobs
    for stage in stages
}

for stage in stages:
    model.AddNoOverlap(var_task_intervals[job, stage] for job in jobs)

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

    print('===========================  TASKS SUMMARY  ===========================')
    for job in jobs:
        print(f"job_{job}  start: {solver.Value(var_job_starts[job])}   end:{solver.Value(var_job_ends[job])}")
        for stage in stages:
            print(f'Stage {stage} ',
                  solver.Value(var_task_starts[job, stage]), solver.Value(var_task_ends[job, stage]),
                  )

    print('Make-span:', solver.Value(make_span))

elif status == cp_model.INFEASIBLE:
    print("Infeasible")
elif status == cp_model.MODEL_INVALID:
    print("Model invalid")
else:
    print(status)
