"""
This is the optimize module in which the Data Scientist
should work in order to adapt the code to it's project
formulation.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Set
import copy

from ortools.sat.python import cp_model
import time

from scheduling.config import ScheduleModelFeatures, SchedulingConfig
from scheduling.data_classes import Job, Machine, OptInput, Resource, Task
from scheduling.optimize.compute_kpi import compute_kpis
from scheduling.optimize.util import (
    dediscretize_time,
    define_anticipated_time_horizon,
    discretize_resource_quantities,
    discretize_time,
    insert_changeovers_in_schedule,
    insert_resource_allocation_in_schedule,
)
import functools

logger = logging.getLogger(__name__)

COMPATIBLE_FEATURES = [
    ScheduleModelFeatures.base,
    ScheduleModelFeatures.due_dates,
    ScheduleModelFeatures.task_resources,
    ScheduleModelFeatures.task_families,
    ScheduleModelFeatures.precedences,
    ScheduleModelFeatures.changeover_resources,
]


def log_function(func):
    name = func.__name__

    @functools.wraps(func)
    def wrapped_func(*args, **kwargs):
        start_time = time.time()
        ret_val = func(*args, **kwargs)
        duration = time.time() - start_time
        logger.debug(f'Completed "{name}" in {duration:0.2f} seconds')
        return ret_val

    return wrapped_func


@dataclass
class Solution:
    """
    The necessary outputs required to define the schedule
    """

    machine_allocation: Dict[Task, Machine]  # The chosen machine for a task
    start_time: Dict[Task, float]  # The start time for a task
    end_time: Dict[Task, float]  # The end time for a task
    features: Set[
        ScheduleModelFeatures
    ]  # Set of all features applied through the model


@dataclass
class Variables:
    """
    All variables of the optimization problem, used for decisions and result values
    Type `Any` shorthand for a solver variable object, or a result float/int
    """

    # 1. job variables
    job_ends: Dict[Job, Any] = None  # indexed by (job)
    job_tardiness: Dict[Job, Any] = None  # indexed by (job)
    # 2. task variables
    task_starts: Dict[Task, Any] = None  # indexed by (task)
    task_ends: Dict[Task, Any] = None  # indexed by (task)
    # 3. task x machine variables
    machine_task_presences: Dict[
        Tuple[Machine, Task], Any
    ] = None  # indexed by (machine, task)
    machine_task_intervals: Dict[
        Tuple[Machine, Task], Any
    ] = None  # indexed by (machine, task)
    machine_task_starts: Dict[
        Tuple[Machine, Task], Any
    ] = None  # indexed by (machine, task)
    machine_task_ends: Dict[
        Tuple[Machine, Task], Any
    ] = None  # indexed by (machine, task)
    # 4. Keep tracks of the arcs
    machine_direct_precedence: Dict[Tuple[Machine, Task, Task], Any] = None


@dataclass
class OptOutput:

    """
    All needed outputs from the optimization model
    """

    status: str
    objective_value: float


def run(
    continuous_model_input: OptInput,
    model_config: SchedulingConfig,
) -> Solution:
    """
    Build and run the scheduling optimisation problem
    * Instantiates the optimisation model
    * Creates the decision variables
    * Creates the objective parts
    * Creates the constraints
    * Runs the model
    * Outputs the optimal results and the model status

    The Scheduling CP formulation can be found in the docs section

    Args:
        continuous_model_input: Instance of OptInput
        model_config: SchedulingConfig object containing configuration for solving

    Returns:
        solution: Instance of Solution class capturing the essential
        details about the schedule (start time, end time, machine allocation etc)
    """
    # Create model specs
    model = cp_model.CpModel()

    # Load model input and discretize it
    DISCRETE_PARAM = model_config.discrete_parameter
    MAX_RESOURCE_QUANTITY = model_config.max_resource_quantity
    model_input = discretize_resource_quantities(
        discretize_time(continuous_model_input, DISCRETE_PARAM), MAX_RESOURCE_QUANTITY
    )

    # Use safety_factor=10 to make sure the time horizon will always be large enough
    max_domain_time = define_anticipated_time_horizon(model_input, safety_factor=10)
    variables = create_variables(
        model=model,
        model_input=model_input,
        variables=Variables(),
        max_domain_time=max_domain_time,
    )
    add_objective(
        model=model,
        model_input=model_input,
        variables=variables,
        config=model_config,
        max_domain_time=max_domain_time,
    )
    add_model_constraints(model=model, model_input=model_input, variables=variables)

    # Solve model
    # Define solver
    solver = cp_model.CpSolver()
    if model_config.cp_solver_direct_arguments is not None:
        for k, v in model_config.cp_solver_direct_arguments.items():
            logger.debug(f"Setting cp parameter {k}={v}")
            setattr(solver.parameters, k, v)
    if model_config.stop_time_limit is not None:
        solver.parameters.max_time_in_seconds = model_config.stop_time_limit
    if model_config.cp_solver_callback_arguments is not None:
        callback = CPCallback(**model_config.cp_solver_callback_arguments)
    else:
        callback = CPCallback()

    status = solver.Solve(model=model, solution_callback=callback)
    status_decode = get_cp_model_status_name(status)
    logger.info(f"Model returned the status: '{status_decode}'")

    # Generate output
    if status == cp_model.OPTIMAL:
        logging.info("Optimal solution found")
    elif status == cp_model.FEASIBLE:
        logging.info("Sub-optimal solution found")
    elif status == cp_model.MODEL_INVALID:
        raise Exception("Model invalid")
    else:
        raise Exception("Solution validation failed")

    # Extract standardised solution
    solution = postprocess_solution(model_input, solver, variables, model_config)

    # De-discretize solution
    dediscretize_solution = dediscretize_time(
        solution, DISCRETE_PARAM, continuous_model_input
    )

    kpis_dict_dediscrete = compute_kpis(
        solution=dediscretize_solution,
        model_input=continuous_model_input,
        config=model_config,
    )

    return dediscretize_solution


def postprocess_solution(
    model_input: OptInput, solver, variables: Variables, model_config: SchedulingConfig
) -> Solution:
    """
    Extract model solution into the specified format
    Args:
        model_input: Instance of OptInput
        solver: Instance of OptOutput
        variables: Instance of Variables (decision
        variables used in the CP model)
        model_config: Instance of SchedulingConfig
    """
    # 1. Extract solution
    solution = extract_solution(solver, variables)

    # 2. Extract Machine allocation
    machine_allocation = {
        task: machine
        for task in model_input.sets.tasks
        for machine in model_input.mappings.tasks_to_machines[task]
        if solution.machine_task_presences[machine, task] == 1
    }

    # 3. Extract model features
    features = set(model_input.features).intersection(COMPATIBLE_FEATURES)

    # 4. Defining solution instance
    solution_instance = Solution(
        machine_allocation=machine_allocation,
        start_time=solution.task_starts,
        end_time=solution.task_ends,
        features=features,
    )

    return solution_instance


@log_function
def add_model_constraints(
    model: cp_model.CpModel, model_input: OptInput, variables: Variables
):
    """
    Builds all constraints and adds them to the model
    Args:
        model: Instance of CpModel
        model_input: Instance of OptInput, dataclass containing
        objects necessary to define the scheduling model
        variables: dataclass object of all decision variables
    """
    # Constraint 0: Add flexible machine-task selection
    add_flexible_machine_task_selection_constraints(model, model_input, variables)

    # Constraint 1: No overlap across the tasks for each machine
    add_no_machine_overlap_constraints(model, model_input, variables)

    # # Constraint 2: Precedence-based constraint for tasks
    if ScheduleModelFeatures.precedences in model_input.features:
        add_task_precedence_constraints(model, model_input, variables)

    # # Constraint 3: Late delivery & tardiness
    if ScheduleModelFeatures.due_dates in model_input.features:
        add_tardiness_definition_constraints(model, model_input, variables)

    # Constraint 4: Change-overs & Setup times
    add_sequence_setup_constraints(model, model_input, variables)

    # Constraint 5: Resource constraints
    add_resources_constraints_with_cumulative(model, model_input, variables)


@log_function
def add_flexible_machine_task_selection_constraints(
    model: cp_model.CpModel, model_input: OptInput, variables: Variables
):
    """
    Flexible machine-task selection
    Args:
        model: Instance of CpModel
        model_input: Instance of OptInput, dataclass containing
        objects necessary to define the scheduling model
        variables: dataclass object of all decision variables
    """
    # Load variables
    tasks = model_input.sets.tasks
    tasks_to_machines = model_input.mappings.tasks_to_machines

    # 1. Task-level variables
    for task in tasks:
        candidate_machines = tasks_to_machines[task]
        # Select exactly one machine-task presence variable
        task_candidate_machines = [
            variables.machine_task_presences[machine, task]
            for machine in candidate_machines
        ]
        model.AddExactlyOne(task_candidate_machines)
        # 2. Link candidates machine-tasks to the task
        # Candidate tasks represent the choice of machine for tasks
        for machine in candidate_machines:
            # Link the task interval with the task x machine interval, if chosen
            model.Add(
                variables.task_starts[task]
                == variables.machine_task_starts[machine, task]
            ).OnlyEnforceIf(variables.machine_task_presences[machine, task])
            model.Add(
                variables.task_ends[task] == variables.machine_task_ends[machine, task]
            ).OnlyEnforceIf(variables.machine_task_presences[machine, task])


@log_function
def add_resources_constraints_with_cumulative(
    model: cp_model.CpModel, model_input: OptInput, variables: Variables
):
    """
    Add limited resources constraints. This constraint make sure that the intervals
    are allocated up to the resource capacity.

    In this specific use-case, the resource is constant and we leverage the built-in
    method AddCumulative.

    Cumulative constraint is as follows:
     for all t:
          sum(demands[i]
            if (start(intervals[i]) <= t < end(intervals[i])) and
            (intervals[i] is present)) <= capacity
    Args:
        model: Instance of CpModel
        model_input: Instance of OptInput, dataclass containing
        objects necessary to define the scheduling model
        variables: dataclass object of all decision variables
    """
    # 1. Load variables and coefficients
    resources_to_tasks = model_input.mappings.resources_to_tasks
    resources_req = model_input.coefficients.resources_req
    resources = model_input.sets.resources
    resources_available = model_input.coefficients.resources_available
    resources_to_constraint = resources.intersection(resources_to_tasks)

    # 2. Loop across each resource
    for resource in resources_to_constraint:
        intervals, demands = [], []
        # 2.1. Define the upper bound of the resource availability
        resource_available_ub = resources_available[resource]
        # 2.2 Add tasks requirements
        for (machine, task), interval in variables.machine_task_intervals.items():
            if task in resources_to_tasks[resource]:
                intervals.append(interval)
                demands.append(resources_req[task, resource])
        # 2.3 Create cumulative constraints
        model.AddCumulative(intervals, demands, resource_available_ub)


@log_function
def add_no_machine_overlap_constraints(
    model: cp_model.CpModel, model_input: OptInput, variables: Variables
):
    """
    Overlap constraint is defined over the interval variables. model.AddNoOverlap
    takes a list of intervals as an input and make sure that 2 variables do not
    overlap one with another
    Args:
        model: Instance of CpModel
        model_input: Instance of OptInput, dataclass containing
        objects necessary to define the scheduling model
        variables: dataclass object of all decision variables
    """
    # Load variables
    machines = model_input.sets.machines
    machines_to_tasks = model_input.mappings.machines_to_tasks

    # Constraint 1: # No overlap across the tasks for each machine
    for machine in machines:
        intervals = [
            variables.machine_task_intervals[machine, task]
            for task in machines_to_tasks[machine]
        ]
        model.AddNoOverlap(intervals)


def add_task_precedence_constraints(
    model: cp_model.CpModel, model_input: OptInput, variables: Variables
):
    """
    Add constraints related to precedence rules between tasks
    If a precedence constraint exists between a tuple of 2 tasks (t1, t2),
    then the start of a task t2 should happen after the end of task t1.
    Args:
        model: Instance of CpModel
        model_input: Instance of OptInput, dataclass containing
        objects necessary to define the scheduling model
        variables: dataclass object of all decision variables
    """
    precedence = model_input.sets.precedence

    for task_before, task_after in precedence:
        # Task-precedence time: start[task_after] >= end[task_before]
        model.Add(variables.task_starts[task_after] >= variables.task_ends[task_before])


@log_function
def add_tardiness_definition_constraints(
    model: cp_model.CpModel, model_input: OptInput, variables: Variables
):
    """
    For each job, the tardiness is a measure of a delay
    between the job end time and its expected due date.
    If Job end <= Due Date: tardiness = 0
    If Job end > Due Date: tardiness = Job end - Due Date
    Args:
        model: Instance of CpModel
        model_input: Instance of OptInput, dataclass containing
        objects necessary to define the scheduling model
        variables: dataclass object of all decision variables

    """
    # Load variables
    jobs = model_input.sets.jobs
    due_dates = model_input.coefficients.due_date
    jobs_to_tasks = model_input.mappings.jobs_to_tasks

    for job in jobs:
        # 1. Define job start/end time
        model.AddMaxEquality(
            variables.job_ends[job],
            [variables.task_ends[task] for task in jobs_to_tasks[job]],
        )
        # 2. Define job tardiness
        model.AddMaxEquality(
            variables.job_tardiness[job],
            [0, variables.job_ends[job] - due_dates[job]],
        )


@log_function
def add_sequence_setup_constraints(
    model: cp_model.CpModel, model_input: OptInput, variables: Variables
):
    """
    Adds machine-setup definition & finds the correct sequence of tasks.

    Adds a circuit constraint from a sparse list of arcs that encode the graph.

    1. An arc is a tuple (task_1, task_2, boolean). The arc is selected in the
    circuit if the boolean is true. tasks must be represented as integers between 0
    and the number of tasks - 1.

    2. A circuit is a unique Hamiltonian path in a subgraph of the total graph. for
    example: (0, task_2, True), (task_2, task_3, True), (task_3, 0, True) In case a
    task n is not in the path (i.e not assigned to the machine), then there must be a
    loop arc 'task n -> task n' associated with a true boolean. Otherwise this
    constraint will fail.

    Hence, we define 4 types of arc:

    1. If it is the first task on the machine: arc from dummy node 0 to task node (0,
    task, boolean)

    2. If it is the last task on the machine: arc from task node to dummy node 0 (
    task, 0, boolean)

    3. If task 1 is before task 2: arc(task_1, task_2, True) --> start[task_2] >=
    end[task_1] + setup[task_1, task_2]

    4. If task 1 is optional (i.e flexible choice of machine): arc: (task_1, task_1,
    True) -->  Task not present on machine (i.e independent from other scheduled tasks)

    Args:
        model: Instance of CpModel
        model_input: Instance of OptInput, dataclass containing
        objects necessary to define the scheduling model
        variables: dataclass object of all decision variables

    """
    machines = model_input.sets.machines
    machines_to_tasks = model_input.mappings.machines_to_tasks

    def _distance_between_tasks_on_machine(task1, task2, machine):
        if ScheduleModelFeatures.task_families in model_input.features:
            changeover_time = model_input.coefficients.changeover_time
            tasks_to_family = model_input.mappings.tasks_to_family
            return changeover_time[
                tasks_to_family[task1], tasks_to_family[task2], machine
            ]
        return 0

    # Step 1: Loop over each machine
    for machine in machines:
        arcs = []
        # List of all feasible arcs within a machine. Arcs are boolean to specify
        # circuit from node to node
        machine_tasks = set(machines_to_tasks[machine])

        for node_1, task_1 in enumerate(machine_tasks):
            mt_1 = task_1 + "_" + machine
            # Initial arc from the dummy node (0) to a task.
            arcs.append(
                [0, node_1 + 1, model.NewBoolVar("first" + "_" + mt_1)]
            )  # if mt_1 follows dummy node 0
            # Final arc from an arc to the dummy node (0).
            arcs.append(
                [node_1 + 1, 0, model.NewBoolVar("last" + "_" + mt_1)]
            )  # if dummy node 0 follows mt_1

            # For optional task on machine (i.e other machine choice)
            # Self-looping arc on the node that corresponds to this arc.
            arcs.append(
                [
                    node_1 + 1,
                    node_1 + 1,
                    variables.machine_task_presences[(machine, task_1)].Not(),
                ]
            )

            for node_2, task_2 in enumerate(machine_tasks):
                if node_1 != node_2:
                    mt2_after_mt1 = variables.machine_direct_precedence[
                        machine, task_1, task_2
                    ]
                    # Add sequential boolean constraint: mt_2 follows mt_1
                    arcs.append([node_1 + 1, node_2 + 1, mt2_after_mt1])

                    # We add the reified precedence to link the literal with the
                    # times of the two tasks.
                    min_distance = _distance_between_tasks_on_machine(
                        task_1, task_2, machine
                    )
                    (
                        model.Add(
                            variables.task_starts[task_2]
                            >= variables.task_ends[task_1] + min_distance
                        ).OnlyEnforceIf(mt2_after_mt1)
                    )

        # Constraint to find 1 feasible circuit for each node of the arcs
        # If we consider 3 tasks and only task 2 and task 3 are planned on this machine:
        #  1. arc(0, task_2)=1
        #  2. arc(task_2, task_3) = 1 --> start[task_3] >= end[task_2] + setup
        #  3. arc(task_3, 0) = 1
        # Since task 1 is not planned on this machine: arc(task_1, task_1) = 1
        model.AddCircuit(arcs)


@log_function
def add_objective(
    model: cp_model.CpModel,
    model_input: OptInput,
    variables: Variables,
    config: SchedulingConfig,
    max_domain_time: int,
):
    """
    Builds all objective parts and adds them to the model.

    Args:
        model: Instance of CpModel
        model_input: contains the optimisation input data that
        are specific to the scheduling model
        variables: dataclass of all optimisation variables
        config: Instance of SchedulingConfig capturing essential parameters
        for running scheduling module
        max_domain_time: Integer value indicating the maximum timehorizon of scheduling

    """
    logger.info("Create model objective")
    weights = [
        config.objective_weight_makespan,
        config.objective_weight_tardiness,
        config.objective_weight_sum_of_ends,
    ]
    if any(abs(round(w) - w) > 1e-4 for w in weights):
        logger.warning(f"Objective weights are non-integer: {weights!r}")
        cp_model_obj_fn_scale_factor = config.cp_model_obj_fn_scale_factor
        weights = [int(w * cp_model_obj_fn_scale_factor) for w in weights]
        logger.warning(f"Scaling non-integer weights to: {weights!r}")
    else:
        weights = [int(w) for w in weights]

    objective = 0

    if weights[0] > 0:
        logger.info(f"Adding objective 'makespan' with weight {weights[0]}")
        make_span = model.NewIntVar(0, max_domain_time, "make_span")
        model.AddMaxEquality(
            make_span, [variables.task_ends[task] for task in variables.task_ends]
        )
        objective += weights[0] * make_span

    if weights[1] > 0:
        if ScheduleModelFeatures.due_dates in model_input.features:
            logger.info(f"Adding objective 'tardiness' with weight {weights[1]}")
            objective += weights[1] * sum(
                [variables.job_tardiness[job] for job in model_input.sets.jobs]
            )
        else:
            logger.warning("Asked for tardiness objective, but model has no tardiness")

    if weights[2] > 0:
        logger.info(f"Adding objective 'sum-of-end' with weight {weights[2]}")
        objective += weights[2] * sum(variables.task_ends.values())

    if isinstance(objective, int) and objective == 0:
        logger.warning("Asking to solve with a zero objective")
    model.Minimize(objective)


@log_function
def create_variables(
    model: cp_model.CpModel,
    model_input: OptInput,
    variables: Variables,
    max_domain_time: float,
) -> Variables:
    """
    Create all relevant optimization variables.

    Args:
        model: Instance of CpModel
        model_input: contains the optimisation input data that are specific to the UC
        variables: dataclass that contains all variables
        max_domain_time: Maximum domain time. Could be fine-tuned for each task if
          constraint related to task start/end
    Returns:
        Dict[str, IndexedVariable] : a Dict of IndexedVariable
    """
    # Load sets
    logger.info("Getting the sets for building up variables")
    jobs = model_input.sets.jobs
    tasks = model_input.sets.tasks
    tasks_to_machines = model_input.mappings.tasks_to_machines
    machines_to_tasks = model_input.mappings.machines_to_tasks
    processing_times = model_input.coefficients.processing_time

    # Create core variables
    # 1. Job variables
    variables.job_ends = {
        job: model.NewIntVar(0, max_domain_time, f"job_end_{job}") for job in jobs
    }
    if ScheduleModelFeatures.due_dates in model_input.features:
        variables.job_tardiness = {
            job: model.NewIntVar(0, max_domain_time, f"tardiness_{job}") for job in jobs
        }
    # 2. Task variables
    variables.task_starts = {
        task: model.NewIntVar(0, max_domain_time, f"start_{task}") for task in tasks
    }
    variables.task_ends = {
        task: model.NewIntVar(0, max_domain_time, f"end_{task}") for task in tasks
    }
    variables.task_durations = {
        task: model.NewIntVar(
            min(
                processing_times[task, machine] for machine in tasks_to_machines[task]
            ),  # min_duration
            max(
                processing_times[task, machine] for machine in tasks_to_machines[task]
            ),  # max_duration
            f"duration_{task}",
        )
        for task in tasks
    }
    # 3. task x machine variables
    variables.machine_task_starts = {
        (m, t): model.NewIntVar(0, max_domain_time, f"start_{m}_{t}")
        for t in tasks
        for m in tasks_to_machines[t]
    }
    variables.machine_task_ends = {
        (m, t): model.NewIntVar(0, max_domain_time, f"end_{m}_{t}")
        for t in tasks
        for m in tasks_to_machines[t]
    }
    variables.machine_task_presences = {
        (m, t): model.NewBoolVar(f"presence_{m}_{t}")
        for t in tasks
        for m in tasks_to_machines[t]
    }
    variables.machine_task_intervals = {
        (m, t): model.NewOptionalIntervalVar(
            variables.machine_task_starts[m, t],  # start of machine_task
            processing_times[t, m],  # Fixed duration of machine_task
            variables.machine_task_ends[m, t],  # end of machine_task
            variables.machine_task_presences[
                m, t
            ],  # boolean to capture if machine_task exists
            f"interval_{m}_{t}",
        )
        for t in tasks
        for m in tasks_to_machines[t]
    }
    # 4. task precedence variables
    m_t1_t2 = {
        (m, t1, t2)
        for t1 in tasks
        for m in tasks_to_machines[t1]
        for t2 in machines_to_tasks[m]
        if t1 != t2
    }
    variables.machine_direct_precedence = {
        (m, t1, t2): model.NewBoolVar(f"{t2} follows {t1} on {m}")
        for (m, t1, t2) in m_t1_t2
    }

    for var_name, var in variables.__dict__.items():
        if var:
            logger.info(
                "Created variable '{}' with {} entries".format(var_name, len(var))
            )
    logger.info(
        "Created {} types of variables with {} entries".format(
            len(variables.__dict__.keys()),
            sum([len(v) for v in variables.__dict__.values() if v]),
        )
    )
    return variables


def extract_solution(solver: cp_model.CpSolver, variables):
    """Extracts the solution from the variables
    Three kinds of variables have been considered
    Integer, Boolean and Interval Variable
    Incase of other types of variables,
    this function would require modification
    """

    # Initializing solution class
    solution = copy.deepcopy(variables)
    for varname, vardict in variables.__dict__.items():
        setattr(solution, varname, None)

    # Assigning variable values
    for varname, vardict in variables.__dict__.items():
        if vardict is not None:
            setattr(
                solution,
                varname,
                {
                    k: solver.Value(v) if type(v) not in [cp_model.IntervalVar] else v
                    for k, v in vardict.items()
                },
            )
        else:
            logger.warning(f"Variable '{varname}' is defined but not used in model")
    return solution


class CPCallback(cp_model.CpSolverSolutionCallback):
    """Display the objective value and time of intermediate solutions."""

    def __init__(self, solution_limit=None):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__solution_count = 0
        self.__start_time = time.time()
        self.solution_limit = solution_limit

    def on_solution_callback(self):
        """Called on each new solution."""
        current_time = time.time()
        obj = self.ObjectiveValue()
        logger.debug(
            "Solution %i, time = %0.2f s, objective = %i"
            % (self.__solution_count, current_time - self.__start_time, obj)
        )

        # ExperimentTracker().log_metric("objective", obj)
        self.__solution_count += 1
        if (
            self.solution_limit is not None
            and self.__solution_count >= self.solution_limit
        ):
            self.StopSearch()

    def solution_count(self):
        """Returns the number of solutions found."""
        return self.__solution_count


def get_cp_model_status_name(status):
    """
    Extracts and returns the model status
    """
    cp_status = {
        cp_model.UNKNOWN: "UNKNOWN",
        cp_model.MODEL_INVALID: "MODEL_INVALID",
        cp_model.FEASIBLE: "FEASIBLE",
        cp_model.INFEASIBLE: "INFEASIBLE",
        cp_model.OPTIMAL: "OPTIMAL",
    }

    status_decode = cp_status.get(status, "Unknown status")
    return status_decode
