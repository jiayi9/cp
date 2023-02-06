"""
This is the optimize module in which the Data Scientist
should work in order to adapt the code to it's project
formulation.
"""
import logging
import random
from dataclasses import dataclass
from typing import Any, Dict, Tuple

from job_shop_model.config import SimpleModelConfig
from job_shop_model.data_classes import Campaign, Job, Machine, OptInput, Solution, Task
from ortools.sat.python import cp_model

from vmx_lib.util import log_function
from vmx_lib.vmx_data_context.vmx_context import VMXContext

logger = logging.getLogger(__name__)

cp_status = {
    0: "UNKNOWN",
    1: "MODEL_INVALID",
    2: "FEASIBLE",
    3: "INFEASIBLE",
    4: "OPTIMAL",
}
OPTIONAL_JOBS = False
random.seed = 1


@dataclass
class Variables:
    """
    All variables of the optimization problem, used for decisions and result values
    Type `Any` shorthand for a solver variable object, or a result float/int
    """

    # 1. job variables
    job_starts: Dict[Job, Any] = None  # indexed by (job)
    job_ends: Dict[Job, Any] = None  # indexed by (job)
    job_tardiness: Dict[Job, Any] = None  # indexed by (job)
    job_completions: Dict[Job, Any] = None  # indexed by (job)
    # 2. task variables
    task_starts: Dict[Task, Any] = None  # indexed by (task)
    task_ends: Dict[Task, Any] = None  # indexed by (task)
    task_presences: Dict[Task, Any] = None  # indexed by (task)
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
    # 4. campaign variables
    campaign_starts: Dict[Campaign, Any] = None  # indexed by (campaign)
    campaign_durations: Dict[Campaign, Any] = None  # indexed by (campaign)
    campaign_ends: Dict[Campaign, Any] = None  # indexed by (campaign)
    campaign_presences: Dict[Campaign, Any] = None  # indexed by (campaign)
    campaign_intervals: Dict[Campaign, Any] = None  # indexed by (campaign)
    tc_presences: Dict[Tuple[Task, Campaign], Any] = None  # indexed by (machine, task)


@log_function
def create_model(
    model_config: SimpleModelConfig,
) -> Tuple[cp_model.CpModel, cp_model.CpSolver]:
    """
    Create an instance of the model, and update the solver configs

    Args:
        model_config: SimpleModelConfig object containing configuration for solving
    Returns:
        A Tuple containing:
        * CpModel
        * CpSolver
    """
    logger.info("Creating model")

    # define model
    model = cp_model.CpModel()

    # Define solver
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = model_config.stop_time_limit
    solver.parameters.num_search_workers = 8  # TODO: Investigate VMX model_config
    # solver.parameters.log_search_progress = True
    solver.parameters.random_seed = 10

    return model, solver


def run(
    context: VMXContext, model_input: OptInput, model_config: SimpleModelConfig
) -> Solution:
    """Build and run the Planning UC optimisation problem

    * Instantiates the optimisation model
    * Creates the decision variables
    * Creates the objective parts
    * Creates the constraints
    * Update the search strategy
    * Runs the model
    * Outputs the optimal results and the model status

    The Planning UC formulation can be found in the docs section

    Args:
        context: Context file
        model_input: contains the optimisation input data that are specific to
            the UC
        model_config: SimpleModelConfig object containing configuration for solving

    Returns:
        A Tuple containing:
        * A dictionary mapping the decision variables to their optimal values
        * The model status after running the optimisation

    """
    # Create model specs
    model, solver = create_model(model_config=model_config)

    variables = create_variables(
        model=model,
        model_input=model_input,
        variables=Variables(),
        max_domain_time=10000,
    )
    add_objective(model=model, model_input=model_input, variables=variables)
    add_model_constraints(model=model, model_input=model_input, variables=variables)
    # add_search_strategy(model=model, opt_vars=opt_vars)  # TODO: Define the best search strategy

    # Solve model
    status = solver.Solve(
        model=model, solution_callback=cp_model.ObjectiveSolutionPrinter()
    )
    # solver_summary = get_dict_from_txt(solver.ResponseStats())  # TODO: How to implement solver into results

    # Generate output
    if status == cp_model.OPTIMAL:
        logging.info("Optimal solution found")
    elif status == cp_model.FEASIBLE:
        logging.info("Sub-optimal solution found")
    else:
        # TODO: Define what to do when no solution is found ?
        raise Exception("Solution validation failed")

    # Extract standardised solution
    solution = postprocess_solution(model_input, solver, variables)

    return solution


def extract_solution(solver: cp_model.CpSolver, variables: Variables) -> Variables:
    """Extract the solver solution"""
    solution = Variables()
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


def postprocess_solution(model_input, solver, variables: Variables) -> Solution:
    """Extract model solution into the specified format"""
    # 1. Extract solution
    solution = extract_solution(solver, variables)

    # 2. Extract Machine allocation
    machine_allocation = {
        task: machine
        for task in model_input.sets.tasks
        for machine in model_input.mappings.tasks_to_machines[task]
        if solution.machine_task_presences[machine, task] == 1
    }

    return Solution(
        machine_allocation=machine_allocation,
        start_time=solution.task_starts,
        end_time=solution.task_ends,
    )


def get_campaign_info(model_input, solver, opt_vars):
    """ Get campaign info"""
    # 1. Load data
    tasks = model_input.sets.tasks

    # 2. Get campaign presence
    task_to_campaign = {
        t: c
        for t, c in opt_vars["tc_presences"]
        if solver.Value(opt_vars["tc_presences"][t, c])
    }
    campaigns = set(task_to_campaign.values())
    campaign_to_tasks = {
        c: [t for t in tasks if task_to_campaign[t] == c]
        for c in campaigns
    }

    # 3. Get campaign values
    campaign_presences = {
        c: solver.Value(opt_vars["c_presences"][c])
        for c in opt_vars["c_presences"]
    }
    campaign_starts = {
        c: solver.Value(opt_vars["c_starts"][c])
        for c in opt_vars["c_starts"]
        if campaign_presences[c]
    }
    campaign_ends = {
        c: solver.Value(opt_vars["c_ends"][c])
        for c in opt_vars["c_ends"]
        if campaign_presences[c]
    }
    campaign_duration = {
        c: solver.Value(opt_vars["c_durations"][c])
        for c in opt_vars["c_durations"]
        if campaign_presences[c]
    }

    return {
        c: {'start': campaign_starts[c],
            'end': campaign_ends[c],
            'duration': campaign_duration[c],
            'tasks': campaign_to_tasks[c]
            }
        for c in campaigns
    }


@log_function
def add_search_strategy(model, opt_vars: Dict[str, Dict]):
    """Function to fine-tune the search strategy"""
    task_start_vars = opt_vars["task_start_vars"]
    model.AddDecisionStrategy(
        [task_start_vars[k] for k in task_start_vars],
        cp_model.CHOOSE_MIN_DOMAIN_SIZE,
        cp_model.SELECT_MIN_VALUE,
    )

    task_end_vars = opt_vars["task_end_vars"]
    model.AddDecisionStrategy(
        [task_end_vars[k] for k in task_end_vars],
        cp_model.CHOOSE_MIN_DOMAIN_SIZE,
        cp_model.SELECT_MIN_VALUE,
    )


@log_function
def add_model_constraints(
    model: cp_model.CpModel, model_input: OptInput, variables: Variables
):
    """
    Builds all constraints and adds them to the model.

    Args:
        model: Instance of CpModel
        model_input: contains the optimisation input data that are specific to the UC
        variables: dataclass object of all optimisation variables
    """
    # TODO: Leverage logger feature from mip_v1
    # Constraint 0: Add job completion constraint
    add_job_completion_constraints(model, model_input, variables)

    # Constraint 0: Add flexible machine-task selection
    add_flexible_machine_task_selection_constraints(model, model_input, variables)

    # Constraint 1: No overlap across the tasks for each machine
    add_no_machine_overlap_constraints(model, model_input, variables)

    # # Constraint 2: Precedence-based constraint for tasks
    add_task_precedence_constraints(model, model_input, variables)

    # # Constraint 3: Late delivery & tardiness
    add_tardiness_definition_constraints(model, model_input, variables)

    # Constraint 4: Define campaigns
    add_campaign_constraints(model, model_input, variables)
    # Constraint 4: Change-overs & Setup times
    # add_sequence_setup_constraints(model, model_input, opt_vars)

    # Constraint 5: Resource constraints
    # add_resources_constraints_with_cumulative(model, model_input, opt_vars)
    # add_resources_constraints_with_decomposition(model, model_input, opt_vars)

    # Constraint 6: Capture inventory & inventory limit
    # add_inventory_constraints(model, model_input, opt_vars)


@log_function
def add_campaign_constraints(
    model: cp_model.CpModel, model_input: OptInput, variables: Variables
):
    """
    Adds campaign constraints.
    """
    # 1. Create dummy data  TODO: Move to pre-processing
    dummy_data = create_campaign_dummy_data(model_input=model_input)

    # 2. Create variables
    update_campaign_variables(
        dummy_data=dummy_data,
        variables=variables,
        model=model,
        model_input=model_input,
        max_domain_time=10000,
    )

    # 3. Create constraints
    # 3.1 Campaign definition
    add_campaign_definition_constraints(
        dummy_data=dummy_data, variables=variables, model=model, model_input=model_input
    )
    add_campaign_circuit_constraints(
        dummy_data=dummy_data, variables=variables, model=model, model_input=model_input
    )
    add_campaign_length_constraints(
        dummy_data=dummy_data, variables=variables, model=model
    )


@log_function
def add_campaign_length_constraints(
    dummy_data: Dict, model: cp_model.CpModel, variables: Variables
):
    # 1. Load data
    campaigns = dummy_data["campaigns"]
    campaign_to_tasks = dummy_data["campaign_to_tasks"]
    campaign_size = dummy_data["campaign_size"]
    campaign_duration = dummy_data["campaign_duration"]
    tc_presences = variables.tc_presences
    c_durations = variables.campaign_durations
    c_presences = variables.campaign_presences

    # 2. Create constraints
    for c in campaigns:
        #  1. Campaign size
        model.Add(
            sum(tc_presences[t, c] for t in campaign_to_tasks[c]) <= campaign_size[c]
        )
        # 2. Campaign duration
        model.Add(c_durations[c] <= campaign_duration[c]
                  ).OnlyEnforceIf(c_presences[c])


@log_function
def add_campaign_circuit_constraints(
    dummy_data: Dict,
    model: cp_model.CpModel,
    model_input: OptInput,
    variables: Variables,
):
    c_presences = variables.campaign_presences
    machines = model_input.sets.machines
    c_starts = variables.campaign_starts
    c_ends = variables.campaign_ends
    m2c = dummy_data["machine_to_campaigns"]
    # Step 1: Loop over each machine
    for machine in machines:
        arcs = []  # List of all feasible arcs within a machine.
        campaigns = set(m2c[machine])
        for node_1, campaign_1 in enumerate(campaigns):
            mc1 = machine + "_" + campaign_1
            # Initial arc from the dummy node (0) to a task.
            arcs.append(
                [0, node_1 + 1, model.NewBoolVar("first" + "_" + mc1)]
            )  # if mc1 follows dummy node 0
            # Final arc from an arc to the dummy node (0).
            arcs.append(
                [node_1 + 1, 0, model.NewBoolVar("last" + "_" + mc1)]
            )  # if dummy node 0 follows mc1

            # For optional campaigns on machine
            arcs.append([node_1 + 1, node_1 + 1, c_presences[campaign_1].Not()])

            for node_2, campaign_2 in enumerate(campaigns):
                if node_1 == node_2:
                    continue
                mc2 = machine + "_" + campaign_2
                mc2_after_mc1 = model.NewBoolVar(
                    f"{mc2} follows {mc1}"
                )  # bool: mc2 follows mc1
                arcs.append([node_1 + 1, node_2 + 1, mc2_after_mc1])

                # We add the reified precedence to link the literal with the times of the two tasks.
                min_distance = distance_between_campaigns(
                    dummy_data=dummy_data,
                    from_campaign=campaign_1,
                    to_campaign=campaign_2,
                )

                model.Add(
                    c_starts[campaign_2] >= c_ends[campaign_1] + min_distance
                ).OnlyEnforceIf(
                    mc2_after_mc1
                ).OnlyEnforceIf(
                    c_presences[campaign_1]
                ).OnlyEnforceIf(
                    c_presences[campaign_2]
                )
        # Constraint to find 1 feasible circuit for each node of the arcs
        model.AddCircuit(arcs)


def distance_between_campaigns(dummy_data, from_campaign, to_campaign):
    """Returns the distance from a campaign to another campaign"""
    campaign_to_family = dummy_data["campaign_to_family"]

    # 1. Fixed setup: Fixed setup time at the end of each campaign
    fixed_setup = 1

    # 2. Family setup: Additional setup if there is a family change
    from_family = campaign_to_family[from_campaign]
    to_family = campaign_to_family[to_campaign]
    family_setup = 2 if from_family != to_family else 0

    return fixed_setup + family_setup


@log_function
def add_campaign_definition_constraints(
    dummy_data: Dict,
    model: cp_model.CpModel,
    model_input: OptInput,
    variables: Variables,
):
    """
    Add campaign definition
    """
    # 1. Load data
    machines = model_input.sets.machines
    machines_to_tasks = model_input.mappings.machines_to_tasks
    processing_times = model_input.coefficients.processing_time
    campaigns = dummy_data["campaigns"]
    m2c = dummy_data["machine_to_campaigns"]
    campaign_to_tasks = dummy_data["campaign_to_tasks"]
    c2m = dummy_data["campaign_to_machine"]
    t2c = dummy_data["task_to_campaigns"]
    c_intervals = variables.campaign_intervals
    c_starts = variables.campaign_starts
    c_ends = variables.campaign_ends
    c_presences = variables.campaign_presences
    c_durations = variables.campaign_durations
    machine_task_presence_vars = variables.machine_task_presences
    machine_task_start_vars = variables.machine_task_starts
    machine_task_end_vars = variables.machine_task_ends
    tc_presences = variables.tc_presences

    # 1. Campaign definition: Start & duration based on tasks that belongs to the campaign
    for c in campaigns:
        #  1. Duration definition
        model.Add(
            c_durations[c]
            == sum(
                processing_times[t, c2m[c]] * tc_presences[t, c]
                for t in campaign_to_tasks[c]
            )
        )
        # 2. Start-end definition
        # TODO: MinEquality ?
        for t in campaign_to_tasks[c]:
            model.Add(c_starts[c] <= machine_task_start_vars[c2m[c], t]).OnlyEnforceIf(
                tc_presences[t, c]
            )
            model.Add(c_ends[c] >= machine_task_end_vars[c2m[c], t]).OnlyEnforceIf(
                tc_presences[t, c]
            )
        # 3. Link c & tc presence: If 1 task is scheduled on a campaign -> presence[t, c] = 1 ==> presence[c] == 1
        model.AddMaxEquality(
            c_presences[c], [tc_presences[t, c] for t in campaign_to_tasks[c]]
        )

    # 2. Definition of the bool var: if a task belongs to a campaign
    for m in machines:
        # 1. One task belongs to at most 1 campaign
        for t in machines_to_tasks[m]:
            model.Add(
                machine_task_presence_vars[m, t]
                == sum(tc_presences[t, c] for c in m2c[m] if c in t2c[t])
            )
        # 2. No campaign overlap
        campaign_intervals = [c_intervals[c] for c in m2c[m]]
        model.AddNoOverlap(campaign_intervals)


@log_function
def create_campaign_dummy_data(
    model_input: OptInput,
):
    # 1. Load data
    jobs = model_input.sets.jobs
    processing_time = model_input.coefficients.processing_time
    machines = model_input.sets.machines
    tasks = model_input.sets.tasks
    machines_to_tasks = model_input.mappings.machines_to_tasks
    task_to_job = model_input.mappings.task_to_job

    # machines = {'m1', 'm2'}
    # tasks = {'t1', 't2', 't3'}
    # machines_to_tasks = {'m1': ['t1', 't2'], 'm2':['t2', 't3']}

    # family -> job -> task       X        machine

    # 2. Create dummy data
    machines_to_jobs = {
        machine: set([task_to_job[task] for task in machines_to_tasks[machine]])
        for machine in machines_to_tasks
    }
    job_to_family = {job: random.randint(0, 1) for job in jobs}
    families = set(job_to_family.values())
    task_to_family = {task: job_to_family[task_to_job[task]] for task in tasks}
    family_to_tasks = {
        family: [task for task in tasks if task_to_family[task] == family]
        for family in families
    }
    family_to_jobs = {
        family: [job for job in jobs if job_to_family[job] == family]
        for family in families
    }
    max_machine_family_campaigns = {
        (machine, family): len(
            [j for j in family_to_jobs[family] if j in machines_to_jobs[machine]]
        )
        for family in families
        for machine in machines
    }
    machine_family_campaigns = {
        (machine, family, f"{machine}_{family}_{campaign}")
        for machine, family in max_machine_family_campaigns
        for campaign in list(range(0, max_machine_family_campaigns[machine, family]))
        if max_machine_family_campaigns[machine, family] > 0
    }
    campaign_to_family = {c: f for m, f, c in machine_family_campaigns}
    campaign_to_machine = {c: m for m, f, c in machine_family_campaigns}
    campaigns = {c for m, f, c in machine_family_campaigns}
    family_to_campaigns = {
        f: [c for c in campaigns if campaign_to_family[c] == f] for f in families
    }
    machine_to_campaigns = {
        m: [c for c in campaigns if campaign_to_machine[c] == m] for m in machines
    }
    tasks_to_machines = model_input.mappings.tasks_to_machines
    task_to_campaigns = {
        t: [
            c
            for c in campaigns
            if campaign_to_family[c] == task_to_family[t]
            if campaign_to_machine[c] in tasks_to_machines[t]
        ]
        for t in tasks
    }
    campaign_size = {c: 2 for c in campaigns}
    campaign_duration = {c: 2 * max(processing_time.values()) for c in campaigns}
    campaign_to_tasks = {
        c: [t for t in tasks if c in task_to_campaigns[t]] for c in campaigns
    }
    return {
        "job_to_family": job_to_family,
        "task_to_family": task_to_family,
        "machines_to_jobs": machines_to_jobs,
        "families": families,
        "family_to_jobs": family_to_jobs,
        "family_to_tasks": family_to_tasks,
        "campaigns": campaigns,
        "campaign_to_family": campaign_to_family,
        "campaign_to_machine": campaign_to_machine,
        "machine_to_campaigns": machine_to_campaigns,
        "family_to_campaigns": family_to_campaigns,
        "task_to_campaigns": task_to_campaigns,
        "campaign_to_tasks": campaign_to_tasks,
        "campaign_size": campaign_size,
        "campaign_duration": campaign_duration
    }


def update_campaign_variables(
    dummy_data: Dict,
    variables: Variables,
    model: cp_model.CpModel,
    model_input: OptInput,
    max_domain_time: float = 10000,
) -> None:
    """Update campaign variables.
    Args:
        model: Instance of CpModel
        model_input: contains the optimisation input data that are specific to the UC
        max_domain_time: Maximum domain time. Could be fine-tuned for each task if constraint related to task start/end
    Returns:
        Dict[str, IndexedVariable] : a Dict of IndexedVariable
    """
    # 1. Load data
    tasks = model_input.sets.tasks
    t2c = dummy_data["task_to_campaigns"]
    campaigns = dummy_data["campaigns"]

    # 2. Create variables
    # 2.1 Campaign vars
    variables.campaign_starts = {
        c: model.NewIntVar(0, max_domain_time, f"start_{c}") for c in campaigns
    }
    variables.campaign_durations = {
        c: model.NewIntVar(0, max_domain_time, f"c_duration_{c}") for c in campaigns
    }
    variables.campaign_ends = {
        c: model.NewIntVar(0, max_domain_time, f"mc_end_{c}") for c in campaigns
    }
    variables.campaign_presences = {
        c: model.NewBoolVar(f"c_presence_{c}") for c in campaigns
    }
    variables.campaign_intervals = {
        c: model.NewOptionalIntervalVar(
            variables.campaign_starts[c],  # campaign start
            variables.campaign_durations[c],  # campaign duration
            variables.campaign_ends[c],  # campaign end
            variables.campaign_presences[c],  # campaign presence
            f"c_interval_{c}",
        )
        for c in campaigns
    }
    # 2.2 Define task-campaign bool
    variables.tc_presences = {
        (t, c): model.NewBoolVar(f"tc_presence_{t}_{c}") for t in tasks for c in t2c[t]
    }


@log_function
def add_inventory_constraints(
    model: cp_model.CpModel, model_input: OptInput, opt_vars: Dict[str, Dict]
):
    """
    The rank of a task is the ordering of a task.
    For instance, for 2 tasks: task 1 & task 2, such as end_time[task_1] >= end_time[task_2]:
    task_rank[task_1] = 0, rank_end_time[0] = end_time[task_1]
    task_rank[task_2] = 1, rank_end_time[1] = end_time[task_2]

    Balance definition:
    For rank in list_ranks:
         reservoir[rank] >= reservoir[rank-1]
                            + waste_produced[rank]
                             - (rank_end_time[rank] - end_time[rank -1]) x release speed

    # Reservoir constraint - Defined by the lower & upper bound of the variable
    Reservoir >= 0
    Reservoir <= max_reservoir

    Variables:
    # Integer variables:
    reservoir[rank]: Keep track of the reservoir inventory at each task rank
    rank_end_time[rank]: Time pointer for each rank (= end time of the task of same rank)
    waste_produced[rank]: Waste produced at rank
    task_rank[task]: Rank of a specific task

    # Boolean variables
    same_task_rank[task, rank]: Capture the rank for a task. 1 task can only have 1 rank. 1 rank can only have 1 task

    Constraints:
    1. Each task should have a different rank (all different)
    2. rank_end_time[rank] >= rank_end_time[rank - 1]
    3. task_rank[task] == rank --> rank_end_time[rank] == task_end_vars[task]
    4. 1 rank can only bo associated to 1 task

    """
    # Load variables & coefficients
    task_end_vars = opt_vars["task_end_vars"]
    tasks = model_input.sets.tasks
    waste_quantity = {task: 20 for task in tasks}
    release_speed = 1

    # Step 1: Create Variables
    # 1.1: Rank-level variables: A rank is the ordering of the task based on its end_time
    list_ranks = list(range(0, len(tasks)))
    reservoir_vars = {}
    rank_end_time = {}
    waste_produced_vars = {}
    for rank in list_ranks:
        reservoir_vars[rank] = model.NewIntVar(0, 40, f"Reservoir_rank_{rank}")
        rank_end_time[rank] = model.NewIntVar(0, 10000, f"rank_end_time_{rank}")
        waste_produced_vars[rank] = model.NewIntVar(0, 10000, f"waste_produced_{rank}")
    # 1.2: Task-level variable: For each task, need to define its rank
    task_rank_vars = {}
    for task in tasks:
        task_rank_vars[task] = model.NewIntVar(0, len(tasks) - 1, f"task_rank_{task}")
    # 1.3: task x Rank
    same_task_rank_vars = {}
    for rank in list_ranks:
        for task in tasks:
            same_task_rank_vars[task, rank] = model.NewBoolVar(
                f"same_rank_{task}_{rank}"
            )

    # Step 2: Add constraints
    # 2.1 Each task has a different ranking
    model.AddAllDifferent([task_rank_vars[task] for task in task_rank_vars])
    # for task in tasks:
    #     # Each task should be associated to exactly 1 rank [Optional]
    #     model.AddExactlyOne([same_task_rank_vars[task, rank] for rank in list_ranks])
    for rank in list_ranks:
        # Each rank should be associated to exactly 1 task
        model.AddExactlyOne([same_task_rank_vars[task, rank] for task in tasks])
        # Rank definition
        for task in tasks:
            # Task-rank definition & implication
            model.Add(task_rank_vars[task] == rank).OnlyEnforceIf(
                same_task_rank_vars[task, rank]
            )
            model.Add(rank_end_time[rank] == task_end_vars[task]).OnlyEnforceIf(
                same_task_rank_vars[task, rank]
            )
            # Waste quantity
            model.Add(waste_produced_vars[rank] == waste_quantity[task]).OnlyEnforceIf(
                same_task_rank_vars[task, rank]
            )

        if rank > 0:
            model.Add(rank_end_time[rank] >= rank_end_time[rank - 1])
        # Balance constraints
        if rank == 0:
            initial_balance = model.NewConstant(30)
            #  (end_time[task_rank] - end_time[task_rank -1) x release speed
            model.Add(
                reservoir_vars[rank]
                >= (
                    initial_balance
                    + waste_produced_vars[rank]  # task-dependent quantity --> variable
                    - (rank_end_time[rank]) * release_speed
                )
            )
        if rank > 0:
            model.Add(
                reservoir_vars[rank]
                >= (
                    reservoir_vars[rank - 1]
                    + waste_produced_vars[rank]  # task-dependent quantity --> variable
                    - (rank_end_time[rank] - rank_end_time[rank - 1]) * release_speed
                )
            )

    # Update opt vars
    opt_vars["task_rank_vars"] = task_rank_vars


@log_function
def add_job_completion_constraints(
    model: cp_model.CpModel, model_input: OptInput, variables: Variables
):
    """A job is valid if all of its tasks have been scheduled"""
    # Load variables
    jobs = model_input.sets.jobs
    job_completions = variables.job_completions
    task_presences = variables.task_presences
    jobs_to_tasks = model_input.mappings.jobs_to_tasks

    for job in jobs:
        job_completion = job_completions[job]
        job_tasks_presence = [task_presences[task] for task in jobs_to_tasks[job]]
        model.AddMinEquality(job_completion, job_tasks_presence)


@log_function
def add_flexible_machine_task_selection_constraints(
    model: cp_model.CpModel, model_input: OptInput, variables: Variables
):
    """Flexible machine-task selection"""
    # Load variables
    tasks = model_input.sets.tasks
    tasks_to_machines = model_input.mappings.tasks_to_machines

    # 1. Task-level variables
    for task in tasks:
        candidate_machines = tasks_to_machines[task]
        task_candidate_machines = [
            variables.machine_task_presences[machine, task]
            for machine in candidate_machines
        ]
        # 2. Link candidates machine-tasks to the task
        # Candidates tasks represent the choices to schedule a task across different machines
        # Task is present if the task has been scheduled on 1 machine
        model.AddMaxEquality(variables.task_presences[task], task_candidate_machines)
        for machine in candidate_machines:
            # Link the task interval with the task x machine interval if the latter is chosen (presence = True)
            model.Add(
                variables.task_starts[task]
                == variables.machine_task_starts[machine, task]
            ).OnlyEnforceIf(variables.machine_task_presences[machine, task])
            model.Add(
                variables.task_ends[task] == variables.machine_task_ends[machine, task]
            ).OnlyEnforceIf(variables.machine_task_presences[machine, task])

        # Select exactly one presence variable only if not optional jobs
        if OPTIONAL_JOBS:
            model.AddAtMostOne(task_candidate_machines)
        else:
            model.AddExactlyOne(task_candidate_machines)


@log_function
def add_resources_constraints_with_decomposition(
    model: cp_model.CpModel, model_input: OptInput, opt_vars: Dict[str, Dict]
):
    """
    Add limited resources constraints.
    This constraint make sure that the intervals are allocated up to the resource capacity.

    Inspired by:
    - Paper: 'Why cumulative decomposition is not as bad as it sounds.' [A. Schutt et al]
    - Work from Hakan Kjellerstrand: http://www.hakank.org/google_or_tools/furniture_moving_sat.py

    In this specific use-case, the resource is variable

    Cumulative constraint is as follows:
     for all t in period p:
          sum(demands[i]
            if (start(intervals[i]) <= t < end(intervals[i])) and
            (intervals[i] is present)) <= capacity[p]
    """
    # Load variables
    start = opt_vars["task_start_vars"]
    end = opt_vars["task_end_vars"]
    tasks = model_input.sets.tasks

    # Create shift variables - Only for test-purpose
    # Shift definition --> Only constraint for a subset of relevant shifts
    # (faster computation if only selecting relevant shifts)
    shift_resources = {"shift_1": 3, "shift_2": 2}
    shift_start = {"shift_1": 0, "shift_2": 20}
    shift_end = {"shift_1": 20, "shift_2": 100}
    shifts = list(shift_resources.keys())

    for shift in shifts:
        max_resources = shift_resources[shift]
        for t in range(shift_start[shift], shift_end[shift]):
            bb = []  # All resource requirements for a specific unit of time
            for task in tasks:
                task_resource_required = 1
                # Task starts before time unit t
                # b1: start[task] < t
                b1 = model.NewBoolVar(f"{task}_{shift}_{t}")
                model.Add(start[task] <= t).OnlyEnforceIf(b1)
                model.Add(start[task] > t).OnlyEnforceIf(b1.Not())

                # Task ends after time unit t
                # b2 = t < end[task]
                b2 = model.NewBoolVar(f"{task}_{shift}_{t}")
                model.Add(t < end[task]).OnlyEnforceIf(b2)
                model.Add(t >= end[task]).OnlyEnforceIf(b2.Not())

                # Task overlaps with time unit (start < t < end)
                # b3 = b1 and b2 (b1 * b2)
                b3 = model.NewBoolVar(f"{task}_{shift}_{t}")
                model.AddBoolAnd([b1, b2]).OnlyEnforceIf(b3)
                model.AddBoolOr([b1.Not(), b2.Not()]).OnlyEnforceIf(b3.Not())

                # Task resources requirement: if overlap & resources required > 0
                # b4 = b1 * b2 * r[i]
                b4 = model.NewIntVar(
                    0, max_resources, f"task_resource_requirement_{task}_{t}"
                )
                model.AddMultiplicationEquality(b4, [b3, task_resource_required])
                bb.append(b4)
            model.Add(sum(bb) <= max_resources)


@log_function
def add_resources_constraints_with_cumulative(
    model: cp_model.CpModel, model_input: OptInput, opt_vars: Dict[str, Dict]
):
    """
    Add limited resources constraints.
    This constraint make sure that the intervals are allocated up to the resource capacity.

    In this specific use-case, the resource is constant and we leverage the built-in method AddCumulative.

    Cumulative constraint is as follows:
     for all t:
          sum(demands[i]
            if (start(intervals[i]) <= t < end(intervals[i])) and
            (intervals[i] is present)) <= capacity

    """
    # Load variables & coefficients
    machine_task_interval_vars = opt_vars["machine_task_interval_vars"]
    intervals = [
        machine_task_interval_vars[machine, task]
        for (machine, task) in machine_task_interval_vars
    ]
    resource_demand = [1 for (machine, task) in machine_task_interval_vars]
    resource_available = 3

    # Create custom capacity per shift
    shift_resources = {"shift_1": 3, "shift_2": 2}
    shift_start = {"shift_1": 0, "shift_2": 20}
    shift_end = {"shift_1": 20, "shift_2": 10000}
    shifts = list(shift_resources.keys())
    for shift in shifts:
        frozen_resource = resource_available - shift_resources[shift]
        if frozen_resource > 0:
            frozen_interval = model.NewFixedSizeIntervalVar(
                start=shift_start[shift],
                size=shift_end[shift] - shift_start[shift],
                name=f"frozen_resources_{shift}",
            )
            intervals.append(frozen_interval)
            resource_demand.append(frozen_resource)

    # Create cumulative constraints
    model.AddCumulative(intervals, resource_demand, resource_available)


@log_function
def add_no_machine_overlap_constraints(
    model: cp_model.CpModel, model_input: OptInput, variables: Variables
):
    """Overlap constraint is defined over the interval variables.
    model.AddNoOverlap takes a list of intervals as an input and make sure that 2 variables
    do not overlap one with another.
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
    """Add late delivery constraints"""
    precedence = model_input.sets.precedence

    for task_before, task_after in precedence:
        # Task-precedence presence: task_before not present => task_after not present
        model.AddImplication(
            variables.task_presences[task_before].Not(),
            variables.task_presences[task_after].Not(),
        )
        # Task-precedence time: start[task_after] >= end[task_before]
        model.Add(variables.task_starts[task_after] >= variables.task_ends[task_before])


@log_function
def add_tardiness_definition_constraints(
    model: cp_model.CpModel, model_input: OptInput, variables: Variables
):
    """Add tardiness definition"""
    # Load variables
    processing_times = model_input.coefficients.processing_time
    jobs = model_input.sets.jobs
    due_dates = model_input.coefficients.due_date
    jobs_to_tasks = model_input.mappings.jobs_to_tasks
    tasks_to_machines = model_input.mappings.tasks_to_machines

    for job in jobs:
        # 1. Define job start/end time
        model.AddMaxEquality(
            variables.job_ends[job],
            [variables.task_ends[task] for task in jobs_to_tasks[job]],
        )
        model.AddMaxEquality(
            variables.job_starts[job],
            [variables.task_starts[task] for task in jobs_to_tasks[job]],
        )
        # 2. Define job penalty
        # 2.1 Estimate max job duration (worst case scenario)
        max_job_duration = sum(
            [
                max(
                    [
                        processing_times[task, machine]
                        for machine in tasks_to_machines[task]
                    ]
                )
                for task in jobs_to_tasks[job]
            ]
        )
        # 2.2 Estimate job penalty only if job is not completed
        job_end_ub = (
            variables.job_ends[job].Proto().domain[1]
        )  # upper bound for job_end
        job_penalty = model.NewIntVar(0, job_end_ub * 10, f"job_penalty_{job}")
        model.Add(
            job_penalty
            >= (
                job_end_ub
                + max_job_duration  # estimated worst case scenario
                - (
                    variables.job_ends[job] - variables.job_starts[job]
                )  # actual job duration
            )
        ).OnlyEnforceIf(variables.job_completions[job].Not())
        # 3. Define job tardiness
        model.AddMaxEquality(
            variables.job_tardiness[job],
            [0, variables.job_ends[job] - due_dates[job], job_penalty],
        )


@log_function
def add_sequence_setup_constraints(
    model: cp_model.CpModel, model_input: OptInput, opt_vars: Dict[str, Dict]
):
    """
    Adds machine-setup definition & finds the correct sequence of tasks.

    Adds a circuit constraint from a sparse list of arcs that encode the graph.

    1. An arc is a tuple (task_1, task_2, boolean). The arc is selected in the circuit if the boolean is true.
       tasks must be represented as integers between 0 and the number of tasks - 1.

    2. A circuit is a unique Hamiltonian path in a subgraph of the total graph.
       for example: (0, task_2, True), (task_2, task_3, True), (task_3, 0, True)
       In case a task n is not in the path (i.e not assigned to the machine), then there must be a
       loop arc 'task n -> task n' associated with a true boolean. Otherwise this constraint will fail.

    Hence, we define 4 types of arc:
    1. If it is the first task on the machine: arc from dummy node 0 to task node (0, task, boolean)
    2. If it is the last task on the machine: arc from task node to dummy node 0 (task, 0, boolean)
    3. If task 1 is before task 2: arc(task_1, task_2, True) --> start[task_2] >= end[task_1] + setup[task_1, task_2]
    4. If task 1 is optional (i.e flexible choice of machine):
       arc: (task_1, task_1, True) -->  Task not present on machine (i.e independent from other scheduled tasks)
    """
    task_start_vars = opt_vars["task_start_vars"]
    task_end_vars = opt_vars["task_end_vars"]
    machine_task_presence_vars = opt_vars["machine_task_presence_vars"]
    machines = model_input.sets.machines
    machines_to_tasks = model_input.mappings.machines_to_tasks

    # Step 1: Loop over each machine
    for machine in machines:
        arcs = (
            []
        )  # List of all feasible arcs within a machine. Arcs are boolean to specify circuit from node to node
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
                    machine_task_presence_vars[(machine, task_1)].Not(),
                ]
            )

            for node_2, task_2 in enumerate(machine_tasks):
                if node_1 == node_2:
                    continue
                mt_2 = task_2 + "_" + machine
                # Add sequential boolean constraint: mt_2 follows mt_1
                mt2_after_mt1 = model.NewBoolVar(f"{mt_2} follows {mt_1}")
                arcs.append([node_1 + 1, node_2 + 1, mt2_after_mt1])

                # We add the reified precedence to link the literal with the
                # times of the two tasks.
                min_distance = distance_between_jobs(from_task=mt_1, to_task=mt_2)
                (
                    model.Add(
                        task_start_vars[task_2] >= task_end_vars[task_1] + min_distance
                    ).OnlyEnforceIf(mt2_after_mt1)
                )

        # Constraint to find 1 feasible circuit for each node of the arcs
        # If we consider 3 tasks and only task 2 and task 3 are planned on this machine:
        #  1. arc(0, task_2)=1
        #  2. arc(task_2, task_3) = 1 --> start[task_3] >= end[task_2] + setup
        #  3. arc(task_3, 0) = 1
        # Since task 1 is not planned on this machine: arc(task_1, task_1) = 1
        model.AddCircuit(arcs)


def distance_between_jobs(from_task, to_task):
    """Returns the distance between from_task to to_task"""
    return 2


@log_function
def add_objective(
    model: cp_model.CpModel,
    model_input: OptInput,
    variables: Variables,
):
    """
    Builds all objective parts and adds them to the model.

    Args:
        model: Instance of CpModel
        model_input: contains the optimisation input data that are specific to the UC
        variables: dataclass of all optimisation variables

    """
    logger.info("Create model objective")

    # Tardiness definition
    tardiness = sum([variables.job_tardiness[job] for job in model_input.sets.jobs])
    model.Minimize(tardiness)

    # # Make-span definition
    # task_ends = variables.task_ends
    # make_span = model.NewIntVar(0, max_domain_time, 'make_span')
    # model.AddMaxEquality(make_span, [task_ends[task] for task in task_ends])
    # model.Minimize(make_span)


@log_function
def create_variables(
    model: cp_model.CpModel,
    model_input: OptInput,
    variables: Variables,
    max_domain_time: float = 10000,
) -> Variables:
    """Create all relevant optimization variables.
    Args:
        model: Instance of CpModel
        model_input: contains the optimisation input data that are specific to the UC
        variables: dataclass that contains all variables
        max_domain_time: Maximum domain time. Could be fine-tuned for each task if constraint related to task start/end
    Returns:
        Dict[str, IndexedVariable] : a Dict of IndexedVariable
    """
    # Load sets
    logger.info("Getting the sets for building up variables")
    jobs = model_input.sets.jobs
    tasks = model_input.sets.tasks
    tasks_to_machines = model_input.mappings.tasks_to_machines
    processing_times = model_input.coefficients.processing_time

    # Create core variables
    # 1. Job variables
    variables.job_completions = {
        job: model.NewBoolVar(f"job_completed_{job}") for job in jobs
    }
    variables.job_ends = {
        job: model.NewIntVar(0, max_domain_time, "job_end" + job) for job in jobs
    }
    variables.job_starts = {
        job: model.NewIntVar(0, max_domain_time, "job_start" + job) for job in jobs
    }
    variables.job_tardiness = {
        job: model.NewIntVar(0, max_domain_time * 10, "tardiness" + job) for job in jobs
    }
    # 2. Task variables
    variables.task_starts = {
        task: model.NewIntVar(0, max_domain_time, "start" + task) for task in tasks
    }
    variables.task_ends = {
        task: model.NewIntVar(0, max_domain_time, "end" + task) for task in tasks
    }
    variables.task_presences = {
        task: model.NewBoolVar("presence" + task) for task in tasks
    }
    variables.task_durations = {
        task: model.NewIntVar(
            min(
                processing_times[task, machine] for machine in tasks_to_machines[task]
            ),  # min_duration
            max(
                processing_times[task, machine] for machine in tasks_to_machines[task]
            ),  # max_duration
            "duration" + task,
        )
        for task in tasks
    }
    # 3. task x machine variables
    variables.machine_task_starts = {
        (m, t): model.NewIntVar(0, max_domain_time, "start" + m + t)
        for t in tasks
        for m in tasks_to_machines[t]
    }
    variables.machine_task_ends = {
        (m, t): model.NewIntVar(0, max_domain_time, "end" + m + t)
        for t in tasks
        for m in tasks_to_machines[t]
    }
    variables.machine_task_presences = {
        (m, t): model.NewBoolVar("presence" + m + t)
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
            "interval" + m + t,
        )
        for t in tasks
        for m in tasks_to_machines[t]
    }

    return variables


def get_dict_from_txt(txt: str, separator: str = ":"):
    """Get dict from a txt file"""
    lines = txt.split("\n")
    dict_out = {}
    for line in lines:
        line_split = line.split(separator)
        if len(line_split) > 1 and "CpSolverResponse" not in line:
            key = line_split[0]
            value = line_split[1].lstrip().rstrip()
            dict_out[key] = value
    return dict_out