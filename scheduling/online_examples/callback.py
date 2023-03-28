from ortools.sat.python import cp_model


class VarArraySolutionPrinter(cp_model.CpSolverSolutionCallback):
    """Print intermediate solutions."""

    def __init__(self, variables):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__variables = variables
        print(variables)
        self.__solution_count = 0

    def on_solution_callback(self):
        self.__solution_count += 1
        # print('----------------------------------')
        # print(self.__variables)
        # for x in self.__variables:
        #     print(x)
        #     print(self)
        #     print(self.Value(x))

        for v in self.__variables:
            # print(v)
            # print(self.Value(vars[v]))
            print('%s=%i' % (v, self.Value(v)), end=' ')

            # print(f"{v}, {self.Value(v)}", end=' ')


        print()

    def solution_count(self):
        return self.__solution_count


def SearchForAllSolutionsSampleSat():
    """Showcases calling the solver to search for all solutions."""
    # Creates the model.
    model = cp_model.CpModel()

    # Creates the variables.
    num_vals = 3
    tasks = {'a', 'b', 'c'}
    vars = {
        task: model.NewIntVar(0, num_vals - 1, f"task_{task}") for task in tasks
    }
    # x = model.NewIntVar(0, num_vals - 1, 'x')
    # y = model.NewIntVar(0, num_vals - 1, 'y')
    # z = model.NewIntVar(0, num_vals - 1, 'z')

    # Create the constraints.
    model.Add(vars['a'] != vars['b'])

    # Create a solver and solve.
    solver = cp_model.CpSolver()
    # solution_printer = VarArraySolutionPrinter([x, y, z])
    solution_printer = VarArraySolutionPrinter(vars.values())

    # Enumerate all solutions.
    solver.parameters.enumerate_all_solutions = True
    # Solve.
    status = solver.Solve(model, solution_printer)

    #status = solver.Solve(model)


    # print('Status = %s' % solver.StatusName(status))
    # print('Number of solutions found: %i' % solution_printer.solution_count())


SearchForAllSolutionsSampleSat()
