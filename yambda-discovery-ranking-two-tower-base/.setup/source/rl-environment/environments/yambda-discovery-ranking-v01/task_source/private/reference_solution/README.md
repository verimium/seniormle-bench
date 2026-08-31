# Private Reference Solution

`reference_solver.py` is the task-owned implementation used to materialize the
generated Harbor Oracle solution. It is kept outside `harbor/` because agents
must not receive the reference implementation as part of the task environment.
