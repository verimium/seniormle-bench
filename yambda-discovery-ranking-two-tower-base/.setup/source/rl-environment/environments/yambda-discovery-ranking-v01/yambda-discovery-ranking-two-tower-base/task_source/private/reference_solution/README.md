# Private Reference Solution

The two-tower family uses the shared CPU PyTorch implementation in
`../../yambda_two_tower/assets/reference_solver.py` and
`../../yambda_two_tower/assets/two_tower_model.py` as its Oracle. The builder
materializes those files under both `solution/` and `tests/private/` in every
standalone Harbor package.
