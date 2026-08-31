# Yambda two-tower task family

This directory contains authoring sources shared by the Yambda two-tower task
family. It is owned by the base task's `task_source/` directory. The case-1
profile is a child of the base task and contains only its task contract,
instructions, verifier behavior, and other variant-owned files.

`builder.py` materializes every profile as a complete Harbor task. Generated
packages contain copied source files, requirements, image contexts, public
data, and private verifier assets; they do not import from this directory or
depend on another generated task.

`assets/reference_solver.py` is the base task's CPU PyTorch reference, while
`assets/two_tower_model.py` contains model primitives shared by the family. A
case that changes the training policy owns its solver under
`private/reference_solution/`; the builder selects that variant-specific source
without changing the parent task's reference.

Each profile keeps its task instruction at the profile root. Its `harbor/`
directory groups the task contract and runtime/test templates, while `private/`
contains author-owned reference and verifier inputs. The case-1 profile also
declares its production model in `model_overlay.json`; exact source replacements
and precomputed reference/production hashes keep that delta reviewable and
bounded.
