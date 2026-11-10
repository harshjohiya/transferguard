# Contributing

1. Clone the [repository](https://github.com/harshjohiya/transferguard.git), create a Python 3.11 environment, and follow the [README setup](README.md#local-setup).
2. Create a feature branch from `main`, for example `git switch -c experiment/new-condition`.
3. Keep experiments reproducible: state the dataset, split, seed, source corruption, architecture, training schedule, and evaluation rule. Preserve existing experimental results.
4. Commit notebook code with outputs and execution counts cleared. Do not commit datasets, checkpoints, caches, local paths, credentials, or large generated files.
5. Record new measurements with provenance and distinguish observations from hypotheses. Do not describe validation results as held-out test results.
6. Open a pull request against `main` with a short summary, changed files, validation steps, and any limitations for review.
