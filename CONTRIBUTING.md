# Contributing to NHSMM

Thank you for your interest in contributing to NHSMM (Neural Hidden Semi-Markov Models).
We welcome contributions including bug fixes, new examples, documentation improvements,
performance optimizations, and research-driven extensions.

## Getting Started

- Fork the repository
- Create a feature branch from `main`
- Keep changes focused and well-documented

If you are new to the project, look for issues labeled `good first issue`.

## Pull Requests

- Use clear, descriptive commit messages
- Ensure examples and tests pass locally
- Update documentation when behavior or usage changes
- Be responsive to review feedback

## Adding New Examples

1. Open an issue proposing the example and explain how it differs from existing ones.
2. Implement the example following the project structure.
3. Add a `README.md` describing purpose, usage, and expected output.
4. Register the example in `run_python_examples.sh` if applicable.
5. Add a short entry to `docs/source/index.rst`.
6. Build and verify documentation locally.
7. Ensure all examples and tests pass.

## Bug Fixes and Improvements

- Add or update tests when fixing bugs
- Keep changes minimal and focused
- Run the full example suite before submitting

GPU access is recommended for model-related changes. A cloud instance such as AWS `g4dn.4xlarge`
is a reasonable baseline.

## License

By contributing, you agree that your contributions will be licensed under the LICENSE file.
