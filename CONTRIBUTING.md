# Contributing to Drift Bound Labs

Thanks for helping build a reproducible research platform.

## Development setup

```bash
python -m pip install -e ".[dev]"
pytest
ruff check .
mypy src
```

## Guidelines

- Pass NumPy RNGs explicitly; do not rely on global RNG state.
- Keep `generated-results/` out of git (already gitignored).
- Prefer focused commits with descriptive messages.
- Do not commit secrets, tokens, large datasets, or notebook outputs.
- Label baselines honestly (baseline / simplified / future work).
- Symbolic names (Seer, Vision, Birth Control, etc.) are research metaphors — document operational meaning in code and docs.

## Pull requests

Open PRs against `main`. CI must pass (ruff, mypy, pytest). Do not force-push shared branches without coordination.
