# Drift Bound Labs

> *Build systems that stay honest under drift: measure what changes, pay for what you observe, and never promote a story that has not survived an independent check.*

Reproducible experiments in **adaptive assurance**, **agent populations**, **nonlinear control**, and **physical sensing** under uncertain and changing conditions.

## Scope

| Area | Status |
|------|--------|
| Core experiment primitives (RNG, events, metrics, provenance) | **Implemented** |
| Canonical Population + Birth Control + Genome lifecycle | **Implemented** |
| Vision evidence channel | **Implemented** |
| Guarded Seer (discovery / validation / deployment) | **Implemented** |
| Baselines: discrete HMM, Page-Hinkley, Fixed Share | **Implemented** (simplified where noted) |
| Three Little Pigs benchmark CLI | **Implemented** |
| RF systems | **Docs/manifests only** — no fabricated results |
| Sel’kov, alkaloids, BH/WH, karma, schema, imprinting, RF→agent | **Deferred** (documented, not activated) |

**Warning on symbolic names:** *Seer*, *Vision*, *Birth Control*, *Genome*, *alkaloid*, *Black Hole / White Hole*, etc. are research metaphors. Their operational meaning is defined in code and docs — they are not mystical claims.

Legacy concept mapping (read-only inspection of prior work): [`docs/smartdream_legacy_mapping.md`](docs/smartdream_legacy_mapping.md).

## Quick start

```bash
python -m pip install -e ".[dev]"
pytest
ruff check .
mypy src
python -m driftbound.experiments.three_little_pigs --seed 42
```

## Repository map

```text
src/driftbound/          # Installable package (src layout)
  core/                  # RNG, events, metrics, provenance, experiment helpers
  smartdream/            # Agents, Population, Vision, Seer, controllers, lifecycle
  baselines/             # HMM, change detection, switching experts
  experiments/           # CLI entry points
experiments/             # Configs + area READMEs (agent_systems, rf_systems)
docs/                    # Research program, standards, terminology, legacy map
tests/                   # unit / integration / statistical
notebooks/               # Exploratory only; clear outputs before commit
generated-results/       # Gitignored CLI outputs
```

## Reproduction

1. Install with `pip install -e ".[dev]"` (Python ≥ 3.12).
2. Run `pytest` (includes fixed-seed determinism checks).
3. Run the Three Little Pigs CLI with `--seed 42`; JSON lands in `generated-results/`.
4. Same seed ⇒ same metrics (deterministic NumPy `Generator`s passed explicitly).

## License

MIT — see [`LICENSE`](LICENSE). Cite via [`CITATION.cff`](CITATION.cff).
