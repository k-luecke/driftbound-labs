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
2. Run `pytest` (includes fixed-seed determinism and adversarial fairness checks).
3. Run the Three Little Pigs CLI with `--seed 42`; JSON lands in `generated-results/`.
4. Same seed ⇒ same metrics (deterministic NumPy `Generator`s passed explicitly).
5. Optional lifecycle: `--evolution-enabled` wires canonical Population / Birth Control
   between episodes; omit the flag for fair non-evolving controller comparison.

### TLP fairness invariants (hardening)

- Environment schedules and observation/noise draws are **pre-generated** on a world
  RNG independent of controller RNGs (identical worlds across controllers).
- HMM action selection **filters the current observation** once per step (not lag-only).
- **Decision correctness** is `action == best_action(regime)`, separate from physical
  survival/consequence in the reward decomposition.
- Guarded Seer targets `brick_under_cunning_wolf` (brick optimal under cunning wolf),
  not ordinary wolf where wood is optimal.
- Exploration, authority overrides, promotions/demotions, and their costs are logged
  as distinct audit fields — not collapsed into one opaque score.

## CI

GitHub Actions workflow: [`.github/workflows/ci.yml`](.github/workflows/ci.yml)
(install, ruff, mypy, pytest on push/PR). A duplicate reference copy also lives at
[`docs/ci/github-actions.yml`](docs/ci/github-actions.yml).

## License

MIT — see [`LICENSE`](LICENSE). Cite via [`CITATION.cff`](CITATION.cff).
