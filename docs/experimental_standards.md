# Experimental Standards

## Reproducibility

- Pass `numpy.random.Generator` instances explicitly; never mutate global RNG state.
- Record seed, config, package version, and timestamp in provenance.
- Write numeric outputs under `generated-results/` (gitignored).
- Prefer scripts over notebooks for claimed results; clear notebook outputs.

## Evaluation hygiene

- **No leakage:** discovery evidence must not enter validation; held-out schedules stay held out.
- **Equal budgets:** controllers compared under the same step count, observation probability, and cost parameters unless the ablation is the point.
- **Honest labels:** mark code as baseline / simplified / future work.
- **Costs:** observation, intervention, and fallback costs appear in the reward decomposition:
  `net = correctness + consequence_utility - observation_cost - intervention_cost - fallback_cost`.

## Statistics (CI-small)

- Fixed seeds in CI.
- Report mean ± sample std across replications for primary metrics.
- Multiple testing: Guarded Seer uses **Bonferroni** (conservative); BH/FDR is future work.
- Do not claim significance without pre-registered contrasts and adequate power.

## RF / physical sensing

- No fabricated spectra or detection rates.
- Include spectrum/legal disclaimer; prefer simulation until compliant hardware paths exist.

## Security & secrets

- Never commit tokens, credentials, private keys, or local absolute paths in shared docs.
