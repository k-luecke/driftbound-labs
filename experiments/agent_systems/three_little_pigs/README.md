# Three Little Pigs

Adaptive assurance benchmark comparing controllers under regime shifts, partial observation, costs, and out-of-model failures.

## Run

```bash
python -m driftbound.experiments.three_little_pigs --seed 42 --replications 3
```

## Controllers

| Name | Label |
|------|--------|
| `hmm` | Simplified discrete HMM baseline |
| `hmm_change` | HMM + Page-Hinkley change detection |
| `switching_experts` | Fixed Share switching experts baseline |
| `guarded_seer_hybrid` | HMM + Guarded Seer temporary authority |
| `fixed_policy` | Fixed material policy |

Configs: `configs/`. Results: `generated-results/` (gitignored).

## OOM claim (honest)

Seer target: `brick_under_silent_threat`. Observation encoding for `silent_threat`
aliases calm (code 0), so HMM action selection prefers straw while brick is optimal.
`cunning_wolf` is in-model (obs 2 → brick). Hard-coded brick exploration ≠ validated
Seer authority — both are logged separately in `decision_audit`.

## Genome traits in TLP

When `--evolution-enabled` supplies Agents: `exploration_tendency`, `observation_sensitivity`,
`caution`, and `validation_threshold` causally affect controller behavior.
`learning_rate` is expressed in the phenotype but not yet consumed by TLP controllers.

