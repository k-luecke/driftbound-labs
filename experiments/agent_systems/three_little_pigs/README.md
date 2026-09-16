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
