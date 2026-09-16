# SmartDream → Drift Bound Labs Legacy Mapping

This document maps concepts inspected **read-only** via `gh api` on
[`k-luecke/smartdream`](https://github.com/k-luecke/smartdream) to the Drift Bound
Labs reimplementation. **No code is imported or copied wholesale** from that
repository. Symbolic names are retained only where they aid continuity; semantics
are redefined operationally here.

Inspection date: 2026-09-16 (via GitHub Contents API on `main`).

## Summary table

| Legacy (smartdream) | Location (legacy) | Drift Bound Labs stance |
|---------------------|-------------------|-------------------------|
| Agent | `agents/agent.py` | Reimplemented: lean dataclass + genome/phenotype/memory/metrics |
| Genome (y/x/v) | `agents/genome.py` | Reimplemented: bounded numeric traits + crossover/mutation/phenotype |
| Species / generation cycle | `agents/species.py` | **Deferred** symbolic species cycles; numeric genomes only |
| Memory | `memory/memory.py` | Reimplemented: bounded deque, no pandas |
| Birth Control | `ecosystem/birth_control.py` | Reimplemented on **canonical Population** (legacy kept its own list) |
| Vitality | `ecosystem/vitality_model.py` | Simplified scalar `vitality`/`health` on Agent |
| Karma | `ecosystem/karma_engine.py` | **Deferred** |
| Vision / VisionChannel | `symbolic/vision*.py` | Reimplemented as evidence records (observer/subject/cost/… ) |
| Seer / SeerEngine | `symbolic/seer.py`, `seer/detector.py` | Reimplemented as **Guarded Seer** with window separation + Bonferroni |
| Schema engine | `identity/schema_engine.py` | **Deferred** |
| Imprinting | `identity/imprinting.py` | **Deferred** |
| Alkaloid harvester/refiner | `harvest/*` | **Deferred** |
| BlackHole / WhiteHole | `substrate/blackhole.py`, `whitehole.py` | **Deferred** |
| Controller / Kalman / Dose | `simulation/controller.py`, `core/*` | New controllers: HMM / change / Fixed Share / Seer hybrid |
| Ecosystem engine | `ecosystem/ecosystem_engine.py` | Folded into Population + Birth Control |
| Sel’kov / cosmological metaphors | README / substrate myths | **Deferred** |
| Three Little Pigs | Not found as a first-class module in tree listing | **New** benchmark designed for this monorepo |

## Concept notes

### Agent

Legacy `Agent` wired many engines (Seer, Kalman, Ritual, Schema, Vitality, Karma,
Ecosystem, VisionChannel, Dose, ImmuneResponse, PathTraining) in `__init__`.
Drift Bound Labs keeps a **minimal** agent: id, genome, phenotype, vitality/health,
memory, correctness/consequence/utility tallies, generation, parent ids, alive flag,
controller/expert params. Additional mythic engines are deferred.

### Memory

Legacy used a list of dict events plus pandas DataFrame export. DBL uses a capacity-
limited `deque` of `MemoryRecord`s — enough for lifecycle experiments without heavy deps.

### Genome / species / mutation

Legacy genome had symbolic `y` (species), continuous `x` (e.g. entry_threshold), and
`v` (sensitivity); species templates (Seraph/Mycelid/Chorite) and Wuxing-style
generation cycles gated mating. DBL genomes are **fully numeric** with explicit bounds
and tested mutation-in-bounds behavior. Symbolic species mating rules are deferred.

### Birth Control

Legacy `BirthControl` maintained its **own** `population` list and culled by
`compute_fitness()`. DBL non-negotiable: **one canonical `Population`**; Birth Control
only coordinates eligibility, selection, cull, birth, replacement, lineage, and
extinction protection against that registry.

### Vision

Legacy VisionChannel mixed shared fields, fractal echoes, and broadcast symbolism.
DBL Vision is an **observational evidence channel**: observer, subject, context,
observed action, outcome, `independently_observable`, cost, step. Observation does
**not** imply correctness.

### Seer

Legacy `SeerEngine` promoted via smoothed insight density / symbolic pressure without
hard train/validate separation. DBL **Guarded Seer** enforces discovery / validation /
deployment windows, minimum evidence, **Bonferroni** correction, promotion on
validation only, temporary authority with expiry/demotion, monitoring, and no window
leakage. Authority is advisory — it does not replace the base controller globally.

### Reward / fitness / vitality

Legacy fitness was ad hoc (`compute_fitness` referenced by Birth Control; vitality
decayed separately; karma tracked “symbolic integrity”). DBL uses explicit reward
decomposition on events and a simple fitness = recent_correctness × vitality +
utility term for cull/selection. Karma is deferred.

### Schema / imprinting / alkaloids / BH–WH / Sel’kov

Present in legacy as identity schemas, early-life imprinting symbols, “alkaloid”
insight distillation, and substrate black/white hole metaphors. All **deferred** in
DBL: documented here for continuity, not activated in code.

### Controller

Legacy trading-oriented perceive/act with confidence thresholds and Kalman fever
flags. DBL controllers are experiment-facing: discrete HMM, HMM+Page-Hinkley, Fixed
Share experts, fixed policy, and Guarded Seer hybrid.

### Three Little Pigs

No dedicated Three Little Pigs module appeared in the smartdream tree listing at
inspection time (notebooks/checklists may reference related stories elsewhere).
DBL introduces a first-class benchmark with Straw/Wood/Brick materials, Wolf /
CunningWolf / calm / storm (OOM) regimes, held-out schedules, costs, equal budgets,
and the controller suite above.

## Design principle

Treat SmartDream as a **concept mine**, not a dependency. Every retained name must
earn an operational definition under experimental standards
(`docs/experimental_standards.md`).
