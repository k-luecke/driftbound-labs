"""Genetic lifecycle: genome, selection, mutation, birth control.

Submodules are imported lazily via attribute access to avoid circular imports
(Agent → Genome → BirthControl → Agent).
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "BirthControl",
    "Genome",
    "Phenotype",
    "mutate_genome",
    "select_parents",
]


def __getattr__(name: str) -> Any:
    if name in ("Genome", "Phenotype"):
        from driftbound.smartdream.lifecycle.genome import Genome, Phenotype

        return {"Genome": Genome, "Phenotype": Phenotype}[name]
    if name == "mutate_genome":
        from driftbound.smartdream.lifecycle.mutation import mutate_genome

        return mutate_genome
    if name == "select_parents":
        from driftbound.smartdream.lifecycle.selection import select_parents

        return select_parents
    if name == "BirthControl":
        from driftbound.smartdream.lifecycle.birth_control import BirthControl

        return BirthControl
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
