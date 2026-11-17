"""Modular agent harness: interfaces, default modules, composition and frozen artifacts."""

from scholarevolve.harness.interfaces import (
    Budget, EpisodeResult, Evidence, Proposal, Skill, StepState, ToolSpec,
)
from scholarevolve.harness.modules import (
    MODULE_NAMES, MODULE_PREFIX, MODULE_RESPONSIBILITIES, ModuleContractError, interface_contract,
    supports_offline_fit, validate_manifest, validate_module,
)
from scholarevolve.harness.runtime import (
    EVOLUTION_SPLIT, Harness, ModuleRuntime, build_harness, load_module,
)

__all__ = [
    "Budget", "EpisodeResult", "Evidence", "Proposal", "Skill", "StepState", "ToolSpec",
    "MODULE_NAMES", "MODULE_PREFIX", "MODULE_RESPONSIBILITIES", "ModuleContractError",
    "interface_contract", "supports_offline_fit", "validate_manifest", "validate_module",
    "EVOLUTION_SPLIT", "Harness", "ModuleRuntime", "build_harness", "load_module",
]
