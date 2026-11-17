# ScholarEvolve

ScholarEvolve evolves the harness of an LLM agent - the code around a frozen backbone that manages
model inputs, action execution and state - by implementing mechanisms from research papers as
module mutations and recombining them through module crossover.

## Method overview

A harness is composed of five modules with fixed interfaces
(`scholarevolve/harness/modules.py`):

| Module | Responsibility |
| --- | --- |
| `tool_interface` | actions available to the agent; turns model proposals into executable calls |
| `context_management` | assembles the model input from instructions, history, memories and skills |
| `skills` | reusable procedures induced from successful trajectories |
| `memories` | experience recorded from episodes and recalled for similar situations |
| `agentic_workflows` | planning and action generation, task decomposition, agent coordination |

Each generation of the search (`scholarevolve/search/evolve.py`) takes the current champion as the
base harness and runs:

1. **Failure audit** (`research/audit.py`). The champion runs on the evolution set D_evo; every
   failed episode is attributed to the agent or the environment, and failure modes are
   consolidated into one taxonomy.
2. **Capability brief** (`research/brief.py`). Recurring agent-attributable failures that an
   inference-time change could address are abstracted into a general capability gap, without
   environment, domain, tool or field names.
3. **Module-specific queries** (`research/queries.py`). For each module separately, the research
   model receives the brief, the module's responsibility and interface, and the current behaviour
   of that module, and proposes broad capability phrases and narrower mechanism phrases, which are
   interleaved.
4. **Retrieval** (`research/arxiv.py`, `research/pool.py`). Each module's queries retrieve arXiv
   titles and abstracts; results are deduplicated by title within the module.
5. **Topic modeling** (`research/topics.py`). TopicGPT-style taxonomy induction, refinement and
   evidence-supported assignment, conditioned on the module's responsibility.
6. **Screening and allocation** (`research/selection.py`). Topics whose mechanism the current
   harness or backbone already provides are removed; papers are selected round robin across
   clusters ordered by size, following retrieval order within each cluster.
7. **Mutation** (`mutation/`). For each selected paper, the paper's abstract, method sections and
   appendix are read from the full text (`mutation/paper.py`), and the official code, when a
   repository exists, is read by a read-only code-reading agent (`mutation/code.py`). A research
   agent writes a mutation blueprint (`mutation/blueprint.py`), and a coding agent implements it
   as a module (`mutation/coding_agent.py`) with the blueprint, the paper text, the code findings,
   the current module implementation, the interface and the environment's action format in its
   workspace. Health probes (`mutation/probes.py`) check importability, interface compatibility,
   artifact construction and reloading, and valid action production; the coding agent repairs
   reported errors within a check budget.
8. **Artifacts** (`harness/artifacts.py`). Mutations with an offline phase build persistent
   artifacts from successful D_evo episodes; artifacts are frozen with their provenance.
9. **Mutation evaluation**. Each mutation is evaluated against the champion on the same D_val tasks
   and trials, giving per-task gains.
10. **Crossover** (`search/crossover.py`). Configurations of one implementation (or the base
    module) per module are ranked by additive predicted gain; a shortlist is evaluated as
    complete harnesses, and each residual between observed and predicted gain is recorded.
11. **Retention** (`search/stats.py`). The evaluated candidate with the highest mean D_val reward
    replaces the champion only if the lower end of its paired bootstrap gain interval is above
    zero. The search stops after `stop_after` generations without a confirmed improvement, or
    when `max_generations` is reached.

## Installation

```bash
pip install -e .            # add [pdf] to read papers without an HTML rendering
```

Environments are installed separately, each with its own interpreter:

- **AppWorld**: install the `appworld` package and data in a separate environment (see the
  AppWorld repository), together with `openai`. Set `APPWORLD_ROOT` to the directory containing
  `data/` and `APPWORLD_PYTHON` to that environment's Python.
- **τ²-bench**: clone and install τ²-bench in its own environment. Set `TAU2_ROOT` to the
  checkout and `TAU2_PYTHON` to that environment's Python.

The default coding agent is the Codex CLI (`npm install -g @openai/codex`). Setting
`"backend": "chat"` in `coding_agent` uses the research model to write and repair modules instead.
`GITHUB_TOKEN` raises the GitHub API rate limit used by the code-reading agent.

## Configuration

`configs/appworld.json` and `configs/tau2_telecom.json` are complete examples. A configuration
has five sections:

- `environment`: the adapter and its settings (paths, step limits, workers; for τ²-bench also the
  user simulator and the stratified D_evo pool).
- `backbone`: the frozen task model and its request arguments.
- `research_model`: the model that audits, writes queries, models topics, screens and plans.
- `coding_agent`: the coding back end, model and check budget.
- `search`: papers per module, trials, crossover budget, confidence level and stopping rule.

API keys are read from the environment variables named by `api_key_env`.

## Usage

```bash
# Run the search. The run directory is resumable: completed stages are reused.
python -m scholarevolve evolve --config configs/appworld.json --out runs/appworld

# Evaluate the final harness on a test split.
python -m scholarevolve evaluate --config configs/appworld.json \
    --spec runs/appworld/champion/spec.json --split test --trials 3 --out runs/appworld/test
```

A run directory contains, per generation, the champion's D_evo evaluation, the audit, the brief,
the queries, paper pools, topic models and selections, one folder per mutation (full-text context,
code findings, blueprint, coding workspace, candidate module and manifest), the frozen artifacts,
all D_val evaluations, and `decision.json`. The final harness is exported to `champion/` as a
self-contained `spec.json` with its modules, artifacts and blueprints.

## Writing a module by hand

A module is a Python file defining `create_module(*, context, **params)` that returns an object
implementing the module's interface (`scholarevolve.harness.modules.interface_contract(name)`
prints it). Modules exchange the data types in `scholarevolve/harness/interfaces.py`. A harness
specification maps each module name to `null` (the default module) or
`{"module": "path/to/file.py", "params": {}}`.
