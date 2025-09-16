## Multi‑Agent Scene Understanding in Violet

Build and evaluate a multi‑agent system that explores an environment, collects observations, and explains it in natural language using LLMs. The aim is to measure how quickly and how accurately agents can form a shared understanding of their world.

This repo extends the Violet simulator with:
- **Custom environment generation** (obstacles and sites) with seeded randomness for reproducible experiments
- **Sensors and actuators** for agent kinematics and perception
- **LLM reasoning** to turn raw observations into concise, natural‑language summaries
- **Evaluation hooks** to track time‑to‑understanding and description quality

### Why this is interesting
- **Embodied intelligence**: bridge low‑level perception and movement with high‑level language.
- **Reproducible science**: deterministic seeds + modular scenarios.
- **Benchmarking**: quantify speed/accuracy trade‑offs across prompts, models, and environments.

---

## Quickstart

Prereqs:
- Python 3.13+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) for fast env management

Run a default simulation:
```sh
uv run main.py
```

What you’ll see:
- A Violet window with agents, one or more obstacles, and multiple sites.
- Agents move, sense, and produce natural‑language status (printed to console).

---

## Project structure

```text
test-violet/
  main.py                 # Entry point: builds a sim, spawns env + agents, runs
  agents.py               # Agent definitions (LLM-enabled agent, etc.)
  sensors.py              # Sensors + Actuator with delta_t-aware kinematics
  environment.py          # (Pluggable) environment builder/generation utilities
  configs/                # Scenario/config files (e.g., YAML)
  images/                 # Sprites for agents, obstacles, sites
  prompts.py              # Prompt templates for LLM reasoning
  llm.py                  # LLM wrapper (model selection, calls, post-processing)
  context.py              # Shared runtime context for agents/LLM
```

---

## How it works

- **Simulation (Violet)**: We use Violet’s `Simulation`, `Agent`, and `Config` primitives for time‑stepped updates and rendering.
- **Environment**: Obstacles and sites are created via a modular builder (see `environment.py`) which supports seeded randomization, placement strategies, and metadata per object.
- **Sensors/Actuators**: In `sensors.py`, agents read world state and move using delta‑time aware kinematics:
  - Position update: `pos += move * (linear_velocity * delta_time)`
  - Orientation update: `move.rotate_ip(angular_velocity * delta_time)`
- **LLM Layer**: Each agent summarizes observations into natural language (see `llm.py` and `prompts.py`).
- **Evaluation**: Measure latency to a correct description and maintain quality scores (e.g., keyword/structure checks, model‑graded comparisons).

---

## Configure and randomize environments

Define scenarios in code or YAML (e.g., `configs/configs.yaml`) and build them before `.run()`.

Example (conceptual YAML):
```yaml
seed: 1234
world:
  width: 1000
  height: 700
obstacles:
  - type: box
    count: 1
    placement: center
sites:
  - type: marker
    count: 4
    placement: random
```

Tie it into the sim (simplified):
```python
from vi import Simulation, Config, Window
from environment import build_environment  # your builder

config = Config(window=Window(1000, 700))
x, y = config.window.as_tuple()

env = build_environment(seed=1234, width=x, height=y)  # returns obstacles & sites

sim = Simulation(config)
for ob in env.obstacles:
    sim = sim.spawn_obstacle(ob.image, ob.x, ob.y)
for s in env.sites:
    sim = sim.spawn_site(s.image, x=s.x, y=s.y)

(
    sim
    .batch_spawn_agents(10, myAgent, images=["images/green.png", "images/red.png"])  # see main.py
    .run()
)
```

---

## Experiments and evaluation

- **Speed**: Time (ticks) until agents produce a correct/complete environment description.
- **Accuracy**: Compare generated text to ground‑truth descriptors (per‑object, global layout). Optionally use an LLM judge.
- **Determinism**: Use seeded PRNGs for movement and environment generation for apples‑to‑apples comparisons.
- **Batching**: See `batch_run_experiments.py` for automating multiple seeds/scenarios (extend as needed).

---

## Extending the system

- **New obstacle/site types**: Add a factory to the environment registry and extend placement strategies.
- **New sensors**: Implement a sensor class that reads world state (e.g., proximity, vision) and attach to an agent.
- **New actuators/kinematics**: Use `delta_time` integration for smooth motion; clamp speeds/turn rates per agent.
- **New prompts/LLMs**: Swap prompt templates (`prompts.py`) and model backends (`llm.py`).
- **New metrics**: Add precision/recall on object mentions, structure checks, and latency breakdowns.

---

## Development

Install and run:
```sh
uv run main.py
```

Format/lint: project follows standard Python formatting; keep imports and types clean.

---

## Acknowledgements

- Built on top of [Violet](https://github.com/m-rots/violet), a lightweight 2D agent simulation framework.
- Sprite assets included in `images/` for quick prototyping.

---

## License

MIT (see LICENSE if present). Please respect third‑party model and API licenses.
