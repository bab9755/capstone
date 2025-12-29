from __future__ import annotations

from agents import knowledgeAgent
from vi import Config, Window
from subjects import SubjectAgent
from environment import Environment
from story_registry import create_story_environment
from visualize import LivePlot
import random
import math
from runtime_config import get_runtime_settings

def run_simulation():
    settings = get_runtime_settings()
    env_width = settings["environment"]["width"]
    env_height = settings["environment"]["height"]
    context_length = settings["context"].get("p", 2)
    social_learning_enabled = settings["social_learning_enabled"]
    num_knowledge_agents = settings["agents"]["knowledge"]
    num_subject_agents = settings["agents"]["subjects"]
    live_plot_settings = (settings.get("visualization") or {}).get("live_plot") or {}
    live_plot_enabled = live_plot_settings.get("enabled", True)
    live_plot_interval_ms = live_plot_settings.get("update_interval_ms", 3000)
    ground_truth_snippets = list(settings["ground_truth"].get("snippets", []))

    if not ground_truth_snippets:
        raise ValueError("Active ground truth set must contain at least one snippet.")

    # Align subject agent count with available snippets
    num_fragments = len(ground_truth_snippets)
    if num_subject_agents <= 0 or num_subject_agents > num_fragments:
        num_subject_agents = num_fragments

    ground_truth_snippets = ground_truth_snippets[:num_subject_agents]

    create_story_environment(env_width, env_height, seed=random.randint(0, 10))

    simulation_config = Config(window=Window(env_width, env_height), seed=random.randint(0, 10))
    simulation = Environment(
        llm_provider="Ollama",
        llm_model="gemma3",
        num_knowledge_agents=num_knowledge_agents,
        num_subject_agents=num_subject_agents,
        config=simulation_config,
    )

    num_snapshots = settings.get("num_snapshots", 30)
    snapshot_interval = settings.get("snapshot_interval_seconds", 10.0)
    
    print(f"Running the simulation with the following settings")
    print(f"Environment: {env_width}x{env_height}")
    print(f"Agents: {num_knowledge_agents} knowledge, {num_subject_agents} subjects")
    print(f"Context length: {context_length}")
    print(f"Social learning enabled: {social_learning_enabled}")
    print(f"Snapshots: {num_snapshots} × {snapshot_interval}s = {num_snapshots * snapshot_interval}s total")
    print(f"Ground truth snippets: {len(ground_truth_snippets)}")

    def create_knowledge_agents(*args, **kwargs):
        return knowledgeAgent(context_size=context_length, social_learning_enabled=social_learning_enabled, *args, **kwargs)

    simulation.batch_spawn_agents(num_knowledge_agents, create_knowledge_agents, images=["images/robot.png"])

    grid_cols = math.ceil(math.sqrt(num_subject_agents))
    grid_rows = math.ceil(num_subject_agents / grid_cols)
    x_spacing = env_width / (grid_cols + 1)
    y_spacing = env_height / (grid_rows + 1)
    subject_positions = []

    for row in range(grid_rows):
        for col in range(grid_cols):
            if len(subject_positions) >= num_subject_agents:
                break
            x = (col + 1) * x_spacing
            y = (row + 1) * y_spacing
            subject_positions.append((x, y))

    for fragment, position in zip(ground_truth_snippets, subject_positions):
        simulation.batch_spawn_agents(1, SubjectAgent, images=["images/villager.png"])
        subjects = [a for a in simulation._agents if getattr(a, "role", None) == "SUBJECT"]
        if subjects:
            subject = subjects[-1]
            subject.info = fragment
            subject.pos.update(position)

    
    return simulation

# Create and run the story environment
if __name__ == "__main__":
    simulation = None
    plot: LivePlot | None = None
    try: 
        runtime_settings = get_runtime_settings()
        live_plot_cfg = (runtime_settings.get("visualization") or {}).get("live_plot") or {}
        if live_plot_cfg.get("enabled", True):
            plot = LivePlot()
        else:
            print("ℹ️ Live plotting is disabled via configs.yaml; running headless.")
        simulation = run_simulation()
        simulation.run(plot)
    except KeyboardInterrupt:
        print("\n⏹️  Simulation interrupted by user")
    finally:
        if simulation:
            simulation.stop()
            # Persist results if the simulation ended unexpectedly
            if not getattr(simulation, "_experiment_saved", False):
                simulation.save_experiment_data(plot)
        
        
