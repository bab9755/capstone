from agents import knowledgeAgent
from vi import Config, Simulation, Window
from subjects import SubjectAgent
from environment import Environment
from constants import WIDTH, HEIGHT
from story_registry import create_story_environment
import pygame as pg
from constants import ground_truth
from visualize import LivePlot
import random
import math
from helpers import load_config

def run_simulation():

    config = load_config()

    story_sites = create_story_environment(WIDTH, HEIGHT, seed=random.randint(0, 10))
    context_length = config.get("context", {}).get("p", 2)
    num_knowledge_agents = config.get("agents", {}).get("knowledge_agents", 3)
    num_subject_agents = config.get("agents", {}).get("subject_agents", 5)
    social_learning_enabled = config.get("social_learning_enabled", False)
    
    config = Config(window=Window(WIDTH, HEIGHT), seed=random.randint(0, 10))
    simulation = Environment(llm_provider="Ollama", llm_model="gemma3", num_knowledge_agents=num_knowledge_agents, num_subject_agents=num_subject_agents, config=config)
    
    def create_knowledge_agents(*args, **kwargs):
        return knowledgeAgent(context_size=context_length, social_learning_enabled=social_learning_enabled, *args, **kwargs)
    simulation.batch_spawn_agents(num_knowledge_agents, create_knowledge_agents, images=["images/robot.png"])

    num_fragments = len(ground_truth)
    grid_cols = math.ceil(math.sqrt(num_fragments))
    grid_rows = math.ceil(num_fragments / grid_cols)
    x_spacing = WIDTH / (grid_cols + 1)
    y_spacing = HEIGHT / (grid_rows + 1)
    subject_positions = []

    for row in range(grid_rows):
        for col in range(grid_cols):
            if len(subject_positions) >= num_fragments:
                break
            x = (col + 1) * x_spacing
            y = (row + 1) * y_spacing
            subject_positions.append((x, y))

    for fragment, position in zip(ground_truth, subject_positions):
        simulation.batch_spawn_agents(1, SubjectAgent, images=["images/villager.png"])
        subjects = [a for a in simulation._agents if getattr(a, "role", None) == "SUBJECT"]
        if subjects:
            subject = subjects[-1]
            subject.info = fragment
            subject.pos.update(position)

    
    return simulation

# Create and run the story environment
if __name__ == "__main__":
    plot = LivePlot()
    simulation = None
    try: 

        simulation = run_simulation()
        simulation.run(plot)
    except KeyboardInterrupt:
        print("\n⏹️  Simulation interrupted by user")
    finally:
        if simulation:
            simulation.stop()
            # Save experiment data and plot on exit
            simulation.save_experiment_data(plot)
        
        
