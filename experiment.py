from agents import knowledgeAgent, Villager
from vi import Config, Simulation, Window
from subjects import SubjectAgent
from environment import Environment
from constants import WIDTH, HEIGHT
from story_registry import create_story_environment, story_registry
import pygame as pg
from constants import fragments
from visualize import LivePlot
from metrics import compute_bert_score
import random
import yaml
from pathlib import Path



def load_config():
    config_file = Path("configs/configs.yaml")
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    return config

def run_simulation():

    config = load_config()

    story_sites = create_story_environment(WIDTH, HEIGHT, seed=random.randint(0, 1000000))
    context_length = config.get("context", {}).get("p", 2)
    num_knowledge_agents = config.get("agents", {}).get("knowledge_agents", 3)
    num_subject_agents = config.get("agents", {}).get("subject_agents", 5)
    
    config = Config(window=Window(WIDTH, HEIGHT), seed=random.randint(0, 1000000))
    simulation = Environment(llm_provider="Ollama", llm_model="gemma3", num_knowledge_agents=num_knowledge_agents, num_subject_agents=num_subject_agents, fragments=fragments, config=config)
    
    def create_knowledge_agents(*args, **kwargs):
        return knowledgeAgent(context_size=context_length, *args, **kwargs)
    simulation.batch_spawn_agents(num_knowledge_agents, create_knowledge_agents, images=["images/robot.png"])

    for fragment in fragments:
        simulation.batch_spawn_agents(1, SubjectAgent, images=["images/villager.png"])
        subjects = [a for a in simulation._agents if getattr(a, "role", None) == "SUBJECT"]
        if subjects:
            subjects[-1].info = fragment

    
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
        
        
