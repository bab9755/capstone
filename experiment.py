from agents import knowledgeAgent
from vi import Config, Simulation, Window
from environment import Environment
from constants import WIDTH, HEIGHT
import random

config = Config(window=Window(WIDTH, HEIGHT),seed=3)
simulation = Environment(config)
simulation.batch_spawn_agents(2, knowledgeAgent, images=["images/red.png"])
for i in range(10):
    simulation.spawn_site("images/fighter3.png", x=random.randint(0, WIDTH), y=random.randint(0, HEIGHT))
simulation.run()
