from agents import knowledgeAgent
from vi import Config, Simulation, Window
from environment import Environment
from constants import WIDTH, HEIGHT

config = Config(window=Window(WIDTH, HEIGHT),seed=3)
simulation = Environment(config)
simulation.batch_spawn_agents(2, knowledgeAgent, images=["images/red.png"])
simulation.run()
