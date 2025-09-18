from agents import knowledgeAgent
from vi import Config, Simulation, Window
from environment import Environment
from constants import WIDTH, HEIGHT

config = Config(window=Window(WIDTH, HEIGHT), movement_speed=1, seed=3)
simulation = Environment(config)
simulation.batch_spawn_agents(10, knowledgeAgent, images=["images/red.png"])
simulation.run()
