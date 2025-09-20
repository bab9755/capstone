from llm import LLM
from vi import Agent
from context import Context
from sensors import Sensor, Actuator
import random
from constants import WIDTH, HEIGHT
import pygame as pg
class knowledgeAgent(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
        self.sensor = Sensor(self) # for global coordinate access
        self.actuator = Actuator(self) # for local coordinate access
        self.llm = LLM(self) 
        self.context = Context(self)

        self.seen_agents = set()
        self.seen_agents_time = {} #map seen agents to the last time they were seen

        self.pos.x = random.uniform(0, WIDTH)
        self.pos.y = random.uniform(0, HEIGHT)


    def update(self): # at every tick (timestep), this function will be run
        # State machine for proximity interactions

        self.sensor.check_neighbors()
            

    def exchange_context_with_agents(self, agents):
        """Exchange context with newly encountered agents"""
        for agent in agents:
            # Add context about this interaction
            self.context.add_context()
            print(f"Agent {self.id} exchanged context with Agent {agent.id}")
    
    def reset_proximity_state(self):
        """Reset the proximity state (useful for testing or new scenarios)"""
        self.seen_agents.clear()
        self.current_proximity_agents.clear()
        print(f"Agent {self.id} reset proximity state")

    def get_velocities(self): 
        # at the start of the simulation this is what we get
        linear_speed = 2
        angular_velocity = 0.0

        if self.sensor.border_collision(): # if the agent is on the edge of the world, it randomly changes its angular velocity
            angular_velocity = 10
            linear_speed = 2

        return linear_speed, angular_velocity



    


    