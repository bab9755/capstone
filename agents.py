from llm import LLM
from vi import Agent
from context import Context
from sensors import Sensor, Actuator
import random
from constants import WIDTH, HEIGHT
class knowledgeAgent(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
        self.sensor = Sensor(self) # for global coordinate access
        self.actuator = Actuator(self) # for local coordinate access
        self.llm = LLM(self) 
        self.context = Context(self)

        self.seen_agents = set() 

        self.pos.x = random.uniform(0, WIDTH)
        self.pos.y = random.uniform(0, HEIGHT)


    def update(self): # at every tick (timestep), this function will be run
        # State machine for proximity interactions
        nearby_agents = list(self.in_proximity_performance())
        current_agent_ids = {agent.id for agent in nearby_agents}
        
        new_agents = current_agent_ids - self.seen_agents
        
        if new_agents:
            new_agent_objects = [agent for agent in nearby_agents if agent.id in new_agents]
            print(f"Agent {self.id} is in proximity with NEW agents: {[agent.id for agent in new_agent_objects]}")
            self.seen_agents.update(new_agents)
            

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



    


    