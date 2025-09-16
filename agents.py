from llm import LLM
from vi import Agent
from context import Context
from sensors import Sensor, Actuator
class knowledgeAgent(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
        self.sensor = Sensor(self) # for global coordinate access
        self.actuator = Actuator(self) # for local coordinate access
        self.llm = LLM(self) 
        self.context = Context(self)

    def update(self): # at every tick (timestep), this function will be run
        pass