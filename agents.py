from sensors import CameraSensor
from llm import LLM
from vi import Agent
from context import Context
class knowledgeAgent(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.camera_sensor = CameraSensor(self) 
        self.llm = LLM(self) 
        self.context = Context(self)
    def update(self):
        self.camera_sensor.take_photo()
        self.llm.generate_response(self.camera_sensor.photo)