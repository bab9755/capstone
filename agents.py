from sensors import CameraSensor
from llm import LLM

class Agent:
    def __init__(self):
        self.camera_sensor = CameraSensor()
        self.llm = LLM()

    def update(self):
        self.camera_sensor.take_photo()
        self.llm.generate_response(self.camera_sensor.photo)