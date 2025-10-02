from vi import Agent
from sensors import Actuator
class Subject:
    def __init__(sel, information: str, position: tuple):
        self.information = information
        self.position = position

    def get_information(self):
        return self.information

    def get_position(self):
        return self.position
    
    def get_subject(self):
        return self.subject

class SubjectAgent(Agent):
    def __init__(self, *args, info: str = "", **kwargs):
        super().__init__(*args, **kwargs)
        self.role = "SUBJECT"
        self.info = info
        self.actuator = Actuator(self)

    def update(self):
        # Static subject: no behavior
        pass

    def get_velocities(self):
        return 0, 0


