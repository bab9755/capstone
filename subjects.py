from vi import Agent
from sensors import Actuator, Sensor
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
        self.visible = True  # Whether knowledge agents can interact with this subject

    def update(self):
        # Static subject: no behavior
        pass
    
    def set_visible(self, visible: bool):
        """Set visibility state and update sprite alpha accordingly."""
        self.visible = visible
        # Visual feedback: dim the sprite when invisible
        if hasattr(self, 'image') and self.image is not None:
            self.image.set_alpha(255 if visible else 40)

    def get_velocities(self):
        return 0, 0


