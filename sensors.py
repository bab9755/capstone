from vi.util import Vector2
from vi.simulation import Shared
from constants import WIDTH, HEIGHT
import random
class Sensor:

    def __init__(self, agent):
        self.agent = agent

    def border_collision(self):
        pos = self.agent.pos

        if pos.x < 0 or pos.x > WIDTH or pos.y < 0 or pos.y > HEIGHT:
            return True
        return False



class Actuator: #this is what we are going to use to move the agent

    def __init__(self, agent):
        self.agent = agent
        self.agent.current_angle = random.uniform(0, 360)

    def update_position(self, linear_speed: int, angular_velocity: int): # this should run at every step

        self.agent.current_angle += angular_velocity
        self.agent.current_angle %= 360

        self.agent.move.from_polar((linear_speed, self.agent.current_angle)) #convert from polar to cartesian

        self.agent.pos += self.agent.move
        
        