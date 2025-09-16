from vi.util import Vector2
from vi.simulation import Shared
class Sensor:

    def __init__(self, agent):
        self.agent = agent

    def get_position(self) ->Vector2:
        return self.agent.pos #returns the position of the agent

    def get_next_move(self) -> Vector2:
        return self.agent.move #return the delta of the next move e.g (2, 1) 2 pixels on x and 1 on y



class Actuator: #this is what we are going to use to move the agent

    def __init__(self, agent):
        self.agent = agent

    def move(self, move: Vector2):
        self.agent.move = move 

    def change_image(self, image: str):
        self.agent.image = image 
        

    def update_position(self, angular_velocity: Vector2, linear_velocity: Vector2): # this should run at every step
        # Use simulator delta time to integrate motion
        delta_time = getattr(self.agent.shared, "delta_time", 0.0) # but then each delta time is equal to 1 step

        # Update position using current orientation vector `move`
        if linear_velocity:
            self.agent.pos += self.agent.move * (linear_velocity * delta_time)

        # Update orientation by rotating the orientation vector by the incremental angle
        if angular_velocity:
            incremental_angle = angular_velocity * delta_time
            # Rotate in-place to preserve vector magnitude
            self.agent.move.rotate_ip(incremental_angle)