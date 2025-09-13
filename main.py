from vi import Agent, Config, Simulation, Window
from vi.util import count
import random

class myAgent(Agent):
    def update(self) -> None:
        if count(self.in_proximity_accuracy()) >= 3:
            self.change_image(0)
            self.freeze_movement()
            print("Freezing movement")
        else:
            self.change_image(1)
            self.continue_movement()

config = Config(window=Window(1000, 700))
x, y = config.window.as_tuple()
(
    Simulation(config)
    .spawn_obstacle("images/triangle.png", x//2, y//2)
    .spawn_site("images/obstacle.png", x=375, y=375)
    .batch_spawn_agents(100, myAgent, images=["/Users/boubalkaly/Desktop/development/capstone/test-violet/images/green.png", "/Users/boubalkaly/Desktop/development/capstone/test-violet/images/red.png"])
    .run()
)