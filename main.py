from vi import Agent, Config, Simulation
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


(
    Simulation(Config())
    .batch_spawn_agents(100, myAgent, images=["/Users/boubalkaly/Desktop/development/capstone/test-violet/images/green.png", "/Users/boubalkaly/Desktop/development/capstone/test-violet/images/red.png"])
    .run()
)