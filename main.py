from vi import Agent, Config, Simulation, Window
from vi.util import count
import random

class myAgent(Agent):
    def update(self) -> None:
        self.check_site()

    def check_site(self):
        if self.on_site():
            site_id = self.on_site_id()
            print(f"Agent with id {self.id} is on site {site_id}")
            return

config = Config(window=Window(1000, 700))
x, y = config.window.as_tuple()
(
    Simulation(config)
    .spawn_obstacle("images/triangle.png", x//2, y//2)
    .spawn_site("images/triangle.png", x=random.randint(0, x), y=random.randint(0, y))
    .spawn_site("images/triangle.png", x=random.randint(0, x), y=random.randint(0, y))
    .spawn_site("images/triangle.png", x=random.randint(0, x), y=random.randint(0, y))
    .spawn_site("images/triangle.png", x=random.randint(0, x), y=random.randint(0, y))
    .batch_spawn_agents(10, myAgent, images=["/Users/boubalkaly/Desktop/development/capstone/test-violet/images/green.png", "/Users/boubalkaly/Desktop/development/capstone/test-violet/images/red.png"])
    .run()
)