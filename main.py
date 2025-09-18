from vi import Agent, Config, Simulation, Window
from vi.util import count
import random
from world import build_environment, Environment, ItemSpec, WorldConfig
class myAgent(Agent):
    def update(self) -> None:
        self.check_obstacle()

    def check_obstacle(self):
        if count(self.obstacle_intersections()) >= 1:
            print(f"Agent with id {self.id} is on obstacle")
            return

config = Config(window=Window(1000, 700))
x, y = config.window.as_tuple()

environment = Environment(seed=1, world=WorldConfig(width=x, height=y), obstacles=[ItemSpec(image="images/triangle.png", count=20)], sites=[ItemSpec(image="images/red.png", count=1)])
env = build_environment(environment)

simulation = Simulation(config)

for ob in env["obstacles"]:
    simulation.spawn_obstacle(ob["image"], ob["x"], ob["y"])
for site in env["sites"]:
    simulation.spawn_site(site["image"], x=site["x"], y=site["y"])

(
    simulation
    .spawn_obstacle("images/triangle.png", x//2, y//2)
    .spawn_site("images/triangle.png", x=random.randint(0, x), y=random.randint(0, y))
    .spawn_site("images/triangle.png", x=random.randint(0, x), y=random.randint(0, y))
    .spawn_site("images/triangle.png", x=random.randint(0, x), y=random.randint(0, y))
    .spawn_site("images/triangle.png", x=random.randint(0, x), y=random.randint(0, y))
    .batch_spawn_agents(10, myAgent, images=["/Users/boubalkaly/Desktop/development/capstone/test-violet/images/walle.png", "/Users/boubalkaly/Desktop/development/capstone/test-violet/images/red.png"])
    .run()
)