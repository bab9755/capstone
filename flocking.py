from dataclasses import dataclass

from vi import Agent, Simulation
from vi.config import Config


@dataclass
class FlockingConfig(Config): ...


class FlockingAgent(Agent[FlockingConfig]): ...


(
    # Step 1: Create a new simulation.
    Simulation(FlockingConfig(image_rotation=True, movement_speed=1, radius=50))
    # Step 2: Add 50 agents to the simulation.
    .batch_spawn_agents(50, FlockingAgent, images=["images/triangle.png"])
    # Step 3: Profit! 🎉
    .run()
)
