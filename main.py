from vi import Agent, Config, Simulation


class myAgent(Agent):...
(
    Simulation(Config())
    .batch_spawn_agents(100, myAgent, images=["/Users/boubalkaly/Desktop/development/capstone/test-violet/images/green.png", "/Users/boubalkaly/Desktop/development/capstone/test-violet/images/red.png"])
    .run()
)