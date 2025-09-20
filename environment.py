from vi import Agent, Config, Simulation, Window


class Environment(Simulation):
    def __init__(self, config = None):
        super().__init__(config)

    def _HeadlessSimulation__update_positions(self):
        for sprite in self._agents.sprites():
            agent: Agent = sprite

            linear_speed, angular_velocity = agent.get_velocities()
            agent.actuator.update_position(linear_speed, angular_velocity)