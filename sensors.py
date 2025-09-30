from vi.util import Vector2
from vi.simulation import Shared
from constants import WIDTH, HEIGHT
import random
import pygame as pg
import copy
class Sensor:

    def __init__(self, agent):
        self.agent = agent

    def border_collision(self):
        pos = self.agent.pos

        if pos.x < 0 or pos.x > WIDTH or pos.y < 0 or pos.y > HEIGHT:
            return True
        return False

#TODO use a finite state machine with switch cases between ALONE -> NOT_ALONE
    def check_neighbors(self):
        current_time = pg.time.get_ticks()

        #before any processing, if an agent has been seen for more than 5 ticks, we can remove it from the seen_agents set
        # this is how we handle our unseen -> seen -> unseen cycle
        
        agents_to_remove = set()
        for agent in self.agent.seen_agents:
            if current_time - self.agent.seen_agents_time[agent] > 500:
                agents_to_remove.add(agent)
                print(f"Agent {self.agent.id} has not been seen for more than 500 ticks, removing it from the seen_agents set")
        #
        for agent in agents_to_remove:
            self.agent.seen_agents.remove(agent)
            self.agent.seen_agents_time.pop(agent)

        nearby_agents = list(self.agent.in_proximity_performance())
        current_agent_ids = {agent.id for agent in nearby_agents}
        
        new_agents = current_agent_ids - self.agent.seen_agents
        
        if new_agents:
            new_agent_objects = [agent for agent in nearby_agents if agent.id in new_agents]
            print(f"Agent {self.agent.id} is in proximity with NEW agents: {[agent.id for agent in new_agent_objects]}")
            self.agent.seen_agents.update(new_agents)
            for agent in new_agents:
                self.agent.seen_agents_time[agent] = current_time

            return new_agent_objects


    def exchange_context_with_agents(self, new_agent_objects):
        for agent in new_agent_objects:

            agent.add_context(self.agent.context[-1])



class Actuator: #this is what we are going to use to move the agent

    def __init__(self, agent):
        self.agent = agent
        self.agent.current_angle = random.uniform(0, 360)

    def update_position(self, linear_speed: int, angular_velocity: int): # this should run at every step

        self.agent.current_angle += angular_velocity
        self.agent.current_angle %= 360

        self.agent.move.from_polar((linear_speed, self.agent.current_angle)) #convert from polar to cartesian

        self.agent.pos += self.agent.move
        
        