from llm import LLM
from vi import Agent
from context import Context
from sensors import Sensor, Actuator
import random
from constants import WIDTH, HEIGHT
import pygame as pg
from collections import deque
from story_registry import story_registry
class knowledgeAgent(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
        self.sensor = Sensor(self) # for global coordinate access
        self.actuator = Actuator(self) # for local coordinate access
        self.llm = LLM(self) 
        self.context = [] # list of context objects
        self.message_queue = deque() 
        self.pending_llm_tasks = {}  
        self.type = "SWARM"
        self.seen_agents = set()
        self.seen_agents_time = {} 

        self.pos.x = random.uniform(0, WIDTH)
        self.pos.y = random.uniform(0, HEIGHT)

        self.surrounding_state = "ALONE" # this is to track the state when the agent is surrounded by other ones or not
        self.object_state = "NONE"
        # story discovery tracking
        self.discovered_stories = set()


    def update(self): # at every tick (timestep), this function will be run

        neighbors = list(self.in_proximity_performance())
        number_of_neighbors = len(neighbors)
        is_on_site = self.on_site()

        #finite state machine for surrounding state
        match self.surrounding_state:
            case "ALONE":
                if number_of_neighbors > 0:
                    # here we will exchange information
                    self.sensor.exchange_context_with_agents(neighbors)
                    self.surrounding_state = "NOT_ALONE"
                    print(f"Agent {self.id} is now in the state NOT_ALONE")
            case "NOT_ALONE":
                if number_of_neighbors == 0:
                    self.surrounding_state = "ALONE"

        #finite state machine for object state
        match self.object_state:
            case "NONE":
                if is_on_site:
                    self.object_state = "ON_SITE"
                    self.discover_story_information()
            case "ON_SITE":
                if not is_on_site:
                    self.object_state = "NONE"

        for task_id, result in self.llm.poll():
            idx = self.pending_llm_tasks.pop(task_id, None)
            if idx is not None:
                self.context[idx] = result
                print(f"Agent {self.id} LLM summary ready")
                print(result)
        
        
    
    def reset_proximity_state(self):
        """Reset the proximity state (useful for testing or new scenarios)"""
        self.seen_agents.clear()
        self.current_proximity_agents.clear()
        print(f"Agent {self.id} reset proximity state")

    def discover_story_information(self):
        """Discover story information when entering a site area and add to context."""
        info = story_registry.get_info_at_position(self.pos.x, self.pos.y)
        if not info or info in self.discovered_stories:
            return
        self.discovered_stories.add(info)
        self.add_context(f"DISCOVERY: {info}")

    def add_context(self, context: str):
        # Enqueue an async LLM summary request; append placeholder immediately
        context_str = context
        context_to_process = context_str + "\n" + self.context[-1]
        self.context.append("[summarizing…]")
        placeholder_index = len(self.context) - 1
        task_id = self.llm.submit(context_to_process)
        self.pending_llm_tasks[task_id] = placeholder_index
        print(f"Agent {self.id} queued LLM summarization task {task_id}")

    def get_velocities(self): 
        # at the start of the simulation this is what we get
        linear_speed = 2
        angular_velocity = 0.0

        if self.sensor.border_collision(): # if the agent is on the edge of the world, it randomly changes its angular velocity
            angular_velocity = 10
            linear_speed = 2

        return linear_speed, angular_velocity



    


    
class Villager(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sensor = Sensor(self) # for global coordinate access
        self.actuator = Actuator(self) # for local coordinate access
        self.llm = LLM(self) 
        self.context = context
        self.message_queue = deque() 
        self.pending_llm_tasks = {}  
        self.type = "ENV"

        self.seen_agents = set()
        self.seen_agents_time = {} 

        self.pos.x = random.uniform(0, WIDTH)
        self.pos.y = random.uniform(0, HEIGHT)

    def update(self):
        pass

    def get_velocities(self):
        return 0, 0