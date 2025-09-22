from llm import LLM
from vi import Agent
from context import Context
from sensors import Sensor, Actuator
import random
from constants import WIDTH, HEIGHT
import pygame as pg
from collections import deque
class knowledgeAgent(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
        self.sensor = Sensor(self) # for global coordinate access
        self.actuator = Actuator(self) # for local coordinate access
        self.llm = LLM(self) 
        self.context = ["This is the initial context of agent " + str(self.id)] # list of context objects
        self.message_queue = deque() 
        self.pending_llm_tasks = {}  

        self.seen_agents = set()
        self.seen_agents_time = {} 

        self.pos.x = random.uniform(0, WIDTH)
        self.pos.y = random.uniform(0, HEIGHT)


    def update(self): # at every tick (timestep), this function will be run
        # State machine for proximity interactions

        # this checks all the neighbors and updates the seen_agents set
        agents = self.sensor.check_neighbors()
        if agents:

            self.sensor.exchange_context_with_agents(agents)

        # Process completed LLM tasks without blocking the sim loop
        for task_id, result in self.llm.poll():
            idx = self.pending_llm_tasks.pop(task_id, None)
            if idx is not None:
                # Replace placeholder with actual result
                self.context[idx] = result
                print(f"Agent {self.id} LLM summary ready")
            
    
    def reset_proximity_state(self):
        """Reset the proximity state (useful for testing or new scenarios)"""
        self.seen_agents.clear()
        self.current_proximity_agents.clear()
        print(f"Agent {self.id} reset proximity state")

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



    


    