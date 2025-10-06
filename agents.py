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
        self.role = "KNOWLEDGE_AGENT"
        self.pos.x = random.uniform(0, WIDTH)
        self.pos.y = random.uniform(0, HEIGHT)

        self.surrounding_state = "ALONE" # this is to track the state when the agent is surrounded by other ones or not
        self.object_state = "NONE"
        # story discovery tracking
        self.discovered_stories = set()
        
        # Timer for periodic LLM summarization (every 20 seconds = 20,000ms)
        self.last_summary_time = pg.time.get_ticks()
        self.summary_interval = 20000  # 20 seconds in milliseconds

        self.p = deque(maxlen=1)
        self.t_summary = deque(maxlen=1)
        self.t_received = deque(maxlen=1)


    def update(self): # at every tick (timestep), this function will be run

        neighbors = list(self.in_proximity_performance())
        subjects = [agent for agent in neighbors if agent.role == "SUBJECT"]
        agents = [agent for agent in neighbors if agent.role == "KNOWLEDGE_AGENT"]
        number_of_neighbors = len(agents)
        number_of_subjects = len(subjects)

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
                if number_of_subjects > 0:
                    print("Encountered a site")
                    self.sensor.collect_information_from_subjects(subjects)
                    self.object_state = "ON_SITE"
                    
            case "ON_SITE":
                if number_of_subjects == 0:
                    self.object_state = "NONE"

        # Check if it's time for periodic summarization (every 20 seconds)
        current_time = pg.time.get_ticks()
        if current_time - self.last_summary_time >= self.summary_interval:
            self.run_periodic_summarization()
            self.last_summary_time = current_time
        
        
    
    def run_periodic_summarization(self):
        """Run the LLM summarization chunk every 20 seconds"""
        # Only run if we have data to summarize
        if len(self.p) > 0 and len(self.t_summary) > 0 and len(self.t_received) > 0:
            prompt = f"You are a helpful assistant for a 2D agent simulation. Help me summarize the payload that I am giving you consizely and in a way that is easy to understand. \n\n p: {self.p} \n\n t_summary: {self.t_summary} \n\n t_received: {self.t_received}. Only return the summary, no other text."
            task_id = self.llm.submit(prompt)
            
            # Poll for results and handle them properly
            completed_tasks = self.llm.poll()
            for task_id_result, result in completed_tasks:
                if task_id_result == task_id:  # This is our task
                    self.t_summary.append(result)
                    break
            
            self.p.clear()
            self.t_received.clear()
            print(f"Agent {self.id} ran periodic summarization at {pg.time.get_ticks()}ms")

    def reset_proximity_state(self):
        """Reset the proximity state (useful for testing or new scenarios)"""
        self.seen_agents.clear()
        self.current_proximity_agents.clear()
        print(f"Agent {self.id} reset proximity state")

    def discover_story_information(self):
        """Discover story information when entering a site area and add to context."""
        info = story_registry.get_info_at_position(self.pos.x, self.pos.y)
        if not info or info in self.discovered_stories:
            print("Did not discover story information")
            return
        self.discovered_stories.add(info)
        self.add_context(f"DISCOVERY: {info}")

    def add_context(self, context: str):
        # Enqueue an async LLM summary request; append placeholder immediately
        context_str = context
        context_to_process = context_str + "\n" + self.context[-1] if self.context else context_str
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