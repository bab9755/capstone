from llm import LLM
from vi import Agent
from sensors import Sensor, Actuator
import random
import pygame as pg
from collections import deque
from runtime_config import get_runtime_settings

_SETTINGS = get_runtime_settings()
_ENV_WIDTH = _SETTINGS["environment"]["width"]
_ENV_HEIGHT = _SETTINGS["environment"]["height"]

from story_registry import story_registry
class knowledgeAgent(Agent):
    def __init__(self, context_size: int = 2, social_learning_enabled: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
        self.sensor = Sensor(self) # for global coordinate access
        self.actuator = Actuator(self) # for local coordinate access
        self.llm = LLM(self) 
        self.context = [] # list of context objects
        self.message_queue = deque() 
        self.pending_llm_tasks = {}  
        self.role = "KNOWLEDGE_AGENT"
        self.pos.x = random.uniform(0, _ENV_WIDTH)
        self.pos.y = random.uniform(0, _ENV_HEIGHT)

        self.surrounding_state = "ALONE" # this is to track the state when the agent is surrounded by other ones or not
        self.object_state = "NONE"
        # story discovery tracking
        self.discovered_stories = set()
        self.agent_id = None

        
        
        # Timer for periodic LLM summarization (every 20 seconds = 20,000ms)
        self.last_summary_time = pg.time.get_ticks()
        self.summary_interval = 10000  # 20 seconds in milliseconds

        self.p = deque(maxlen=context_size)
        self.t_summary = deque(maxlen=1)
        self.t_received = deque(maxlen=context_size)
        self._pending_summary_task_id = None
        self._pending_interaction_task_id = None
        self._pending_private_info_task_id = None
        self.social_learning_enabled = social_learning_enabled
        self.summary_history = []
        self.score_history = []
        #

    def is_llm_busy(self):
        """Check if agent is currently processing any LLM task"""
        has_pending_summary = (
            self._pending_summary_task_id is not None or
            self._pending_interaction_task_id is not None or
            self._pending_private_info_task_id is not None
        )
        if has_pending_summary or (len(self.pending_llm_tasks) > 0):
            print(f"Agent {self.id} is busy with pending tasks.")
            return True
        else:
            return False

    def update(self): # at every tick (timestep), this function will be run

        neighbors = list(self.in_proximity_performance())
        subjects = [agent for agent in neighbors if agent.role == "SUBJECT"]
        agents = [agent for agent in neighbors if agent.role == "KNOWLEDGE_AGENT"]
        number_of_neighbors = len(agents)
        number_of_subjects = len(subjects)

        #finite state machine for surrounding state

        if self.social_learning_enabled:
            match self.surrounding_state:
                case "ALONE":
                    if number_of_neighbors > 0:
                        # here we will exchange information (if not busy)
                        if not self.is_llm_busy():
                            self.sensor.exchange_context_with_agents(agents)
                            self.summarize_interaction()
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
                    if not self.is_llm_busy():
                        self.sensor.collect_information_from_subjects(subjects)
                        self.summarize_private_information()
                    self.object_state = "ON_SITE"
                    
            case "ON_SITE":
                if number_of_subjects == 0:
                    self.object_state = "NONE"

        # Process any completed LLM tasks
        for task_id_result, result in self.llm.poll():
            # Interaction summarization completion (merge summaries from agent interaction)
            if self._pending_interaction_task_id and task_id_result == self._pending_interaction_task_id:
                print(f"Agent {self.id} received interaction summary: {result}")
                self.t_summary.append(result)
                self.summary_history.append(result)
                self._pending_interaction_task_id = None
            # Private information summarization completion
            elif self._pending_private_info_task_id and task_id_result == self._pending_private_info_task_id:
                print(f"Agent {self.id} received private info summary: {result}")
                self.t_summary.append(result)
                self.summary_history.append(result)
                self._pending_private_info_task_id = None
            # Legacy periodic summarization completion (for backward compatibility)
            elif self._pending_summary_task_id and task_id_result == self._pending_summary_task_id:
                print(f"Agent {self.id} received LLM summary: {result}")
                self.t_summary.append(result)
                self.summary_history.append(result)
                self._pending_summary_task_id = None
        
        
    
    def summarize_interaction(self):
        """
        Summarize interaction between agents by merging this agent's summary 
        with summaries received from other agents.
        """
        # Avoid overlapping tasks
        if self._pending_interaction_task_id is not None:
            return
        
        # Get agent's own summary
        own_summary = " ".join(self.t_summary) if self.t_summary else ""
        
        received_summaries = " ".join(self.t_received)
        context_str = own_summary + " " + received_summaries
        print(f"Agent {self.id} merging summaries for interaction: {context_str[:100]}...")
        
        self._pending_interaction_task_id = self.llm.submit(context_str)
        print(f"Agent {self.id} scheduled interaction summarization at {pg.time.get_ticks()}ms")
    
    def summarize_private_information(self):
        """
        Summarize private information collected from sites (stored in p).
        """
        # Avoid overlapping tasks
        if self._pending_private_info_task_id is not None:
            return

        own_summary = self.t_summary[-1] if len(self.t_summary) > 0 else ""
        # Get private information collected from sites
        private_info = " ".join(self.p) if self.p else ""
        
        # Only proceed if we have private information to summarize

        
        context_str = own_summary + " " + private_info
        print(f"Agent {self.id} summarizing private information: {context_str[:100]}...")
        
        self._pending_private_info_task_id = self.llm.submit(context_str)
        print(f"Agent {self.id} scheduled private information summarization at {pg.time.get_ticks()}ms")
    
    def run_periodic_summarization(self):
        """
        Legacy method - kept for backward compatibility.
        Use summarize_interaction() or summarize_private_information() instead.
        """
        # Avoid overlapping tasks
        if self._pending_summary_task_id is not None:
            return
        # Only run if there is something to summarize
        print(f"Agent {self.id} is sending the following payload to the LLM: {list(self.p)} {list(self.t_summary)} {list(self.t_received)}")
        context = list(self.p) + list(self.t_summary) + list(self.t_received)
        context_str = " ".join(context)
        self._pending_summary_task_id = self.llm.submit(context_str)
        # Clear inputs for next window
        print(f"Agent {self.id} scheduled periodic summarization at {pg.time.get_ticks()}ms")

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


    # removed DB logging


    


    
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

        self.pos.x = random.uniform(0, _ENV_WIDTH)
        self.pos.y = random.uniform(0, _ENV_HEIGHT)

    def update(self):
        pass

    def get_velocities(self):
        return 0, 0