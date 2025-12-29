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
        self.actuator = Actuator(self) 
        self.llm = LLM(self) 
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
        # NOTE: Only update t_summary here. summary_history is managed by record_snapshot()
        for task_id_result, result in self.llm.poll():
            if self._pending_interaction_task_id and task_id_result == self._pending_interaction_task_id:
                print(f"Agent {self.id} received interaction summary: {result}")
                self.t_summary.append(result)
                self._pending_interaction_task_id = None
            elif self._pending_private_info_task_id and task_id_result == self._pending_private_info_task_id:
                print(f"Agent {self.id} received private info summary: {result}")
                self.t_summary.append(result)
                self._pending_private_info_task_id = None
            elif self._pending_summary_task_id and task_id_result == self._pending_summary_task_id:
                print(f"Agent {self.id} received LLM summary: {result}")
                self.t_summary.append(result)
                self._pending_summary_task_id = None
        
        
    
    def summarize_interaction(self):
        """
        Summarize interaction between agents by merging this agent's summary 
        with summaries received from other agents.
        """
        if self.is_llm_busy():
            return
        
        own_summary = " ".join(self.t_summary) if self.t_summary else ""
        received_summaries = " ".join(self.t_received)
        
        print(f"Agent {self.id} merging summaries for interaction...")
        self._pending_interaction_task_id = self.llm.submit_interaction(
            summary=own_summary,
            received_info=received_summaries
        )
        print(f"Agent {self.id} scheduled interaction summarization at {pg.time.get_ticks()}ms")
    
    def summarize_private_information(self):
        """
        Summarize private information collected from sites (stored in p).
        """
        if self.is_llm_busy():
            return

        own_summary = " ".join(self.t_summary) if self.t_summary else ""
        private_info = " ".join(self.p) if self.p else ""
        
        print(f"Agent {self.id} summarizing private information...")
        self._pending_private_info_task_id = self.llm.submit_private_info(
            summary=own_summary,
            private_info=private_info
        )
        print(f"Agent {self.id} scheduled private information summarization at {pg.time.get_ticks()}ms")
    def get_velocities(self): 
        # at the start of the simulation this is what we get
        linear_speed = 2
        angular_velocity = 0.0

        if self.sensor.border_collision(): # if the agent is on the edge of the world, it randomly changes its angular velocity
            angular_velocity = 10
            linear_speed = 5

        return linear_speed, angular_velocity


    # removed DB logging