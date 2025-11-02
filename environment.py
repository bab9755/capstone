from vi import Agent, Config, Simulation, Window
from communication import CommunicationManager
from constants import fragments
from sentence_transformers import SentenceTransformer, util
import json
import pandas as pd
import matplotlib.pyplot as plt
from llm import LLM
import pygame as pg
from metrics import compute_bert_score
from visualize import LivePlot
import queue
from concurrent.futures import ThreadPoolExecutor, Future, as_completed


ground_truth = """On a sunny Saturday, families explore the city zoo. Children rush to watch the lions being fed, while others gather by the pond where ducks splash and a little girl tosses crumbs. A vendor serves ice cream near a bench where an elderly couple enjoys the parrots’ chatter. Inside the humid reptile house, students sketch snakes for a biology project. A loudspeaker announces the upcoming penguin show, and crowds head toward the stadium. A child’s balloon drifts into a tree as laughter fills the air, blending with animal calls and the scent of popcorn."""

class Environment(Simulation):
    def __init__(self, llm_provider: str, llm_model: str, config=None,  num_knowledge_agents: int = 2, num_subject_agents: int = 5, fragments: list[str] = fragments):
        super().__init__(config)

        self.start_time = pg.time.get_ticks()
        self.last_plot_time = self.start_time

        self.executor = ThreadPoolExecutor(max_workers=10)
        self.score_queue = queue.Queue()

        

    def _HeadlessSimulation__update_positions(self):
        for sprite in self._agents.sprites():
            agent: Agent = sprite

            linear_speed, angular_velocity = agent.get_velocities()
            agent.actuator.update_position(linear_speed, angular_velocity)

    def run(self, plot: LivePlot):
        """Run the simulation until it's ended by closing the window or when the `vi.config.Schema.duration` has elapsed."""
        self._running = True
        self.plot = plot

        while self._running:
            self.tick()

            self.process_score_queue()
            if pg.time.get_ticks() - self.last_plot_time >= 5000: #update the plot every 5 seconds
                self.update_plot(self.plot)
                self.last_plot_time = pg.time.get_ticks()

        return self._metrics


    def process_score_queue(self):
        while not self.score_queue.empty():
            try:
                agent_id, timestamp, score = self.score_queue.get_nowait()
                self.plot.update(timestamp, score, agent_id=agent_id)
            except queue.Empty:
                break



    def update_plot(self, plot): # this will run every 5 seconds
        current_time = pg.time.get_ticks()
        for agent in self._agents:
            if agent.role == "KNOWLEDGE_AGENT":
                summary = " ".join(agent.t_summary)
                
                agent_id = agent.id
                tick = current_time
                
                if not summary or summary.strip() == "":
                    self.score_queue.put((agent_id, tick, 0.0))
                else:
                    future = self.executor.submit(compute_bert_score, summary, ground_truth)
                    
                    def on_complete(fut, aid=agent_id, t=tick):
                        try:
                            score = fut.result()
                            self.score_queue.put((aid, t, score))
                        except Exception as e:
                            print(f"Error computing BERT score for agent {aid} at tick {t}: {e}")
                    
                    future.add_done_callback(lambda f, aid=agent_id, t=tick: on_complete(f, aid, t))
                
                