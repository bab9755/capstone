from vi import Agent, Config, Simulation, Window
from communication import CommunicationManager
from constants import fragments, ground_truth, ground_truth_summary, ground_truth_facts, ground_truth_text
from sentence_transformers import SentenceTransformer, util
import json
import pandas as pd
import matplotlib.pyplot as plt
from llm import LLM
import pygame as pg
from metrics import compute_bert_score, compute_score, compute_final_score
from visualize import LivePlot
import queue
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from helpers import load_config
from pathlib import Path

config = load_config()
metric = config.get("metric", "cosine-bert")

social_learning_enabled = config.get("social_learning_enabled", False)
num_knowledge_agents = config.get("agents", {}).get("knowledge_agents", 10)
metric = config.get("metric", "cosine-bert")


class Environment(Simulation):
    def __init__(self, llm_provider: str, llm_model: str, config=None,  num_knowledge_agents: int = 2, num_subject_agents: int = 5):
        
        super().__init__(config)

        self.start_time = pg.time.get_ticks()
        self.last_plot_time = self.start_time
        self.tick_count = 0

        self.executor = ThreadPoolExecutor(max_workers=10)
        self.score_queue = queue.Queue()

        # Precompute ground-truth facts once for efficiency
        

    def _HeadlessSimulation__update_positions(self):
        for sprite in self._agents.sprites():
            agent: Agent = sprite

            linear_speed, angular_velocity = agent.get_velocities()
            agent.actuator.update_position(linear_speed, angular_velocity)

    def run(self, plot: LivePlot):
        """Run the simulation until it's ended by closing the window or when the `vi.config.Schema.duration` has elapsed."""
        self._running = True
        self.plot = plot
        
        # Maximum simulated seconds
        MAX_SIMULATED_SECONDS = 300000  # 300000 simulated seconds

        while self._running:
            self.tick()
            self.tick_count += 1  # Increment simulated time counter

            # Check if we've reached the maximum simulated time
            if self.tick_count >= MAX_SIMULATED_SECONDS:
                print(f"⏱️  Simulation stopped: 300000 simulated seconds reached ({self.tick_count} ticks)")
                self._running = False
                break

            self.process_score_queue()
            if pg.time.get_ticks() - self.last_plot_time >= 3000: #update the plot every 5 seconds
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
                    agent.summary_history.append("")
                    agent.score_history.append(0.0)
                    self.score_queue.put((agent_id, tick, 0.0))
                else:
                    if metric == "cosine-bert":
                        future = self.executor.submit(compute_score, summary, ground_truth_text)
                    elif metric == "bert-score":
                        future = self.executor.submit(compute_bert_score, summary, ground_truth_text)
                    elif metric == "cosine-bm25":
                        future = self.executor.submit(compute_final_score, summary, ground_truth_text)
                    else:
                        raise ValueError(f"Invalid metric: {metric}")
                    knowledge_agent = agent
                    def on_complete(fut, knowledge_agent=knowledge_agent, aid=agent_id, t=tick):
                        try:
                            score = fut.result()
                            knowledge_agent.score_history.append(score)
                            self.score_queue.put((aid, t, score))
                        except Exception as e:
                            print(f"Error computing BERT score for agent {aid} at tick {t}: {e}")
                    
                    future.add_done_callback(lambda f, knowledge_agent=knowledge_agent, aid=agent_id, t=tick: on_complete(f, knowledge_agent, aid, t))
    
    def save_experiment_data(self, plot: LivePlot):
        """Save experiment data and plot to the experiments directory"""
        try:
            # Collect summaries per agent (keys as string agent ids)
            data = {}
            for agent in self._agents:
                if getattr(agent, "role", None) == "KNOWLEDGE_AGENT":
                    agent_key = str(agent.id)
                    summaries = getattr(agent, "summary_history", None)
                    if summaries is None:
                        last = list(getattr(agent, "t_summary", []))
                        summaries = last if last else []
                    data[agent_key] = list(summaries)
                    data[agent_key + "_score"] = list(agent.score_history)
            # Root experiments directory
            root_dir = None

            if num_knowledge_agents == 1:
                root_dir = Path("experiments/single_agent")
            elif num_knowledge_agents > 1:
                if social_learning_enabled:
                    root_dir = Path("experiments/swarm_social_learning")
                else:
                    root_dir = Path("experiments/swarm_self_learning")
            
            
            if root_dir is not None:
                root_dir.mkdir(parents=True, exist_ok=True)

            # Determine next experiment folder number
            existing_dirs = [p for p in root_dir.iterdir() if p.is_dir() and p.name.isdigit()]
            next_idx = 1
            if existing_dirs:
                try:
                    next_idx = max(int(p.name) for p in existing_dirs) + 1
                except ValueError:
                    next_idx = 1

            run_dir = root_dir / str(next_idx)
            run_dir.mkdir(parents=True, exist_ok=True)

            # Save JSON
            json_path = run_dir / "experiment.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            # Save plot image
            if plot is not None:
                img_path = run_dir / "plot.png"
                try:
                    plot.save(img_path)
                except Exception as e_img:
                    print(f"⚠️ Failed to save plot image: {e_img}")

            print(f"💾 Saved experiment to {run_dir}")
        except Exception as e:
            print(f"⚠️ Failed to save experiment data: {e}")
                
                