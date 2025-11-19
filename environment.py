from __future__ import annotations

from vi import Agent, Config, Simulation, Window, HeadlessSimulation
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
from pathlib import Path
from runtime_config import get_runtime_settings
from datetime import datetime


class Environment(Simulation):
    def __init__(self, llm_provider: str, llm_model: str, config=None,  num_knowledge_agents: int = 2, num_subject_agents: int = 5):
        
        super().__init__(config)

        self.runtime_settings = get_runtime_settings()
        self.metric = self.runtime_settings["metric"]
        self.num_knowledge_agents = num_knowledge_agents
        self.num_subject_agents = num_subject_agents
        self.social_learning_enabled = self.runtime_settings["social_learning_enabled"]
        self.profile_key = self.runtime_settings.get("profile_key")

        live_plot_settings = (
            (self.runtime_settings.get("visualization") or {}).get("live_plot") or {}
        )
        self.live_plot_enabled = bool(live_plot_settings.get("enabled", True))
        self.plot_update_interval_ms = int(
            live_plot_settings.get("update_interval_ms", 3000)
        )

        ground_truth_bundle = self.runtime_settings["ground_truth"]
        self.ground_truth_key = ground_truth_bundle.get("name")
        self.ground_truth_text = ground_truth_bundle.get("text", "")
        self.ground_truth_facts = ground_truth_bundle.get("facts", [])
        self.ground_truth_summary = ground_truth_bundle.get("summary", "")
 
        self.start_time = pg.time.get_ticks()
        self.last_plot_time = self.start_time
        self.tick_count = 0

        self.executor = ThreadPoolExecutor(max_workers=10)
        self.score_queue = queue.Queue()
        self.experiment_duration = 2000
        self.plot: LivePlot | None = None

        # Precompute ground-truth facts once for efficiency
        

    def _HeadlessSimulation__update_positions(self):
        for sprite in self._agents.sprites():
            agent: Agent = sprite

            linear_speed, angular_velocity = agent.get_velocities()
            agent.actuator.update_position(linear_speed, angular_velocity)

    def run(self, plot: LivePlot | None = None, *, max_duration_seconds: float | None = None):
        """Run the simulation until it's ended by closing the window, `vi.config.Schema.duration`, or max_duration_seconds."""
        self._running = True
        self.plot = plot if (plot is not None and self.live_plot_enabled) else None
        
        while self._running:
            self.tick()
            self.tick_count += 1  # Increment simulated time counter

            if max_duration_seconds is not None:
                elapsed_seconds = self._elapsed_sim_seconds()
                if elapsed_seconds >= max_duration_seconds:
                    self.stop()
                    break

            self.process_score_queue()
            if (
                pg.time.get_ticks() - self.last_plot_time
                >= self.plot_update_interval_ms
            ):  # update cadence configurable via runtime settings
                self.update_plot(self.plot)
                self.last_plot_time = pg.time.get_ticks()

        return self._metrics


    def process_score_queue(self):
        while not self.score_queue.empty():
            try:
                agent_id, timestamp, score = self.score_queue.get_nowait()
                if self.plot is not None:
                    self.plot.update(timestamp, score, agent_id=agent_id)
            except queue.Empty:
                break



    def update_plot(self, plot):  # triggered on the configured cadence
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
                    metric_name = (self.metric or "").lower()
                    if metric_name == "cosine-bert":
                        future = self.executor.submit(compute_score, summary, self.ground_truth_facts)
                    elif metric_name == "bert-score":
                        future = self.executor.submit(compute_bert_score, summary, self.ground_truth_facts)
                    elif metric_name == "cosine-bm25":
                        future = self.executor.submit(compute_final_score, summary, self.ground_truth_facts)
                    else:
                        raise ValueError(f"Invalid metric: {self.metric}")
                    knowledge_agent = agent
                    def on_complete(fut, knowledge_agent=knowledge_agent, aid=agent_id, t=tick):
                        try:
                            score = fut.result()
                            knowledge_agent.score_history.append(score)
                            self.score_queue.put((aid, t, score))
                        except Exception as e:
                            print(f"Error computing BERT score for agent {aid} at tick {t}: {e}")
                    
                    future.add_done_callback(lambda f, knowledge_agent=knowledge_agent, aid=agent_id, t=tick: on_complete(f, knowledge_agent, aid, t))

    def _elapsed_sim_seconds(self) -> float:
        fps = getattr(self.config, "fps", None) if hasattr(self, "config") else None
        if fps:
            try:
                fps_value = float(fps)
                if fps_value > 0:
                    return self.tick_count / fps_value
            except (TypeError, ValueError):
                pass
        return (pg.time.get_ticks() - self.start_time) / 1000.0
    
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

            base_dir = Path("experiments")
            profile_key = (self.profile_key or "unspecified").replace(" ", "_")
            swarm_type = self.runtime_settings.get("swarm_type") or ("social_learning" if self.social_learning_enabled else "self_learning")
            if self.num_knowledge_agents == 1:
                cohort_dirname = "single_agent"
            else:
                cohort_dirname = f"{swarm_type}_swarm"

            agents_dir = base_dir / profile_key / cohort_dirname / f"ka-{self.num_knowledge_agents}"
            scenario_dirname = (self.ground_truth_key or "ground_truth").replace(" ", "_")
            scenario_dir = agents_dir / scenario_dirname
            scenario_dir.mkdir(parents=True, exist_ok=True)

            existing_runs = [p for p in scenario_dir.iterdir() if p.is_dir() and p.name.startswith("run_")]
            next_idx = 1
            if existing_runs:
                try:
                    next_idx = max(int(p.name.split("_")[-1]) for p in existing_runs) + 1
                except ValueError:
                    next_idx = 1

            run_dir = scenario_dir / f"run_{next_idx:04d}"
            run_dir.mkdir(parents=True, exist_ok=True)

            metadata = {
                "created_at_utc": datetime.utcnow().isoformat() + "Z",
                "profile": profile_key,
                "swarm_type": swarm_type,
                "cohort_dirname": cohort_dirname,
                "num_knowledge_agents": self.num_knowledge_agents,
                "num_subject_agents": self.num_subject_agents,
                "social_learning_enabled": self.social_learning_enabled,
                "metric": self.metric,
                "ground_truth_key": self.ground_truth_key,
                "ground_truth_summary": self.ground_truth_summary,
                "ground_truth_snippet_count": len(self.runtime_settings["ground_truth"].get("snippets", [])),
                "context": self.runtime_settings.get("context", {}),
            }
 
            # Save JSON
            json_path = run_dir / "experiment.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            metadata_path = run_dir / "metadata.json"
            with open(metadata_path, "w", encoding="utf-8") as meta_file:
                json.dump(metadata, meta_file, ensure_ascii=False, indent=2)

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
                
                