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
        
        # Fixed number of data points for consistency
        self.num_snapshots = int(self.runtime_settings.get("num_snapshots", 30))
        self.snapshot_interval_seconds = float(self.runtime_settings.get("snapshot_interval_seconds", 10.0))
        self.snapshots_recorded = 0

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
        self.pending_score_futures: list[Future] = []  # Track pending score computations
        self.experiment_duration = 2000
        self.plot: LivePlot | None = None

        # Precompute ground-truth facts once for efficiency
        
        self._experiment_saved = False

    def _HeadlessSimulation__update_positions(self):
        for sprite in self._agents.sprites():
            agent: Agent = sprite

            linear_speed, angular_velocity = agent.get_velocities()
            agent.actuator.update_position(linear_speed, angular_velocity)

    def clean_llm_result(self, result: str) -> str:
        """Clean and validate LLM result. Returns empty string if invalid."""
        if not result:
            return ""
        
        cleaned = result.strip()
        while cleaned and cleaned[0] in '"\'\\ ' and cleaned[-1] in '"\'\\ ':
            cleaned = cleaned.strip().strip('"\'\\')
        empty_responses = {'""', "''", '\"\"', "\'\'", "null", "None", "N/A", "n/a", "empty"}
        if cleaned.lower() in empty_responses or cleaned in empty_responses:
            return ""
        
        # Skip very short nonsense
        if len(cleaned) < 5:
            return ""
        
        # Must contain at least one alphanumeric character
        if not any(c.isalnum() for c in cleaned):
            return ""
        
        return cleaned
    def run(self, plot: LivePlot | None = None):
        """
        Run the simulation until all snapshots are recorded.
        Duration = num_snapshots × snapshot_interval_seconds
        """
        self._running = True
        self.plot = plot if (plot is not None and self.live_plot_enabled) else None
        self.next_snapshot_time = self.snapshot_interval_seconds
        
        total_duration = self.num_snapshots * self.snapshot_interval_seconds
        print(f"🚀 Starting experiment: {self.num_snapshots} snapshots × {self.snapshot_interval_seconds}s = {total_duration}s total")
        
        while self._running:
            self.tick()
            self.tick_count += 1

            elapsed_seconds = self._elapsed_sim_seconds()
            
            # Check if it's time for a snapshot
            if elapsed_seconds >= self.next_snapshot_time and self.snapshots_recorded < self.num_snapshots:
                self.record_snapshot()
                self.next_snapshot_time += self.snapshot_interval_seconds
            
            # Stop when we have all snapshots
            if self.snapshots_recorded >= self.num_snapshots:
                print(f"✅ All {self.num_snapshots} snapshots recorded. Ending experiment.")
                self.stop()
                break

            self.process_score_queue()

        # Wait for all pending score computations to finish
        run_plot = self.plot
        self.wait_for_pending_scores()
        self.save_experiment_data(run_plot)
        if run_plot is not None:
            run_plot.show()

        return self._metrics


    def process_score_queue(self):
        while not self.score_queue.empty():
            try:
                agent_id, timestamp, score = self.score_queue.get_nowait()
                if self.plot is not None:
                    self.plot.update(timestamp, score, agent_id=agent_id)
            except queue.Empty:
                break



    def record_snapshot(self):
        """Record a snapshot for ALL agents. Always records exactly one data point per agent."""
        current_time = pg.time.get_ticks()
        snapshot_index = self.snapshots_recorded  # Current snapshot index (0-based)
        
        for agent in self._agents:
            if agent.role == "KNOWLEDGE_AGENT":
                summary = " ".join(agent.t_summary)
                agent_id = agent.id
                tick = current_time
                
                # Clean the summary
                cleaned_summary = self.clean_llm_result(summary)
                
                # Initialize score_by_index dict if not exists
                if not hasattr(agent, 'score_by_index'):
                    agent.score_by_index = {}
                
                # ALWAYS append to summary_history (empty string if no valid summary)
                agent.summary_history.append(cleaned_summary if cleaned_summary else "")
                
                if not cleaned_summary:
                    # No valid summary - record 0.0 score at this index
                    agent.score_by_index[snapshot_index] = 0.0
                    self.score_queue.put((agent_id, tick, 0.0))
                else:
                    # Valid summary - compute score asynchronously
                    metric_name = (self.metric or "").lower()
                    if metric_name == "cosine-bert":
                        future = self.executor.submit(compute_score, cleaned_summary, self.ground_truth_facts)
                    elif metric_name == "bert-score":
                        future = self.executor.submit(compute_bert_score, cleaned_summary, self.ground_truth_facts)
                    elif metric_name == "cosine-bm25":
                        future = self.executor.submit(compute_final_score, cleaned_summary, self.ground_truth_facts)
                    else:
                        raise ValueError(f"Invalid metric: {self.metric}")
                    
                    # Capture snapshot_index for this callback
                    knowledge_agent = agent
                    idx = snapshot_index
                    def on_complete(fut, knowledge_agent=knowledge_agent, aid=agent_id, t=tick, idx=idx):
                        try:
                            score = fut.result()
                            knowledge_agent.score_by_index[idx] = score  # Store by index, not append
                            self.score_queue.put((aid, t, score))
                        except Exception as e:
                            print(f"Error computing score for agent {aid}: {e}")
                            knowledge_agent.score_by_index[idx] = 0.0  # Fallback
                    
                    future.add_done_callback(lambda f, knowledge_agent=knowledge_agent, aid=agent_id, t=tick, idx=idx: on_complete(f, knowledge_agent, aid, t, idx))
                    self.pending_score_futures.append(future)
        
        self.snapshots_recorded += 1
        print(f"📸 Snapshot {self.snapshots_recorded}/{self.num_snapshots} recorded")

    def wait_for_pending_scores(self, timeout: float = 30.0):
        """Wait for all pending score computations to complete."""
        if not self.pending_score_futures:
            return
        print(f"⏳ Waiting for {len(self.pending_score_futures)} pending score computations...")
        for future in as_completed(self.pending_score_futures, timeout=timeout):
            pass  # Callbacks already handle the results
        # Clean up completed futures
        self.pending_score_futures = [f for f in self.pending_score_futures if not f.done()]
        print("✅ All score computations complete.")

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
        if self._experiment_saved:
            print("ℹ️ Experiment data already saved; skipping duplicate save.")
            return
        try:
            # Generate logical timestamps: [0, 10, 20, 30, ...] based on snapshot interval
            interval = self.snapshot_interval_seconds
            timestamps = [int(i * interval) for i in range(self.num_snapshots + 1)]
            
            # Build data structure with timestamp -> score/summary maps
            data = {
                "timestamps": timestamps,
                "snapshot_interval_seconds": interval,
                "num_snapshots": self.num_snapshots,
                "agents": {}
            }
            
            for agent in self._agents:
                if getattr(agent, "role", None) == "KNOWLEDGE_AGENT":
                    agent_key = str(agent.id)
                    summaries = list(getattr(agent, "summary_history", []))
                    score_by_index = getattr(agent, "score_by_index", {})
                    
                    # Build scores array from indexed dict (ensures correct order)
                    scores = [score_by_index.get(i, 0.0) for i in range(len(timestamps))]
                    
                    print(f"Agent {agent_key}: {len(summaries)} summaries, {len(scores)} scores")
                    
                    # Create timestamp -> value maps
                    data["agents"][agent_key] = {
                        "scores": {str(t): scores[i] if i < len(scores) else 0.0 for i, t in enumerate(timestamps)},
                        "summaries": {str(t): summaries[i] if i < len(summaries) else "" for i, t in enumerate(timestamps)}
                    }

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
                "num_snapshots": self.num_snapshots,
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
            self._experiment_saved = True
        except Exception as e:
            print(f"⚠️ Failed to save experiment data: {e}")
                
                