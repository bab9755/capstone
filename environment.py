from __future__ import annotations

from vi import Agent, Config, Simulation, Window, HeadlessSimulation
from sentence_transformers import SentenceTransformer, util
import json
import numpy as np
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
import random
import math


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
        self.num_snapshots = int(self.runtime_settings.get("num_snapshots", 60))
        self.snapshot_interval_seconds = float(self.runtime_settings.get("snapshot_interval_seconds", 2.0))
        self.snapshots_recorded = 0

        # Subject visibility / information teleportation settings
        teleport_settings = self.runtime_settings.get("information_teleportation", {})
        self.teleportation_enabled = teleport_settings.get("enabled", False)
        self.teleportation_mode = teleport_settings.get("mode", "shuffle")  # "shuffle", "decay", or "dynamic_pool"
        self.subject_visibility_probability = teleport_settings.get("visibility_probability", 0.75)
        self.decay_count = teleport_settings.get("decay_count", 3)  # subjects to remove per interval (decay mode)
        self.decay_probability = teleport_settings.get("decay_probability", 0.08)  # legacy per-subject leave probability
        self.visibility_interval = teleport_settings.get("interval_seconds", 5.0)
        self.next_visibility_time = self.visibility_interval
        
        # Dynamic pool / continuous decay mode settings
        self.initial_active_count = teleport_settings.get("initial_active_count", 5)
        self.appearance_mean_time = teleport_settings.get("appearance_mean_time", 5.0)
        self.lifetime_mean_time = teleport_settings.get("lifetime_mean_time", 10.0)
        
        # Subject lifetime state
        # - Used by dynamic_pool mode (with snippet pool + recycling)
        # - Also reused by decay mode (no recycling, just one-shot disappearance)
        self.snippet_pool = []  # Inactive snippets waiting to appear (dynamic_pool only)
        self.next_appearance_time = None  # When the next snippet will appear (dynamic_pool only)
        self.subject_lifetimes = {}  # Maps subject agent -> expiry time in seconds

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
            
            # Information teleportation / information decay
            if self.teleportation_enabled:
                if self.teleportation_mode == "dynamic_pool":
                    # Dynamic pool mode: continuous checking (not interval-based)
                    self.dynamic_pool_update()
                elif self.teleportation_mode == "decay":
                    # Continuous-time exponential decay of subjects (no reappearance)
                    self.decay_subject_visibility()
                elif elapsed_seconds >= self.next_visibility_time:
                    # Default "shuffle" visibility pattern at fixed intervals
                    self.shuffle_subject_visibility()
                    self.next_visibility_time += self.visibility_interval
            
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

    def shuffle_subject_visibility(self):
        """
        Randomly toggle visibility of subject agents based on visibility_probability.
        This implements 'information teleportation' - subjects appear/disappear over time.
        """
        subjects = [a for a in self._agents if getattr(a, "role", None) == "SUBJECT"]
        if not subjects:
            return
        
        visible_count = 0
        for subject in subjects:
            is_visible = random.random() < self.subject_visibility_probability
            subject.set_visible(is_visible)
            if is_visible:
                visible_count += 1
        
        print(f"🔀 Shuffled subject visibility: {visible_count}/{len(subjects)} visible ({self.subject_visibility_probability:.0%} probability)")

    def fixed_decay_subject_visibility(self):
        """
        Legacy fixed-count decay:
        Permanently hide a fixed number of visible subjects at each decay step.
        """
        visible_subjects = [a for a in self._agents if getattr(a, "role", None) == "SUBJECT" and getattr(a, "visible", True)]
        if not visible_subjects:
            print("⚠️ No visible subjects remaining to decay")
            return
        
        # Randomly select subjects to hide (up to decay_count or remaining visible)
        num_to_hide = min(self.decay_count, len(visible_subjects))
        subjects_to_hide = random.sample(visible_subjects, num_to_hide)
        
        for subject in subjects_to_hide:
            subject.set_visible(False)
        
        remaining_visible = len(visible_subjects) - num_to_hide
        total_subjects = len([a for a in self._agents if getattr(a, "role", None) == "SUBJECT"])
        print(f"📉 Fixed decay: {num_to_hide} hidden, {remaining_visible}/{total_subjects} still visible")

    def decay_subject_visibility(self):
        """
        Continuous-time exponential decay of subject visibility, similar in spirit
        to dynamic_pool but without reappearing snippets:

        - Each SUBJECT agent is assigned an independent exponential lifetime
          with rate λ = 1 / lifetime_mean_time (if provided).
        - When the current simulated time exceeds that agent's expiry time,
          the subject becomes permanently invisible.
        """
        subjects = [a for a in self._agents if getattr(a, "role", None) == "SUBJECT"]
        if not subjects:
            return

        current_time = self._elapsed_sim_seconds()

        # Determine exponential rate λ. Prefer explicit lifetime_mean_time;
        # if it's not set, approximate from decay_probability / interval.
        lambda_rate = 0.0
        if getattr(self, "lifetime_mean_time", None) and self.lifetime_mean_time > 0:
            lambda_rate = 1.0 / float(self.lifetime_mean_time)
        elif self.decay_probability > 0 and self.visibility_interval > 0:
            approx_mean = float(self.visibility_interval) / float(self.decay_probability)
            lambda_rate = 1.0 / approx_mean

        if lambda_rate <= 0.0:
            # No meaningful decay configured
            return

        # Initialize lifetimes for any visible subject that doesn't yet have one
        for agent in subjects:
            if getattr(agent, "visible", True) and agent not in self.subject_lifetimes:
                lifetime = random.expovariate(lambda_rate)
                self.subject_lifetimes[agent] = current_time + lifetime

        # Check for expired subjects
        expired_subjects = []
        for subject, expiry_time in list(self.subject_lifetimes.items()):
            if (
                getattr(subject, "role", None) == "SUBJECT"
                and getattr(subject, "visible", True)
                and current_time >= expiry_time
            ):
                expired_subjects.append(subject)

        for subject in expired_subjects:
            subject.set_visible(False)
            # For pure decay, do not recycle snippets back into any pool
            if subject in self.subject_lifetimes:
                del self.subject_lifetimes[subject]

        if expired_subjects:
            remaining_visible = len(
                [
                    a
                    for a in self._agents
                    if getattr(a, "role", None) == "SUBJECT" and getattr(a, "visible", True)
                ]
            )
            total_subjects = len(
                [a for a in self._agents if getattr(a, "role", None) == "SUBJECT"]
            )
            print(
                f"🎲 Probabilistic decay: {len(expired_subjects)} subject(s) disappeared "
                f"(continuous-time exponential), {remaining_visible}/{total_subjects} still visible"
            )

    def initialize_dynamic_pool(self, snippet_pool: list):
        """
        Initialize the dynamic pool mode with a pool of inactive snippets.
        Called by experiment.py after spawning initial active subjects.
        """
        self.snippet_pool = list(snippet_pool)  # Copy the pool
        random.shuffle(self.snippet_pool)  # Shuffle for randomness
        
        # Schedule first appearance using exponential distribution
        self._schedule_next_appearance()
        
        # Assign lifetimes to all currently active subjects
        current_time = self._elapsed_sim_seconds()
        for agent in self._agents:
            if getattr(agent, "role", None) == "SUBJECT" and getattr(agent, "visible", True):
                lifetime = random.expovariate(1.0 / self.lifetime_mean_time)
                self.subject_lifetimes[agent] = current_time + lifetime
        
        print(f"🏊 Dynamic pool initialized: {len(self.snippet_pool)} snippets in pool, "
              f"{len(self.subject_lifetimes)} active subjects")

    def _schedule_next_appearance(self):
        """Schedule when the next snippet from the pool will appear."""
        if self.snippet_pool:
            delay = random.expovariate(1.0 / self.appearance_mean_time)
            self.next_appearance_time = self._elapsed_sim_seconds() + delay
        else:
            self.next_appearance_time = None

    def _spawn_subject_from_pool(self):
        """Spawn a new subject agent from the snippet pool at a random position."""
        if not self.snippet_pool:
            return None
        
        snippet = self.snippet_pool.pop(0)
        
        # Import here to avoid circular imports
        from subjects import SubjectAgent
        
        # Spawn the agent
        self.batch_spawn_agents(1, SubjectAgent, images=["images/villager.png"])
        
        # Find the newly spawned subject and configure it
        subjects = [a for a in self._agents if getattr(a, "role", None) == "SUBJECT"]
        if subjects:
            new_subject = subjects[-1]
            new_subject.info = snippet
            new_subject.set_visible(True)
            
            # Random position within environment bounds
            env_width = self.runtime_settings["environment"]["width"]
            env_height = self.runtime_settings["environment"]["height"]
            x = random.uniform(50, env_width - 50)
            y = random.uniform(50, env_height - 50)
            new_subject.pos.update((x, y))
            
            # Assign lifetime
            current_time = self._elapsed_sim_seconds()
            lifetime = random.expovariate(1.0 / self.lifetime_mean_time)
            self.subject_lifetimes[new_subject] = current_time + lifetime
            
            return new_subject
        return None

    def dynamic_pool_update(self):
        """
        Update the dynamic pool: check for expired subjects (disappear) and new appearances.
        Called continuously during the simulation loop.
        """
        current_time = self._elapsed_sim_seconds()
        
        # Check for subjects whose lifetime has expired (disappearances)
        expired_subjects = []
        for subject, expiry_time in list(self.subject_lifetimes.items()):
            if current_time >= expiry_time and getattr(subject, "visible", True):
                expired_subjects.append(subject)
        
        # Process disappearances
        for subject in expired_subjects:
            subject.set_visible(False)
            # Return snippet to the pool
            snippet = getattr(subject, "info", "")
            if snippet:
                self.snippet_pool.append(snippet)
            # Remove from lifetime tracking
            del self.subject_lifetimes[subject]
        
        if expired_subjects:
            active_count = len([a for a in self._agents if getattr(a, "role", None) == "SUBJECT" and getattr(a, "visible", True)])
            print(f"👋 {len(expired_subjects)} subject(s) disappeared (lifetime expired). "
                  f"Pool: {len(self.snippet_pool)}, Active: {active_count}")
        
        # Check for new appearances
        if self.next_appearance_time is not None and current_time >= self.next_appearance_time:
            new_subject = self._spawn_subject_from_pool()
            if new_subject:
                active_count = len([a for a in self._agents if getattr(a, "role", None) == "SUBJECT" and getattr(a, "visible", True)])
                print(f"✨ New subject appeared from pool. Pool: {len(self.snippet_pool)}, Active: {active_count}")
            
            # Schedule next appearance
            self._schedule_next_appearance()
    
    def save_experiment_data(self, plot: LivePlot):
        """Save experiment data and plot to the experiments directory"""
        if self._experiment_saved:
            print("ℹ️ Experiment data already saved; skipping duplicate save.")
            return
        try:
            # Generate logical timestamps: [2, 4, 6, ..., 120] based on snapshot interval
            interval = self.snapshot_interval_seconds
            timestamps = [int((i + 1) * interval) for i in range(self.num_snapshots)]
            
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

            profile_dir = base_dir / profile_key
            profile_dir.mkdir(parents=True, exist_ok=True)

            existing_runs = [p for p in profile_dir.iterdir() if p.is_dir() and p.name.startswith("run_")]
            next_idx = 1
            if existing_runs:
                try:
                    next_idx = max(int(p.name.split("_")[-1]) for p in existing_runs) + 1
                except ValueError:
                    next_idx = 1

            run_dir = profile_dir / f"run_{next_idx:04d}"
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
                "information_teleportation": {
                    "enabled": self.teleportation_enabled,
                    "mode": self.teleportation_mode,
                    "visibility_probability": self.subject_visibility_probability,
                    "decay_count": self.decay_count,
                    "decay_probability": self.decay_probability,
                    "interval_seconds": self.visibility_interval,
                    "initial_active_count": self.initial_active_count,
                    "appearance_mean_time": self.appearance_mean_time,
                    "lifetime_mean_time": self.lifetime_mean_time,
                } if self.teleportation_enabled else {"enabled": False},
            }
 
            # Save JSON
            json_path = run_dir / "experiment.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            metadata_path = run_dir / "metadata.json"
            with open(metadata_path, "w", encoding="utf-8") as meta_file:
                json.dump(metadata, meta_file, ensure_ascii=False, indent=2)

            # Save live plot image (if available)
            if plot is not None:
                img_path = run_dir / "live_plot.png"
                try:
                    plot.save(img_path)
                except Exception as e_img:
                    print(f"⚠️ Failed to save live plot image: {e_img}")
            
            # Generate and save aesthetic score plot
            self._save_score_plot(data, run_dir, swarm_type)
 
            print(f"💾 Saved experiment to {run_dir}")
            self._experiment_saved = True
        except Exception as e:
            print(f"⚠️ Failed to save experiment data: {e}")

    def _save_score_plot(self, data: dict, run_dir: Path, swarm_type: str):
        """Generate and save an aesthetic plot of average score over time."""
        try:
            # Extract data
            timestamps = data["timestamps"]
            agents_data = data["agents"]
            num_agents = len(agents_data)
            
            # Collect all scores and compute mean
            all_scores = []
            for agent_id, agent_data in agents_data.items():
                scores = [agent_data["scores"].get(str(t), 0.0) for t in timestamps]
                all_scores.append(scores)
            
            all_scores = np.array(all_scores)
            mean_scores = np.mean(all_scores, axis=0)
            
            # Set up the figure with a dark, modern aesthetic
            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=(12, 6), facecolor='#0d1117')
            ax.set_facecolor('#0d1117')
            
            # Plot mean score line with gradient fill
            ax.fill_between(timestamps, 0, mean_scores, 
                           color='#58a6ff', alpha=0.15)
            ax.plot(timestamps, mean_scores, 
                   color='#58a6ff', 
                   linewidth=2.5)
            
            # Add markers at key points
            ax.scatter([timestamps[-1]], [mean_scores[-1]], 
                      color='#58a6ff', s=80, zorder=10, edgecolors='white', linewidths=1.5)
            
            # Final score annotation
            final_mean = mean_scores[-1] if len(mean_scores) > 0 else 0
            ax.annotate(f'{final_mean:.3f}', 
                       xy=(timestamps[-1], final_mean),
                       xytext=(15, 0), textcoords='offset points',
                       fontsize=14, fontweight='bold', color='#58a6ff',
                       va='center')
            
            # Styling
            ax.set_xlabel('Time (seconds)', fontsize=13, color='#c9d1d9', labelpad=10)
            ax.set_ylabel('Average Similarity Score', fontsize=13, color='#c9d1d9', labelpad=10)
            
            # Title
            learning_type = "Social Learning" if self.social_learning_enabled else "Self Learning"
            title = f'{learning_type} — Average Score Over Time'
            subtitle = f'{num_agents} agents • {self.ground_truth_key or "scenario"}'
            
            ax.set_title(title, fontsize=16, color='#ffffff', fontweight='bold', pad=15)
            ax.text(0.5, 1.02, subtitle, transform=ax.transAxes, 
                   fontsize=10, color='#8b949e', ha='center', va='bottom')
            
            # Grid
            ax.grid(True, alpha=0.15, color='#30363d', linestyle='-', linewidth=0.5)
            ax.set_axisbelow(True)
            
            # Axis limits
            ax.set_xlim(timestamps[0], timestamps[-1] + 10)
            ax.set_ylim(0, 1.0)
            
            # Spine styling
            for spine in ax.spines.values():
                spine.set_visible(False)
            
            # Tick styling
            ax.tick_params(colors='#8b949e', labelsize=10)
            
            plt.tight_layout()
            
            # Save
            plot_path = run_dir / "scores_over_time.png"
            plt.savefig(plot_path, dpi=150, facecolor='#0d1117', edgecolor='none', bbox_inches='tight')
            plt.close(fig)
            
            print(f"📊 Saved score plot to {plot_path}")
            
        except Exception as e:
            print(f"⚠️ Failed to generate score plot: {e}")
                
                