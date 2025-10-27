import sqlite3
import json
from typing import Optional, Dict, Any, List, Tuple
from contextlib import contextmanager
import os
from datetime import datetime


class Database:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._create_table()# creates tables at startup


    @contextmanager
    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()


    def _create_table(self):
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                        experiment_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT,
                        description TEXT,
                        start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        end_time TIMESTAMP,
                        simulation_seed INTEGER,
                        llm_provider TEXT,
                        llm_model TEXT,
                        num_agents INTEGER,
                        num_fragments INTEGER,
                        summary_interval_ms INTEGER DEFAULT 20000,
                        config_json TEXT
                    )
                
                """)

                cursor.execute("""
                CREATE TABLE IF NOT EXISTS agents (
                        agent_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        experiment_id INTEGER NOT NULL,
                        agent_sim_id INTEGER NOT NULL,
                        role TEXT NOT NULL,
                        spawn_time INTEGER,
                        FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
                    )
                
                """)


                cursor.execute("""

                    CREATE TABLE IF NOT EXISTS summaries (
                        summary_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        agent_id INTEGER NOT NULL,
                        experiment_id INTEGER NOT NULL,
                        simulation_time_ms INTEGER NOT NULL,
                        interval_number INTEGER,
                        summary_text TEXT NOT NULL,
                        summary_length INTEGER NOT NULL,
                        p_content TEXT,  -- Content from subjects (JSON)
                        t_received_content TEXT,  -- Content from agents (JSON)
                        llm_response_time_ms REAL,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (agent_id) REFERENCES agents(agent_id),
                        FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)

                    )
                """)


                cursor.execute(""" 
                
                    CREATE TABLE IF NOT EXISTS similarity_scores (
                        score_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        summary_id INTEGER NOT NULL,
                        metric_name TEXT NOT NULL,
                        metric_value REAL NOT NULL,
                        comparison_target TEXT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (summary_id) REFERENCES summaries(summary_id)
                    )
                """)
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS agent_interactions (
                    interaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id INTEGER NOT NULL,
                    agent1_sim_id INTEGER NOT NULL,
                    agent2_sim_id INTEGER NOT NULL,
                    interaction_type TEXT NOT NULL,
                    simulation_time_ms INTEGER NOT NULL,
                    position_x REAL,
                    position_y REAL,
                    information_exchanged TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
                )
            """)
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS ground_truth (
                    ground_truth_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id INTEGER NOT NULL,
                    fragment_index INTEGER NOT NULL,
                    fragment_text TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
                )
            """)

                conn.commit()

        except sqlite3.Error as e:
            print(f"Error creating tables: {e}")
            conn.rollback()
            raise

    def create_experiment(self, experiment_name: str, experiment_description: str, simulation_seed: int, llm_provider: str, llm_model: str, num_agents: int, num_fragments: int, summary_interval_ms: int, config_json: str):
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                INSERT INTO experiments (name, description, simulation_seed, llm_provider, llm_model, num_agents, num_fragments, summary_interval_ms, config_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (experiment_name, experiment_description, simulation_seed, llm_provider, llm_model, num_agents, num_fragments, summary_interval_ms, config_json))
                conn.commit()
                return cursor.lastrowid # returns the experiment_id
        except sqlite3.Error as e:
            print(f"Error creating experiment: {e}")
            conn.rollback()
            raise

    def add_agent(self, experiment_id: int, agent_sim_id: int, role: str, spawn_time: int):

        try:    
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(""" 
                INSERT INTO agents(experiment_id, agent_sim_id, role, spawn_time)
                VALUES (?, ?, ?, ?)
                """, (experiment_id, agent_sim_id, role, spawn_time))
                conn.commit()
                reutrn conn.lastrowid # returns the agent_id

        except sqlite3.Error as e:
            print(f"Error creating agent: {e}")
            conn.rollback()
            raise

            
    def add_summary(self, agent_id: int, experiment_id: int, simulation_time_ms: int, interval_number: int, summary_text: str, summary_length: int, p_content: str, t_received_content: str, llm_response_time_ms: float):
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                INSERT INTO summaries(agent_id, experiment_id, simulation_time_ms, interval_number, summary_text, summary_length, p_content, t_received_content, llm_response_time_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (agent_id, experiment_id, simulation_time_ms, interval_number, summary_text, summary_length, p_content, t_received_content, llm_response_time_ms))
                conn.commit()
                return cursor.lastrowid # returns the summary_id
        except sqlite3.Error as e:
            print(f"Error adding summary: {e}")
            conn.rollback()
            raise

    def add_interaction(self, experiment_id: int, agent1_sim_id: int, agent2_sim_id: int, interaction_type: str, simulation_time_ms: int, position_x: float, position_y: float, information_exchanged: str):
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                INSERT INTO agent_interactions(experiment_id, agent1_sim_id, agent2_sim_id, interaction_type, simulation_time_ms, position_x, position_y, information_exchanged)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (experiment_id, agent1_sim_id, agent2_sim_id, interaction_type, simulation_time_ms, position_x, position_y, information_exchanged))
                conn.commit()
                return cursor.lastrowid # returns the interaction_id
        except sqlite3.Error as e:
            print(f"Error adding interaction: {e}")
            conn.rollback()
            raise
        
    def add_ground_truth(self, experiment_id: int, fragment_index: int, fragment_text: str):
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                INSERT INTO ground_truth(experiment_id, fragment_index, fragment_text)
                VALUES (?, ?, ?)
                """, (experiment_id, fragment_index, fragment_text))
                conn.commit()
                return cursor.lastrowid # returns the ground_truth_id
        except sqlite3.Error as e:
            print(f"Error adding ground truth: {e}")
            conn.rollback()
            raise
        
    def get_agent_id(self, experiment_id: int, agent_sim_id: int):
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                SELECT agent_id FROM agents WHERE experiment_id = ? AND agent_sim_id = ?
                """, (experiment_id, agent_sim_id))
                return cursor.fetchone()[0] # returns the agent_id
        except sqlite3.Error as e:
            print(f"Error getting agent id: {e}")
            conn.rollback()
            raise
        
    def get_summary_id(self, agent_id: int, experiment_id: int, simulation_time_ms: int, interval_number: int):
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                SELECT summary_id FROM summaries WHERE agent_id = ? AND experiment_id = ? AND simulation_time_ms = ? AND interval_number = ?
                """, (agent_id, experiment_id, simulation_time_ms, interval_number))
                return cursor.fetchone()[0] # returns the summary_id
        except sqlite3.Error as e:
            print(f"Error getting summary id: {e}")
            conn.rollback()
            raise


    def end_experiment(self, experiment_id: int):
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                UPDATE experiments SET end_time = CURRENT_TIMESTAMP WHERE experiment_id = ?
                """, (experiment_id,))
                conn.commit()
        except sqlite3.Error as e:
            print(f"Error ending experiment: {e}")
            conn.rollback()
            raise
        
    def get_experiment_id(self, experiment_name: str):
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                SELECT experiment_id FROM experiments WHERE name = ?
                """, (experiment_name,))
                return cursor.fetchone()[0] # returns the experiment_id
        except sqlite3.Error as e:
            print(f"Error getting experiment id: {e}")
            conn.rollback()
            raise

    



    

    