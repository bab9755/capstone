import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import threading
import time
import matplotlib.cm as cm
from helpers import load_config
config = load_config()
metric = config.get("metric", "cosine-bert")
class LivePlot:
    def __init__(self, title=f"{metric} Score"):
        plt.ion()
        self.fig, self.ax = plt.subplots()
        # Store data and lines for each agent
        self.agent_data = {}  # agent_id -> (x_data, y_data)
        self.agent_lines = {}  # agent_id -> line object
        self.colors = cm.get_cmap('tab10')  # Color map for different agents
        self.ax.set_xlabel("Time (ms)")
        self.ax.set_ylabel(metric)
        self.ax.set_title(title)
        
        # Add grid to the plot
        self.ax.grid(True, alpha=0.3, linestyle='--')
        
        # Set y-axis tick spacing to be larger (every 0.1 units)
        self.ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
        
        self.lock = threading.Lock()


    def update(self, x, y, agent_id=None):
        with self.lock:
            # Use agent_id or a default identifier
            if agent_id is None:
                agent_id = "default"
            
            # Initialize data for new agent
            if agent_id not in self.agent_data:
                self.agent_data[agent_id] = ([], [])
                # Get a unique color for this agent
                color_idx = len(self.agent_lines) % 10
                color = self.colors(color_idx)
                label = f"Agent {agent_id}"
                # Use line with circle markers for better visibility
                line, = self.ax.plot([], [], "-o", color=color, label=label, markersize=4, linewidth=1.5)
                self.agent_lines[agent_id] = line
                # Update legend when a new agent is added
                self.ax.legend()
            
            # Append new data point
            x_data, y_data = self.agent_data[agent_id]
            x_data.append(x)
            y_data.append(y)
            
            # Update the line for this agent
            line = self.agent_lines[agent_id]
            line.set_data(x_data, y_data)
            
            # Update axis limits based on all agents
            self.ax.relim()
            self.ax.autoscale_view()
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.001)

    def save(self, filepath):
        with self.lock:
            try:
                self.fig.savefig(filepath, dpi=200, bbox_inches="tight")
            except Exception as e:
                print(f"⚠️ Failed to save plot image to {filepath}: {e}")
