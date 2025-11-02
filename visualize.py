import matplotlib.pyplot as plt
import threading
import time
import matplotlib.cm as cm

class LivePlot:
    def __init__(self, title="BertScore Evolution"):
        plt.ion()
        self.fig, self.ax = plt.subplots()
        # Store data and lines for each agent
        self.agent_data = {}  # agent_id -> (x_data, y_data)
        self.agent_lines = {}  # agent_id -> line object
        self.colors = cm.get_cmap('tab10')  # Color map for different agents
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("BertScore")
        self.ax.set_title(title)
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
                line, = self.ax.plot([], [], "-", color=color, label=label)
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
