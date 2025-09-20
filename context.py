import time
from collections import defaultdict
class Context:
    def __init__(self, agent):
        self.agent = agent
        self.content = defaultdict(list)
        self.content["snippets"].append("This is the initial context of agent " + str(self.agent.id))

    def add_context(self):
        self.content["time"] = time.time()
        self.content["location"] = self.agent.pos
