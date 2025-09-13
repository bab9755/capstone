import time
class Context:
    def __init__(self, agent):
        self.agent = agent
        self.content = {}

    def add_context(self):
        self.content["time"] = time.time()
        self.content["location"] = self.agent.pos
