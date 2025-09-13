class CameraSensor:
    def __init__(self, agent):
        self.agent = agent

    def take_photo(self):
        image = "background source"
        image.crop(self.agent.pos.x, self.agent.pos.y, self.agent.radius, self.agent.radius)
        return image