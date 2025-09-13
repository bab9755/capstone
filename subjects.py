class Subject:
    def __init__(sel, information: str, position: tuple):
        self.information = information
        self.position = position

    def get_information(self):
        return self.information

    def get_position(self):
        return self.position
    
    def get_subject(self):
        return self.subject