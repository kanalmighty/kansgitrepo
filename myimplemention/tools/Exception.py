
class Exception(Exception):
    def __init__(self,name, message):
        self.name = name
        self.message = message