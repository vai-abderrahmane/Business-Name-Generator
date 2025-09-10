from .regex_filter import is_blocked

class SafetyPipeline:
    def __init__(self):
        pass
    def check(self, text: str):
        return { 'blocked': is_blocked(text) }
