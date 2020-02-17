from opint_framework.core.prototypes.BaseAgent import BaseAgent
import threading


class SampleAgent(BaseAgent):
    lock = threading.RLock()

    def init(self):
        pass

    def processCycle(self):
        print("Hello World from Sample Agent")
