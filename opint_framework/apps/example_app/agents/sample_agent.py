from opint_framework.core.prototypes.BaseAgent import BaseAgent
import threading
import time


class SampleAgent(BaseAgent):
    lock = threading.RLock()

    def init(self):
        pass

    def processCycle(self):
        print("Hello World from Sample Agent")
        time.sleep(10)
