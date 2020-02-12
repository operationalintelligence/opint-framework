#
# This is a Base Class for agents implementation
#


class BaseAgent(object):
    def init(self):
        raise NotImplementedError("Must override")

    def processCycle(self):
        raise NotImplementedError("Must override")

    def execute(self):
        if self.lock.acquire(blocking=False):
            try:
                self.processCycle()
            finally:
                self.lock.release()
