import copy, json
from opint_framework.core.utils.common import freeze

class Issue:
    def __init__(self):
        self.issueID = None
        self.observation_started = None
        self.observation_finished  = None
        self.walltime_loss = 0
        self.nFailed_jobs = 0
        self.observations = []
        self.features = {}
        self.rgbaW = None
        self.name = None
        self.rgbaNF = None


    def merge(self, otherIssue):
        if freeze(self.features) != freeze(otherIssue.features):
            raise AssertionError("Issue objects are not equal")

        if self.issueID > otherIssue.issueID:
            latest = self
            earliest = otherIssue
        else:
            latest = otherIssue
            earliest = self

        mergedObservation = []
        latestObservations = {observation.tick_time: observation for observation in latest.observations}
        earliestObservation = {observation.tick_time: observation for observation in earliest.observations}
        allticks = set(list(latestObservations.keys()) + list(earliestObservation.keys()))

        for tick in allticks:
            mergedObservation.append(latestObservations.get(tick, earliestObservation.get(tick, None)))

        self.observations = copy.deepcopy(mergedObservation)
        self.observation_started = earliest.observation_started
        self.observation_finished = latest.observation_finished
        self.issueID = latest.issueID
        self.recalculateMetrics()
        return self


    def recalculateMetrics(self):
        self.nFailed_jobs = 0
        self.walltime_loss = 0
        self.observations = sorted(self.observations, key=lambda x: x.tick_time, reverse=False)

        for observation in self.observations:
            self.nFailed_jobs += observation.nfailed_jobs
            self.walltime_loss += observation.walltime_loss
        try:
            self.observation_started = self.observations[0].tick_time
            self.observation_finished = self.observations[-1].tick_time
        except:
            pass


class IssueObservation:
    def __init__(self):
        self.tick_time = None
        self.walltime_loss = 0
        self.nfailed_jobs = 0
