class IssuesMapper:

    def __init__(self):
        self.rawcollection = {}
        self.joinedissues = []

    def addMetaData(self, metaDataObj):
        issue = metaDataObj.issue_id_fk
        issueObj = self.rawcollection.get(issue.issue_id, {})
        issueObj['issueID'] = issue.issue_id
        issueObj['observation_started'] = issue.observation_started
        issueObj['observation_finished'] = issue.observation_finished
        issueObj['walltime_loss'] = issue.walltime_loss
        issueObj['nFailed_jobs'] = issue.nFailed_jobs
        issueObj['nSuccess_jobs'] = issue.nSuccess_jobs
        issueObj.setdefault("features", {})[metaDataObj.key] = metaDataObj.value
        self.rawcollection[issue.issue_id] = issueObj

    def joinSimilarIssues(self):
        norepeatIssues = {}
        for issue, descriptor in self.rawcollection.items():
            norepeatIssues.setdefault(self.freeze(descriptor['features']), []).append(issue)

        for key, issuesIDs in norepeatIssues.items():
            features = self.rawcollection[issuesIDs[0]]['features']
            evidences = []
            sumWLoss = 0
            sumJFails = 0
            sumJSucc = 0
            for issue in issuesIDs:
                record = {
                    "observation_started": self.rawcollection[issue]['observation_started'],
                    "observation_finished": self.rawcollection[issue]['observation_finished'],
                    "walltime_loss": self.rawcollection[issue]['walltime_loss'],
                    "nFailed_jobs": self.rawcollection[issue]['nFailed_jobs'],
                    "nSuccess_jobs": self.rawcollection[issue]['nSuccess_jobs'],
                }
                sumWLoss += self.rawcollection[issue]['walltime_loss']
                sumJFails += self.rawcollection[issue]['nFailed_jobs']
                sumJSucc += self.rawcollection[issue]['nSuccess_jobs']
                evidences.append(record)
            self.joinedissues.append({
                "features": features,
                "evidences": evidences,
                "sumWLoss": sumWLoss,
                "sumJFails": sumJFails,
                "sumJSucc": sumJSucc,
                "id": issuesIDs[0],
            })

    def getTopNIsses(self, topN = 10, metric='sumWLoss'):
        if len(self.joinedissues) == 0 and len(self.rawcollection) > 0:
            self.joinSimilarIssues()
        return sorted(self.joinedissues, key=lambda x: x[metric], reverse=True)[:topN]



    def freeze(self,d):
        if isinstance(d, dict):
            return frozenset((key, self.freeze(value)) for key, value in d.items())
        elif isinstance(d, list):
            return tuple(self.freeze(value) for value in d)
        return d
