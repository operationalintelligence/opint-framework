from rest_framework.response import Response
from opint_framework.apps.workload_jobsbuster.models import WorkflowIssue, WorkflowIssueMetadata, WorkflowIssueTicks
from rest_framework.decorators import api_view
import datetime
import os, json
from dateutil.parser import parse
from django.db.models import Q
from opint_framework.apps.workload_jobsbuster.api.views.IssueClass import Issue, IssueObservation
from opint_framework.core.utils.common import freeze

import matplotlib as mpl
import matplotlib.cm as cm
from django.http import JsonResponse
from django.views.decorators.cache import never_cache
import matplotlib.pyplot as plt
import numpy as np

counter = 0
chunksize = 50

OI_DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%S"

"""
API endpoint that allows SampleModel to be viewed or edited.
"""
@never_cache
@api_view(['GET', 'POST'])
def processTimeWindowData(request):
    if 'timewindow' in request.query_params:
        timewindow = request.query_params['timewindow'].split('|')
        datefrom = datetime.datetime.strptime(timewindow[0], OI_DATETIME_FORMAT)
        dateto = datetime.datetime.strptime(timewindow[1], OI_DATETIME_FORMAT)
    else:
        dateto = datetime.datetime.utcnow()
        datefrom = datetime.datetime.utcnow() - datetime.timedelta(hours=12)
    topN = int(request.query_params['topn']) if 'topn' in request.query_params else 20
    metric = (request.query_params['metric']) if 'metric' in request.query_params else 'loss'

    ret = getIssuesWithMets(datefrom, dateto, topN=topN, metric=metric)
    ret = addColorsAndNames(ret)
    ticks, mesuresW, mesuresNF, colorsNF, colorsW = getHistogramData(ret)
    return JsonResponse({"Result":"OK", "issues": serialize(ret), "ticks":ticks, "mesuresW":mesuresW, "mesuresNF":mesuresNF,
                     "colorsNF":colorsNF, "colorsW":colorsW})


def serialize(issues):
    setIssues = []
    for issue in issues:
        issueDict = issue.__dict__
        del issueDict['observations']
        setIssues.append(issueDict)
    return setIssues


def getIssuesWithMets(datefrom, dateto, topN, metric):
    query = Q(Q(issue_id_fk__observation_started__lt=dateto) & Q(issue_id_fk__observation_finished__gt=datefrom))
    issuesRaw = WorkflowIssueMetadata.objects.using('jobs_buster_persistency').select_related('issue_id_fk').filter(query)
    issues = fillIssuesList(issuesRaw)
    issues = addObservations(issues, query)
    issues = mergeIssues(issues)
    issues = getTopNIsses(issues, topN=topN, metric=metric)
    return issues


def getTopNIsses(issues, topN = 10, metric='sumWLoss'):
    if metric == 'loss':
        return sorted(issues, key=lambda x: x.walltime_loss, reverse=True)[:topN]
    if metric == 'fails':
        return sorted(issues, key=lambda x: x.nFailed_jobs, reverse=True)[:topN]


def mergeIssues(issues):
    norepeatIssues = {}
    for issueid, issue in issues.items():
        norepIssue = norepeatIssues.get(freeze(issue.features), None)
        if norepIssue:
            norepeatIssues[freeze(issue.features)] = norepIssue.merge(issue)
        else:
            norepeatIssues[freeze(issue.features)] = issue
    return norepeatIssues.values()


def fillIssuesList(issuesRaw):
    issues = {}
    for issueRaw in issuesRaw:
        issue = issues.setdefault(issueRaw.issue_id_fk.issue_id, Issue())
        issue.features[issueRaw.key] = issueRaw.value
        issue.issueID = issueRaw.issue_id_fk.issue_id
    return issues


def addObservations(issues, query):
    issuesToProcess = list(issues.keys())
    observations = runDBObservationQuery(query)
    for issueID in issuesToProcess:
        if issueID in observations:
            issues[issueID].observations = observations[issueID]
    return issues


def runDBObservationQuery(query):
    observations = {}
    observationsRows = WorkflowIssueTicks.objects.using('jobs_buster_persistency').select_related('issue_id_fk').filter(query)
    for observation in observationsRows:
        issueID = observation.issue_id_fk.issue_id
        issueObservation = IssueObservation()
        issueObservation.walltime_loss = observation.walltime_loss
        issueObservation.nfailed_jobs = observation.nFailed_jobs
        issueObservation.tick_time = observation.tick_time
        observations.setdefault(issueID, []).append(issueObservation)
    return observations


def addColorsAndNames(issues):
    if len(issues) == 0:
        return issues
    cmap = plt.get_cmap('tab20c')
    colors = cmap(np.linspace(0, 1, len(issues)+1))
    for (issue,color) in zip(issues, colors):
        issue.rgbaW = list(color)
    for index, issue in enumerate(issues):
        issue.rgbaNF = list(colors[index])
        issue.name = 'N_'+ str(index)
    return issues


def getHistogramData(issues):
    issuesIDsFiltered = [i.issueID for i in issues]
    issuesNamesFiltered = {i.issueID:i.name for i in issues}
    colorsFilteredNF = {i.issueID:i.rgbaNF for i in issues}
    colorsFilteredW = {i.issueID:i.rgbaW for i in issues}

    mesuresW = {}
    mesuresNF = {}

    for issue in issues:
        for tick in issue.observations:
            entryW = mesuresW.setdefault(tick.tick_time, {})
            entryW[issue.issueID] = tick.walltime_loss
            entryNF = mesuresNF.setdefault(tick.tick_time, {})
            entryNF[issue.issueID] = tick.nfailed_jobs

    ticks = list(mesuresW.keys())
    ticks.sort()
    mesuresWTransponed = {}
    mesuresNFTransponed = {}
    for tick in ticks:
        for issueID in issuesIDsFiltered:
            mesuresWTransponed.setdefault(issueID,[mesuresW[tick].get(issueID, 0)]).append(mesuresW[tick].get(issueID, 0))
            mesuresNFTransponed.setdefault(issueID,[mesuresNF[tick].get(issueID, 0)]).append(mesuresNF[tick].get(issueID, 0))

    histArrayW = []
    histArrayNF = []
    colorsNF = []
    colorsW = []
    for issueID in list(mesuresWTransponed.keys()):
        histArrayW.extend([[issuesNamesFiltered[issueID]] +mesuresWTransponed[issueID]])
        histArrayNF.extend([[issuesNamesFiltered[issueID]] +mesuresNFTransponed[issueID]])
        colorsNF.append(colorsFilteredNF[issueID])
        colorsW.append(colorsFilteredW[issueID])

    return ticks, histArrayW, histArrayNF, colorsNF, colorsW
