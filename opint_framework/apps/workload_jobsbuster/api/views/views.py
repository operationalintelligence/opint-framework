from rest_framework.response import Response
from opint_framework.apps.workload_jobsbuster.models import WorkflowIssue, WorkflowIssueMetadata, WorkflowIssueTicks
from rest_framework.decorators import api_view
import datetime
import os, json
from dateutil.parser import parse
from django.db.models import Q
from opint_framework.apps.workload_jobsbuster.api.views.IssuesMapper import IssuesMapper
import matplotlib as mpl
import matplotlib.cm as cm
from django.http import JsonResponse
from django.views.decorators.cache import never_cache
import matplotlib.pyplot as plt
import numpy as np

counter = 0
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
    if metric == 'loss':
        metric = 'sumWLoss'
    elif metric == 'fails':
        metric = 'sumJFails'

    #datefrom = parse('28-FEB-20 23:00:00')
    #dateto = parse('05-MAR-20 00:00:00')
    ret = getIssuesWithMets(datefrom, dateto, topN=topN, metric = metric)
    ret = addColorsAndNames(ret)
    query = Q(Q(issue_id_fk__observation_started__lt=dateto) & Q(issue_id_fk__observation_finished__gt=datefrom))
    ticks, mesuresW, mesuresNF, colorsNF, colorsW = getHistogramData(ret, query)
    return JsonResponse({"Result":"OK", "issues": ret, "ticks":ticks, "mesuresW":mesuresW, "mesuresNF":mesuresNF,
                     "colorsNF":colorsNF, "colorsW":colorsW})


def getIssuesWithMets(datefrom, dateto, topN, metric):
    query = Q(Q(issue_id_fk__observation_started__lt=dateto) & Q(issue_id_fk__observation_finished__gt=datefrom))
    issues = WorkflowIssueMetadata.objects.using('jobs_buster_persistency').select_related('issue_id_fk').filter(query)
    queryEvidences = query & Q(Q(tick_time__lt=dateto) & Q(tick_time__gt=datefrom))
    evidencesRows = WorkflowIssueTicks.objects.using('jobs_buster_persistency').select_related('issue_id_fk').filter(queryEvidences)
    evidencesW = {}
    evidencesNF = {}

    for evidence in evidencesRows:
        val = evidencesW.get(evidence.issue_id_fk.issue_id, 0)
        val += evidence.walltime_loss
        evidencesW[evidence.issue_id_fk.issue_id] = val
        val = evidencesNF.get(evidence.issue_id_fk.issue_id, 0)
        val += evidence.nFailed_jobs
        evidencesNF[evidence.issue_id_fk.issue_id] = val

    for issue in issues:
        issue.issue_id_fk.nFailed_jobs = evidencesNF.get(issue.issue_id_fk.issue_id, 0)
        issue.issue_id_fk.walltime_loss = evidencesW.get(issue.issue_id_fk.issue_id, 0)

    issuesMapper = IssuesMapper()
    for issue in issues:
        issuesMapper.addMetaData(issue)
    return issuesMapper.getTopNIsses(topN=topN, metric=metric)


def addColorsAndNames(issues):
    if len(issues) == 0:
        return issues
    cmap = plt.get_cmap('tab20c')
    colors = cmap(np.linspace(0, 1, len(issues)+1))

    for (issue,color) in zip(issues, colors):
        issue['rgbaW'] = list(color)

    for index, issue in enumerate(issues):
        issue['rgbaNF'] = list(colors[index])
        issue['name'] = 'N_'+ str(index)
    return issues


def getHistogramData(issues, query):
    histogramTicks = WorkflowIssueTicks.objects.using('jobs_buster_persistency').select_related('issue_id_fk').filter(query)
    issuesIDsFiltered = [i['id'] for i in issues]
    issuesNamesFiltered = {i['id']:i['name'] for i in issues}
    colorsFilteredNF = {i['id']:i['rgbaNF'] for i in issues}
    colorsFilteredW = {i['id']:i['rgbaW'] for i in issues}

    histogramTicks = filter(lambda x: x.issue_id_fk.issue_id in issuesIDsFiltered, histogramTicks)
    mesuresW = {}
    mesuresNF = {}
    for histogramTick in histogramTicks:
        entryW = mesuresW.setdefault(histogramTick.tick_time, {})
        entryW[histogramTick.issue_id_fk.issue_id] = histogramTick.walltime_loss
        entryNF = mesuresNF.setdefault(histogramTick.tick_time, {})
        entryNF[histogramTick.issue_id_fk.issue_id] = histogramTick.nFailed_jobs
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
