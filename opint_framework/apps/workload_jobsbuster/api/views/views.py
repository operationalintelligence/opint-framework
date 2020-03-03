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

counter = 0

"""
API endpoint that allows SampleModel to be viewed or edited.
"""
@api_view(['GET', 'POST'])
def processTimeWindowData(request):
    if 'timewindow' in request.query_params:
        timewindow = request.query_params['timewindow']
        timewindow = json.loads(timewindow)
        datefrom = timewindow['startdate']
        dateto = timewindow['enddate']
    else:
        dateto = datetime.datetime.utcnow()
        datefrom = datetime.datetime.utcnow() - datetime.timedelta(hours=12)
    topN = int(request.query_params['topn']) if 'topn' in request.query_params else 10

    #datefrom = parse('28-FEB-20 23:00:00')
    #dateto = parse('05-MAR-20 00:00:00')
    query = Q(~Q(issue_id_fk__observation_started__gt=dateto) & ~Q(issue_id_fk__observation_finished__lt=datefrom))
    ret = getIssuesWithMets(query, topN=topN)
    ret = addColorsAndNames(ret)
    ticks, mesuresW, mesuresNF, colorsNF, colorsW = getHistogramData(ret, query)
    return Response({"Result":"OK", "issues": ret, "ticks":ticks, "mesuresW":mesuresW, "mesuresNF":mesuresNF,
                     "colorsNF":colorsNF, "colorsW":colorsW})


def getIssuesWithMets(query, topN):
    issues = WorkflowIssueMetadata.objects.using('jobs_buster_persistency').select_related('issue_id_fk').filter(query)
    issuesMapper = IssuesMapper()
    for issue in issues:
        issuesMapper.addMetaData(issue)
    issuesMapper.joinSimilarIssues()
    return issuesMapper.getTopNIsses(topN=topN, metric='sumJFails')


def addColorsAndNames(issues):
    if len(issues) == 0:
        return issues
    normWall = mpl.colors.Normalize(vmin=issues[-1]['sumWLoss'], vmax=issues[0]['sumWLoss'])
    cmapW = cm.hot
    mW = cm.ScalarMappable(norm=normWall, cmap=cmapW)

    for issue in issues:
        rgba = mW.to_rgba(issue['sumWLoss'])
        issue['rgbaW'] = rgba

    normN = mpl.colors.Normalize(vmin=issues[-1]['sumJFails'], vmax=issues[0]['sumJFails'])
    cmapN = cm.hot
    mN = cm.ScalarMappable(norm=normN, cmap=cmapN)

    for index, issue in enumerate(issues):
        rgba = mN.to_rgba(issue['sumJFails'])
        issue['rgbaNF'] = rgba
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
