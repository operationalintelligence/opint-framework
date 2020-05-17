import pandas as pd

def checkisHPC(frame, mergeditems):
    for issue in mergeditems:
        issue.filterByIssue(frame)
        frameNotNull = issue.filterByIssue(frame).loc[frame['SITE'] != "Not specified"]
        frameNull = issue.filterByIssue(frame).loc[frame['SITE'] == "Not specified"]
        if len(frameNotNull.index) > 0:
            issue.isHPC = True
        if len(frameNull.index) > 0:
            issue.isGRID = True

def mergedicts(dict1, dict2):
    outdict = {}
    unionKeys = set(dict1.keys()).union(dict2.keys())
    for errfield in unionKeys:
        outdict[errfield] = {key:(dict1.get(errfield, {}).get(key, 0)+dict2.get(errfield, {}).get(key, 0)) for key in unionKeys}
    return outdict
