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
    for errfield in set(dict1.keys()).union(dict2.keys()):
        messages = set(dict1.get(errfield, {}).keys()).union(dict2.get(errfield, {}).keys())
        outdict[errfield] = {message: dict1.get(errfield, {}).get(message, 0)+dict2.get(errfield, {}).get(message, 0) for message in messages}
    return outdict

