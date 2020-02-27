from rest_framework import viewsets
import sys, traceback
from rest_framework.response import Response
import opint_framework.apps.workload_jobsbuster.api.pandaDataImporter as dataImporter
from opint_framework.apps.workload_jobsbuster.models import WorkflowIssue

import pickle
import opint_framework.apps.workload_jobsbuster.conf.settings as settings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor, Pool, EFstrType
import tempfile
from rest_framework.decorators import api_view
import json
from datetime import datetime
import os

counter = 0

"""
API endpoint that allows SampleModel to be viewed or edited.
"""
@api_view(['GET'])
def processTimeWindowData(request):
    global counter
    os.chdir('/opt/oracle')

    timewindow = request.query_params['timewindow']
    timewindow = json.loads(timewindow)
    datefrom = timewindow['startdate']
    dateto = timewindow['enddate']

    pandasdf = dataImporter.retreiveData(datefrom, dateto)
    pickle.dump(pandasdf, open(settings.datafilespath + "rawdataframe0_daily1.sr", 'wb+'))
    pickle.load(open(settings.datafilespath + "rawdataframe0_daily1.sr", 'rb'))
    preprocessedFrame = preprocessRawData(pandasdf)

    listOfProblems = []
    classifyIssue(preprocessedFrame, listOfProblems, None)
    listOfProblems = sorted(listOfProblems, key=lambda i: i.lostWallTime, reverse=True)

    for item in listOfProblems:
        # issue = WorkflowIssue(observation_started=datetime.now(), observation_finished=datetime.now(),
        #                       walltime_loss=100, failures_counts=100)
        # issue.save(using='jobs_buster_persistency')

        # print(item.features)
        # print(item.lostWallTime / 3600.0 / 24.0 / 365.0)
        # print('\n')
        pass

    return Response({"Result":"OK"})


def preprocessRawData(frame):
    frame['ENDTIME'] = frame['ENDTIME'].astype('datetime64')
    frame['STARTTIME'] = frame['STARTTIME'].astype('datetime64')

    frame['ACTUALCORECOUNT'] = frame['ACTUALCORECOUNT'].apply(lambda x: x if x and x > 0 else 1)
    frame['LOSTWALLTIME'] = (frame['ENDTIME'] - frame['STARTTIME']) / np.timedelta64(1, 's') * frame[
        'ACTUALCORECOUNT']
    frame['LOSTWALLTIME'] = frame.apply(lambda x: x.LOSTWALLTIME if x.JOBSTATUS == 'failed' else 0, axis=1)

    newframe = pd.DataFrame()

    # Cathegorial
    newframe['ATLASRELEASE'] = frame['ATLASRELEASE']
    newframe['PILOTVERSION'] = frame['PILOTID'].apply(
        lambda x: x.split('|')[-1] if x and '|' in x else 'Not specified').str.replace(" ", "%20")
    newframe['CLOUD'] = frame['CLOUD']
    newframe['CMTCONFIG'] = frame['CMTCONFIG']
    newframe['COMPUTINGSITE'] = frame['COMPUTINGSITE']
    newframe['COMPUTINGELEMENT'] = frame['COMPUTINGELEMENT']
    newframe['CREATIONHOST'] = frame['CREATIONHOST']
    newframe['DESTINATIONSE'] = frame['DESTINATIONSE']
    newframe['EVENTSERVICE'] = frame['EVENTSERVICE'].apply(str)
    newframe['PROCESSINGTYPE'] = frame['PROCESSINGTYPE']
    newframe['PRODUSERNAME'] = frame['PRODUSERNAME']
    newframe['RESOURCE_TYPE'] = frame['RESOURCE_TYPE']
    newframe['SPECIALHANDLING'] = frame['SPECIALHANDLING']
    newframe['GSHARE'] = frame['GSHARE']
    newframe['HOMEPACKAGE'] = frame['HOMEPACKAGE']
    newframe['INPUTFILEPROJECT'] = frame['INPUTFILEPROJECT']
    newframe['INPUTFILETYPE'] = frame['INPUTFILETYPE']
    newframe['JEDITASKID'] = frame['JEDITASKID'].apply(str)
    newframe['JOBSTATUS'] = frame['JOBSTATUS'].apply(str)
    newframe['NUCLEUS'] = frame['NUCLEUS'].apply(str)
    newframe['TRANSFORMATION'] = frame['TRANSFORMATION']
    newframe['WORKINGGROUP'] = frame['WORKINGGROUP']
    newframe['REQID'] = frame['REQID'].apply(str)
    newframe.fillna(value="Not specified", inplace=True)

    # Numerical with NA
    newframe['ENDTIME'] = frame['ENDTIME']
    newframe['ACTUALCORECOUNT'] = frame['ACTUALCORECOUNT']
    newframe['LOSTWALLTIME'] = (frame['LOSTWALLTIME'])
    newframe['NINPUTDATAFILES'] = frame['NINPUTDATAFILES']

    newframe['ISSUCCESS'] = frame['JOBSTATUS'].apply(lambda x: 0 if x and x == 'failed' else 1)
    newframe['ISFAILED'] = frame['JOBSTATUS'].apply(lambda x: 1 if x and x == 'failed' else 0)

    newframe.fillna(value=0, inplace=True)

    # Sometimes a job failed even before start
    newframe['STARTTIME'] = frame['STARTTIME']
    return newframe


def analyseProblem(frame_loc):
    newframe_r_X = frame_loc.copy()

    del newframe_r_X['STARTTIME']
    del newframe_r_X['ENDTIME']
    del newframe_r_X['LOSTWALLTIME']
    del newframe_r_X['JOBSTATUS']
    del newframe_r_X['ISSUCCESS']
    del newframe_r_X['ISFAILED']

    categorical_features_indices = list(np.where((newframe_r_X.dtypes == np.object))[0])
    newframe_r_Y = frame_loc[['LOSTWALLTIME']].copy()
    newframe_r_Y['LOSTWALLTIME'] = newframe_r_Y['LOSTWALLTIME'].apply(lambda x: x if x == 0 else np.log(x))
    X_train, X_validation, y_train, y_validation = train_test_split(newframe_r_X, newframe_r_Y, train_size=0.85,
                                                                    random_state=42)
    train_pool = Pool(X_train, label=y_train, cat_features=categorical_features_indices)
    validate_pool = Pool(X_validation, label=y_validation, cat_features=categorical_features_indices)

    with tempfile.TemporaryDirectory() as tmpdirname:
        model = CatBoostRegressor(
            iterations=400,
            random_seed=2,
            logging_level='Silent',
            depth=2,
            learning_rate=0.001,
            task_type="CPU",
            train_dir=tmpdirname,
            used_ram_limit='4gb',

        )
        model.fit(
            train_pool, eval_set=validate_pool,
            plot=False
        )

        feature_importances = model.get_feature_importance(train_pool)
        feature_names = X_train.columns
        interactions = model.get_feature_importance(train_pool, fstr_type=EFstrType.Interaction, prettified=True)
        del model
    return (feature_names, feature_importances, interactions, X_train.columns)


class IssueDescriptor(object):
    features = {}
    timeWindow = {"start": None, "end": None}
    nFailedJobs = 0
    nSuccessJobs = 0
    lostWallTime = 0
    purity = 0
    isSplittable = True

    # We assume that input frame has the same time window as evaluated features
    def filterByIssue(self, frameWhole):
        return frameWhole.loc[(frameWhole[list(self.features)] == pd.Series(self.features)).all(axis=1)]

    def filterByAntiPattern(self, frameWhole):
        return frameWhole.loc[~((frameWhole[list(self.features)] == pd.Series(self.features)).all(axis=1))]

    def appendFeatures(self, parentIssue):
        self.features.update(parentIssue.features)


def scrubStatistics(frame_loc):

    # In this function we determine the principal issue related feature existing in
    # supplied dataframe
    # We count statistics of failed jobs associated with the problem and return correspondent object

    (feature_names, feature_importances, interactions, X_tr_col) = analyseProblem(frame_loc)
    feature_importances_sorted = sorted(zip(feature_importances, feature_names), reverse=True)
    featuresLists = []
    for score, name in feature_importances_sorted:
        if score > 5:
            print('{}: {}'.format(name, score))
        if score > 15:
            featuresLists.append(name)
    if len(featuresLists) > 0:
        groups = frame_loc.groupby(featuresLists[0]).agg(
            {'ISSUCCESS': 'sum', 'ISFAILED': 'sum', 'LOSTWALLTIME': 'sum',
             'ENDTIME': ('min', 'max')}).reset_index().sort_values(by=('LOSTWALLTIME', 'sum'), ascending=False)
    else:
        return None

    foundIssue = IssueDescriptor()
    foundIssue.features = {featuresLists[0]: groups[featuresLists[0]][groups.index[0]]}
    foundIssue.nFailedJobs = groups[('ISFAILED', 'sum')][groups.index[0]]
    foundIssue.nSuccessJobs = groups[('ISSUCCESS', 'sum')][groups.index[0]]
    foundIssue.lostWallTime = groups[('LOSTWALLTIME', 'sum')][groups.index[0]]
    foundIssue.timeWindow = {"start": groups[('ENDTIME', 'min')][groups.index[0]],
                             "end": groups[('ENDTIME', 'max')][groups.index[0]]}
    foundIssue.purity = foundIssue.nFailedJobs / (foundIssue.nFailedJobs + foundIssue.nSuccessJobs)
    return foundIssue


def checkDFEligible(frame_loc):
    ret = True
    if not ((frame_loc.loc[(frame_loc['ISSUCCESS'] == 1)].shape[0] > 3) and (
            frame_loc.loc[(frame_loc['ISFAILED'] == 1)].shape[0] > 5)
            and frame_loc['LOSTWALLTIME'].sum() > 3600 * 24):
        ret = False
    uniqueval = dict(frame_loc.nunique())

    variety = 0
    for feature, value in uniqueval.items():
        if (feature not in ('ISFAILED', 'ISSUCCESS', 'LOSTWALLTIME', 'STARTTIME', 'ENDTIME', 'JOBSTATUS')):
            if value > 1:
                variety += 1
            if variety > 5:
                break
    if variety < 3:
        ret = False

    return ret


def classifyIssue(frame_loc, listOfProblems=None, parentIssue=None, deepnesslog=0):

    global counter

    if parentIssue is not None:
        focusedDFrame = parentIssue.filterByIssue(frame_loc).copy()
    else:
        focusedDFrame = frame_loc.copy()

    nestedIssue = None

    if checkDFEligible(focusedDFrame):
        try:
            nestedIssue = scrubStatistics(focusedDFrame)
        except:
            # print("-" * 60)
            # traceback.print_exc(file=sys.stdout)
            # print("-" * 60)
            pass

    elif parentIssue is not None:
        listOfProblems.append(parentIssue)
        # print("\nAdding issue #", counter, " deepness:", deepnesslog)
        # print("Parent features:", parentIssue.features)
        # print("nSuccessJobs:", parentIssue.nSuccessJobs, " nFailedJobs:", parentIssue.nFailedJobs)
        counter += 1
        return None

    if (parentIssue is not None) and (nestedIssue is not None):
        nestedIssue.appendFeatures(parentIssue)

        if (
                nestedIssue.purity > parentIssue.purity and parentIssue.nSuccessJobs > 0 and parentIssue.nFailedJobs > 0) or not checkDFEligible(
                frame_loc):
            classifyIssue(frame_loc, listOfProblems, nestedIssue, deepnesslog + 1)
        else:
            listOfProblems.append(parentIssue)
            # print("\nAdding issue #", counter, " deepness:", deepnesslog)
            # print("Parent features:", parentIssue.features)
            # print("Nested features:", nestedIssue.features)
            # print("nSuccessJobs:", nestedIssue.nSuccessJobs, " nFailedJobs:", nestedIssue.nFailedJobs)
            counter += 1


    elif (nestedIssue is not None):
        classifyIssue(focusedDFrame, listOfProblems, nestedIssue, deepnesslog + 1)

    if (nestedIssue is not None):
        restDFrame = nestedIssue.filterByAntiPattern(frame_loc).copy()
        if checkDFEligible(restDFrame):
            try:
                classifyIssue(restDFrame, listOfProblems, parentIssue, deepnesslog + 1)
            except:
                # print("-" * 60)
                # traceback.print_exc(file=sys.stdout)
                # print("-" * 60)
                pass

    del focusedDFrame

