from opint_framework.core.prototypes.BaseAgent import BaseAgent
import threading
from opint_framework.apps.workload_jobsbuster.models import WorkflowIssue, WorkflowIssueMetadata, AnalysisSessions, WorkflowIssueTicks
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor, CatBoostClassifier, Pool, EFstrType
import tempfile
import datetime
from django.utils import timezone
import opint_framework.apps.workload_jobsbuster.api.pandaDataImporter as dataImporter
import traceback, sys

class JobsAnalyserAgent(BaseAgent):
    lock = threading.RLock()
    counter = 0
    def init(self):
        pass

    def processCycle(self):
        self.counter = 0

        print("JobsAnalyserAgent started")
        lastSession = AnalysisSessions.objects.using('jobs_buster_persistency').order_by('-timewindow_end').first()
#         if lastSession:
#             timeGap = datetime.datetime.now(datetime.timezone.utc) - lastSession.timewindow_end.astimezone(timezone.utc)
#             if timeGap < datetime.timedelta(minutes=30):
#                 print("JobsAnalyserAgent finished")
#                 return 0
# #            datefrom = lastSession.timewindow_end
# #            minutes = 240
#             datefrom = datetime.datetime.utcnow() - datetime.timedelta(minutes=240)
#             dateto = datetime.datetime.utcnow()
#         else:
#             datefrom = datetime.datetime.utcnow() - datetime.timedelta(minutes=240)
#             dateto = datetime.datetime.utcnow()

        datefrom = datetime.datetime.utcnow() - datetime.timedelta(minutes=360)
        dateto = datetime.datetime.utcnow()

        dbsession = AnalysisSessions.objects.using('jobs_buster_persistency').create(timewindow_start=datefrom,
                                                                                     timewindow_end=dateto)
        pandasdf = dataImporter.retreiveData(datefrom, dateto)

        #pickle.dump(pandasdf, open(settings.datafilespath + "rawdataframe0_daily1.sr", 'wb+'))
        #pandasdf = pickle.load(open(settings.datafilespath + "rawdataframe0_daily1.sr", 'rb'))

        preprocessedFrame = self.preprocessRawData(pandasdf)

        listOfProblems = []
        self.classifyIssue(preprocessedFrame, listOfProblems, None)
        listOfProblems = self.removeDublicates(listOfProblems)
        listOfProblems = sorted(listOfProblems, key=lambda i: i.lostWallTime, reverse=True)

        #pickle.dump(listOfProblems, open(settings.datafilespath + "rawdataframe0_daily2.sr", 'wb+'))
        #listOfProblems = pickle.load(open(settings.datafilespath + "rawdataframe0_daily2.sr", 'rb'))

        self.removeIssuesNewerThan(datefrom)

        for spottedProblem in listOfProblems:
            issue = WorkflowIssue.objects.using('jobs_buster_persistency').create(session_id_fk=dbsession,
                                                                                  observation_started=
                                                                                  spottedProblem.timeWindow[
                                                                                      'start'].replace(
                                                                                      tzinfo=timezone.utc),
                                                                                  observation_finished=
                                                                                  spottedProblem.timeWindow[
                                                                                      'end'].replace(
                                                                                      tzinfo=timezone.utc),
                                                                                  walltime_loss=spottedProblem.lostWallTime,
                                                                                  nFailed_jobs=spottedProblem.nFailedJobs,
                                                                                  nSuccess_jobs=spottedProblem.nSuccessJobs)

            for featureName, featureValue in spottedProblem.features.items():
                metadata = WorkflowIssueMetadata.objects.using('jobs_buster_persistency').create(issue_id_fk=issue,
                                                                                                 key=featureName,
                                                                                                 value=featureValue)
            ticks = self.historgramFailures(spottedProblem.filterByIssue(preprocessedFrame), spottedProblem)
            self.saveFailureTicks(ticks, issue)

        dbsession.analysis_finished = datetime.datetime.now(datetime.timezone.utc)
        dbsession.save(using='jobs_buster_persistency')
        print("JobsAnalyserAgent finished")


    def removeIssuesNewerThan(self, date):
        WorkflowIssue.objects.using('jobs_buster_persistency').filter(observation_started__gt=date).delete()


    def saveFailureTicks(self,ticks,issue):
        ticksEntries = [WorkflowIssueTicks(issue_id_fk=issue, tick_time=t, walltime_loss=w, nFailed_jobs=nf) for t,w,nf in ticks]
        WorkflowIssueTicks.objects.using('jobs_buster_persistency').bulk_create(ticksEntries)


    def historgramFailures(self, dataFrame, spottedProblem):
        grp = dataFrame.loc[(dataFrame['ISFAILED'] == 1) & (dataFrame['ENDTIME'] > spottedProblem.timeWindow['start'])
                            & (dataFrame['ENDTIME'] < spottedProblem.timeWindow['end'])][['ENDTIME', 'LOSTWALLTIME', 'ISFAILED']]\
            .groupby([pd.Grouper(freq="15Min", key='ENDTIME')]).agg({
            'LOSTWALLTIME': sum,
            'ISFAILED': 'count',
        })
        wallList = grp['LOSTWALLTIME'].values.tolist()
        NfailList = grp['ISFAILED'].values.tolist()
        index = grp.index.to_pydatetime().tolist()
        return zip(index, wallList, NfailList)


    def removeDublicates(self, listOfProblems):
        return set(listOfProblems)

    def preprocessRawData(self, frame):
        frame['ENDTIME'] = frame['ENDTIME'].astype('datetime64')
        frame['STARTTIME'] = frame['STARTTIME'].astype('datetime64')

        frame['ACTUALCORECOUNT'] = frame['ACTUALCORECOUNT'].apply(lambda x: x if x and x > 0 else 1)
        frame['LOSTWALLTIME'] = (frame['ENDTIME'] - frame['STARTTIME']) / np.timedelta64(1, 's') * frame[
            'ACTUALCORECOUNT']
        frame['LOSTWALLTIME'] = frame.apply(lambda x: x.LOSTWALLTIME if x.JOBSTATUS == 'failed' else 0, axis=1)

        newframe = pd.DataFrame()

        # Cathegorial
        newframe['ATLASRELEASE'] = frame['ATLASRELEASE']
#        newframe['PILOTVERSION'] = frame['PILOTID'].apply(
#            lambda x: x.split('|')[-1] if x and '|' in x else 'Not specified').str.replace(" ", "%20")
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


    def analyseProblem(self, frame_loc):
        newframe_r_X = frame_loc.copy()

        del newframe_r_X['STARTTIME']
        del newframe_r_X['ENDTIME']
        del newframe_r_X['LOSTWALLTIME']
        del newframe_r_X['JOBSTATUS']
        del newframe_r_X['ISSUCCESS']
        del newframe_r_X['ISFAILED']

        categorical_features_indices = list(np.where((newframe_r_X.dtypes == np.object))[0])
        #newframe_r_Y = frame_loc[['LOSTWALLTIME']].copy()
        #newframe_r_Y['LOSTWALLTIME'] = newframe_r_Y['LOSTWALLTIME'].apply(lambda x: x if x == 0 else np.log(x))

        newframe_r_Y = frame_loc[['ISFAILED']].copy()

        X_train, X_validation, y_train, y_validation = train_test_split(newframe_r_X, newframe_r_Y, train_size=0.85,
                                                                        random_state=42)
        train_pool = Pool(X_train, label=y_train, cat_features=categorical_features_indices)
        validate_pool = Pool(X_validation, label=y_validation, cat_features=categorical_features_indices)

        with tempfile.TemporaryDirectory() as tmpdirname:
            model = CatBoostClassifier(
                iterations=400,
                random_seed=2,
                logging_level='Silent',
                depth=2,
                learning_rate=0.01,
                task_type="CPU",
                train_dir=tmpdirname,
                used_ram_limit='6gb',
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


    def scrubStatistics(self, frame_loc):

        # In this function we determine the principal issue related feature existing in
        # supplied dataframe
        # We count statistics of failed jobs associated with the problem and return correspondent object

        (feature_names, feature_importances, interactions, X_tr_col) = self.analyseProblem(frame_loc)
        feature_importances_sorted = sorted(zip(feature_importances, feature_names), reverse=True)
        featuresLists = []
        score, name = feature_importances_sorted[0]
        if score > 25:
            featuresLists.append(name)
        if len(featuresLists) > 0:
            groups = frame_loc.groupby(featuresLists[0]).agg(
                {'ISSUCCESS': 'sum', 'ISFAILED': 'sum', 'LOSTWALLTIME': 'sum',
                 'ENDTIME': ('min', 'max')}).reset_index().sort_values(by=('LOSTWALLTIME', 'sum'), ascending=False)
        else:
            return None

        foundIssue = self.IssueDescriptor()
        foundIssue.features = {featuresLists[0]: groups[featuresLists[0]][groups.index[0]]}
        foundIssue.nFailedJobs = groups[('ISFAILED', 'sum')][groups.index[0]]
        foundIssue.nSuccessJobs = groups[('ISSUCCESS', 'sum')][groups.index[0]]
        foundIssue.lostWallTime = groups[('LOSTWALLTIME', 'sum')][groups.index[0]]
        foundIssue.timeWindow = {"start": groups[('ENDTIME', 'min')][groups.index[0]],
                                 "end": groups[('ENDTIME', 'max')][groups.index[0]]}
        foundIssue.purity = foundIssue.nFailedJobs / (foundIssue.nFailedJobs + foundIssue.nSuccessJobs)
        return foundIssue


    def checkDFEligible(self, frame_loc):
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


    def classifyIssue(self, frame_loc, listOfProblems=None, parentIssue=None, deepnesslog=0):

        if parentIssue is not None:
            focusedDFrame = parentIssue.filterByIssue(frame_loc).copy()
        else:
            focusedDFrame = frame_loc.copy()

        nestedIssue = None

        if self.checkDFEligible(focusedDFrame):
            try:
                nestedIssue = self.scrubStatistics(focusedDFrame)
            except:
                print("-" * 60)
                traceback.print_exc(file=sys.stdout)
                print("-" * 60)
                pass

        elif parentIssue is not None:
            listOfProblems.append(parentIssue)
            # print("\nAdding issue #", counter, " deepness:", deepnesslog)
            # print("Parent features:", parentIssue.features)
            # print("nSuccessJobs:", parentIssue.nSuccessJobs, " nFailedJobs:", parentIssue.nFailedJobs)
            self.counter += 1
            return None

        if (parentIssue is not None) and (nestedIssue is not None):
            nestedIssue.appendFeatures(parentIssue)

            if (nestedIssue.purity > parentIssue.purity and parentIssue.nSuccessJobs > 0 and parentIssue.nFailedJobs > 0) or not self.checkDFEligible(
                    frame_loc):
                self.classifyIssue(frame_loc, listOfProblems, nestedIssue, deepnesslog + 1)
            else:
                listOfProblems.append(parentIssue)
                # print("\nAdding issue #", counter, " deepness:", deepnesslog)
                # print("Parent features:", parentIssue.features)
                # print("Nested features:", nestedIssue.features)
                # print("nSuccessJobs:", nestedIssue.nSuccessJobs, " nFailedJobs:", nestedIssue.nFailedJobs)
                self.counter += 1


        elif (nestedIssue is not None):
            self.classifyIssue(focusedDFrame, listOfProblems, nestedIssue, deepnesslog + 1)

        if (nestedIssue is not None):
            restDFrame = nestedIssue.filterByAntiPattern(frame_loc).copy()
            if self.checkDFEligible(restDFrame):
                try:
                    self.classifyIssue(restDFrame, listOfProblems, parentIssue, deepnesslog + 1)
                except:
                    # print("-" * 60)
                    # traceback.print_exc(file=sys.stdout)
                    # print("-" * 60)
                    pass

        del focusedDFrame
