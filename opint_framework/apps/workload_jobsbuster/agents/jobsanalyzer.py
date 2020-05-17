from opint_framework.core.prototypes.BaseAgent import BaseAgent
import threading
from opint_framework.apps.workload_jobsbuster.models import WorkflowIssue, WorkflowIssueMetadata, AnalysisSessions, WorkflowIssueTicks
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor, CatBoostClassifier, Pool, EFstrType
from sklearn.feature_extraction.text import CountVectorizer
import hashlib

from opint_framework.apps.workload_jobsbuster.agents.postprocessing import checkisHPC, mergedicts
import tempfile
import datetime
from django.utils import timezone
import opint_framework.apps.workload_jobsbuster.api.pandaDataImporter as dataImporter
import json


diagfields = ['DDMERRORDIAG', 'BROKERAGEERRORDIAG', 'DDMERRORDIAG', 'EXEERRORDIAG', 'JOBDISPATCHERERRORDIAG',
             'PILOTERRORDIAG', 'SUPERRORDIAG', 'TASKBUFFERERRORDIAG']
diagcodefields = ['BROKERAGEERRORCODE', 'DDMERRORCODE', 'EXEERRORCODE', 'JOBDISPATCHERERRORCODE', 'PILOTERRORCODE',
             'SUPERRORCODE', 'TASKBUFFERERRORCODE', 'TRANSEXITCODE']

OI_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"

class JobsAnalyserAgent(BaseAgent):
    lock = threading.RLock()
    errFrequency = None
    errorsToProcess = None

    def init(self):
        pass

    def processCycle(self):
        print("JobsAnalyserAgent started")
        lastSession = AnalysisSessions.objects.using('jobs_buster_persistency').order_by('-timewindow_end').first()

        datefrom = datetime.datetime.utcnow() - datetime.timedelta(minutes=12*60)
        dateto = datetime.datetime.utcnow()
        dbsession = AnalysisSessions.objects.using('jobs_buster_persistency').create(timewindow_start=datefrom,
                                                                                     timewindow_end=dateto)
        pandasdf = dataImporter.retreiveData(datefrom, dateto)
        preprocessedFrame = self.preprocessRawData(pandasdf)

        listOfProblems = []
        self.findIssues(preprocessedFrame, listOfProblems)
        listOfProblems = self.reduceIssues(preprocessedFrame, listOfProblems)
        checkisHPC(preprocessedFrame, listOfProblems)
        for spottedProblem in listOfProblems:
            payload_type = 0
            if spottedProblem.isHPC and spottedProblem.isGRID:
                payload_type = 1
            if spottedProblem.isHPC and not spottedProblem.isGRID:
                payload_type = 2

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
                                                                                  nSuccess_jobs=spottedProblem.nSuccessJobs,
                                                                                  payload_type=payload_type,
                                                                                  err_messages=json.dumps(dict(spottedProblem.errorsStrings)),
                                                                                  )

            for featureName, featureValue in spottedProblem.features.items():
                metadata = WorkflowIssueMetadata.objects.using('jobs_buster_persistency').create(issue_id_fk=issue,
                                                                                                 key=featureName,
                                                                                                 value=featureValue)
            ticks = self.historgramFailures(spottedProblem.filterByIssue(preprocessedFrame), spottedProblem)
            self.saveFailureTicks(ticks, issue)

        AnalysisSessions.objects.using('jobs_buster_persistency').filter(analysis_finished__isnull=False).delete()
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
#        newframe['ATLASRELEASE'] = frame['ATLASRELEASE']
#        newframe['PILOTVERSION'] = frame['PILOTID'].apply(
#            lambda x: x.split('|')[-1] if x and '|' in x else 'Not specified').str.replace(" ", "%20")
#        newframe['CLOUD'] = frame['CLOUD']
#        newframe['CMTCONFIG'] = frame['CMTCONFIG']
        newframe['SITE'] = frame['SITE']
        newframe['COMPUTINGSITE'] = frame['COMPUTINGSITE']
        newframe['COMPUTINGELEMENT'] = frame['COMPUTINGELEMENT']
        newframe['CREATIONHOST'] = frame['CREATIONHOST']
        newframe['DESTINATIONSE'] = frame['DESTINATIONSE']
#        newframe['EVENTSERVICE'] = frame['EVENTSERVICE'].apply(str)
#       newframe['PROCESSINGTYPE'] = frame['PROCESSINGTYPE']
        newframe['PRODUSERNAME'] = frame['PRODUSERNAME']
#        newframe['RESOURCE_TYPE'] = frame['RESOURCE_TYPE']
#        newframe['SPECIALHANDLING'] = frame['SPECIALHANDLING']
#       newframe['GSHARE'] = frame['GSHARE']
#        newframe['HOMEPACKAGE'] = frame['HOMEPACKAGE']
#        newframe['INPUTFILEPROJECT'] = frame['INPUTFILEPROJECT']
#        newframe['INPUTFILETYPE'] = frame['INPUTFILETYPE']
        newframe['JEDITASKID'] = frame['JEDITASKID'].apply(str)
        newframe['JOBSTATUS'] = frame['JOBSTATUS'].apply(str)
        newframe['NUCLEUS'] = frame['NUCLEUS'].apply(str)
#        newframe['TRANSFORMATION'] = frame['TRANSFORMATION']
#        newframe['WORKINGGROUP'] = frame['WORKINGGROUP']
        newframe['REQID'] = frame['REQID'].apply(str)

        # Errors data
        newframe['BROKERAGEERRORCODE'] = frame['BROKERAGEERRORCODE']
        newframe['BROKERAGEERRORDIAG'] = frame['BROKERAGEERRORDIAG']
        newframe['DDMERRORCODE'] = frame['DDMERRORCODE']
        newframe['DDMERRORDIAG'] = frame['DDMERRORDIAG']
        newframe['EXEERRORCODE'] = frame['EXEERRORCODE']
        newframe['EXEERRORDIAG'] = frame['EXEERRORDIAG']
        newframe['JOBDISPATCHERERRORCODE'] = frame['JOBDISPATCHERERRORCODE']
        newframe['JOBDISPATCHERERRORDIAG'] = frame['JOBDISPATCHERERRORDIAG']
        newframe['PILOTERRORCODE'] = frame['PILOTERRORCODE']
        newframe['PILOTERRORDIAG'] = frame['PILOTERRORDIAG']
        newframe['SUPERRORCODE'] = frame['SUPERRORCODE']
        newframe['SUPERRORDIAG'] = frame['SUPERRORDIAG']
        newframe['TASKBUFFERERRORCODE'] = frame['TASKBUFFERERRORCODE']
        newframe['TASKBUFFERERRORDIAG'] = frame['TASKBUFFERERRORDIAG']
        newframe['TRANSEXITCODE'] = frame['TRANSEXITCODE']

        newframe.fillna(value="Not specified", inplace=True)

        # Numerical with NA
        newframe['ENDTIME'] = frame['ENDTIME']
#        newframe['ACTUALCORECOUNT'] = frame['ACTUALCORECOUNT']
        newframe['LOSTWALLTIME'] = (frame['LOSTWALLTIME'])
#        newframe['NINPUTDATAFILES'] = frame['NINPUTDATAFILES']

        newframe['ISSUCCESS'] = frame['JOBSTATUS'].apply(lambda x: 0 if x and x == 'failed' else 1)
        newframe['ISFAILED'] = frame['JOBSTATUS'].apply(lambda x: 1 if x and x == 'failed' else 0)
        newframe['PANDAID'] = frame['PANDAID']
        newframe.fillna(value=0, inplace=True)

        # Sometimes a job failed even before start
        newframe['STARTTIME'] = frame['STARTTIME']
        for field in diagfields:
            self.removeStopWords(field, newframe)
        newframe['combinederrors'] = newframe.apply(self.combineErrors, axis=1)

        frequencyW = newframe.groupby('combinederrors').agg(
            {'ISSUCCESS': 'sum', 'ISFAILED': 'sum', 'LOSTWALLTIME': 'sum',
             'ENDTIME': ('min', 'max')}).reset_index().sort_values(by=('LOSTWALLTIME', 'sum'), ascending=False).head(50)
        frequencyF = newframe.groupby('combinederrors').agg(
            {'ISSUCCESS': 'sum', 'ISFAILED': 'sum', 'LOSTWALLTIME': 'sum',
             'ENDTIME': ('min', 'max')}).reset_index().sort_values(by=('ISFAILED', 'sum'), ascending=False).head(50)
        self.errFrequency = pd.concat([frequencyW, frequencyF])
        for i, row in self.errFrequency.iterrows():
            if row[4] == row[5]:
                self.errFrequency.loc[i, 'ENDTIME'] = (
                row[4] - datetime.timedelta(minutes=10), row[5] + datetime.timedelta(minutes=10))
        self.errorsToProcess = list(set(self.errFrequency['combinederrors'].tolist()))
        return newframe


    def getHash(self, strtohash):
        hash_object = hashlib.sha1(str(strtohash).encode())
        return hash_object.hexdigest()[:6]


    def combineErrors(self, row):
        ret = ''
        for fieldname in diagfields + diagcodefields:
            ret += '|' + self.getHash(row[fieldname])
        return ret


    def removeStopWords(self, diagfield, frame):
        def my_tokenizer(s):
            return s.split()

        vect = CountVectorizer(tokenizer=my_tokenizer, analyzer="word", stop_words=None, preprocessor=None)
        corpus = frame[diagfield].tolist()
        bag_of_words = vect.fit_transform(corpus)
        sum_words = bag_of_words.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in vect.vocabulary_.items()]
        words_freqf = [x[0] for x in filter(lambda x: x[1] < 3, words_freq)]
        # print(words_freqf)
        words_freqf = set(words_freqf)

        def replace_all(text):
            common_tokens = set(my_tokenizer(text)).intersection(words_freqf)
            for i in common_tokens:
                text = text.replace(i, '**REPLACEMENT**') if len(i) > 3 else text
            return text
        frame[diagfield] = frame[diagfield].apply(replace_all)

    def analyseProblem(self, frame):
        subsample_error_segment = frame.copy()
        subsample_col = subsample_error_segment.filter(['COMPUTINGSITE'], axis=1)
        COMPUTINGSITE_COLS = pd.get_dummies(subsample_col, prefix=['COMPUTINGSITE'])
        subsample_col = subsample_error_segment.filter(['DESTINATIONSE'], axis=1)
        DESTINATIONSE_COLS = pd.get_dummies(subsample_col, prefix=['DESTINATIONSE'])
        subsample_col = subsample_error_segment.filter(['JEDITASKID'], axis=1)
        JEDITASKID_COLS = pd.get_dummies(subsample_col, prefix=['JEDITASKID'])
        subsample_col = subsample_error_segment.filter(['COMPUTINGELEMENT'], axis=1)
        COMPUTINGELEMENT_COLS = pd.get_dummies(subsample_col, prefix=['COMPUTINGELEMENT'])
        subsample_col = subsample_error_segment.filter(['CREATIONHOST'], axis=1)
        CREATIONHOST_COLS = pd.get_dummies(subsample_col, prefix=['CREATIONHOST'])
        subsample_col = subsample_error_segment.filter(['NUCLEUS'], axis=1)
        NUCLEUS_COLS = pd.get_dummies(subsample_col, prefix=['NUCLEUS'])
        subsample_col = subsample_error_segment.filter(['REQID'], axis=1)
        REQID_COLS = pd.get_dummies(subsample_col, prefix=['REQID'])
        subsample_col = subsample_error_segment.filter(['PRODUSERNAME'], axis=1)
        PRODUSERNAME_COLS = pd.get_dummies(subsample_col, prefix=['PRODUSERNAME'])
        XDF = pd.concat(
            [COMPUTINGSITE_COLS, DESTINATIONSE_COLS, JEDITASKID_COLS, COMPUTINGELEMENT_COLS, CREATIONHOST_COLS,
             NUCLEUS_COLS, REQID_COLS], axis=1)
        YDF = subsample_error_segment.filter(['JOBSTATUS'], axis=1)

        X_train, X_validation, y_train, y_validation = train_test_split(XDF, YDF, train_size=0.90,
                                                                        random_state=42)
        train_pool = Pool(X_train, label=y_train)
        validate_pool = Pool(X_validation, label=y_validation)

        with tempfile.TemporaryDirectory() as tmpdirname:
            model = CatBoostClassifier(
                iterations=200,
                random_seed=2,
                logging_level='Silent',
                depth=1,
                learning_rate=0.01,
                task_type="CPU",
                train_dir=tmpdirname,
            )
            model.fit(
                train_pool, eval_set=validate_pool,
                #     logging_level='Verbose',  # you can uncomment this for text output
                plot=False
            )

            feature_importances = model.get_feature_importance(train_pool)
            feature_names = X_train.columns
            interactions = model.get_feature_importance(train_pool, fstr_type=EFstrType.Interaction, prettified=True)
            del model

        return (feature_names, feature_importances, interactions, X_train.columns)


    def convertFeaturesImportances(self, feat_impts):
        retList = []
        for feat_imp in feat_impts:
            importance = feat_imp[0]
            feature = feat_imp[1].split('_')[0]
            value = feat_imp[1][len(feature) + 1:]
            retList.append((feature, value, importance))
        return retList


    def createOrthogonalSets(self, frame_loc, features):
        combinations = []
        cathegories = list([feature[0] for feature in features])
        picked = []
        for i, _ in enumerate(cathegories):
            if i not in picked:
                compbination_not_found = True
                for j in range(i + 1, len(cathegories)):
                    if j not in picked:
                        assessment = self.probeMask(frame_loc, [features[i], features[j]])
                        if assessment and assessment[0] > 0:
                            picked.append(j)
                            combinations.append([features[i], features[j]])
                            compbination_not_found = False
                if compbination_not_found:
                    assessment = self.probeMask(frame_loc, [features[i]])
                    if assessment and assessment[0] > 0:
                        combinations.append([features[i]])
        return combinations


    def probeMask(self, frame_loc, filters):
        mask = True
        featuresList = []
        for filt in filters:
            # print(filt)
            mask = mask & (frame_loc[filt[0]] == filt[1])
            featuresList.append(filt[0])
        totals = frame_loc.loc[mask].groupby(list(set(featuresList))).agg(
            {'ISSUCCESS': 'sum', 'ISFAILED': 'sum', 'LOSTWALLTIME': 'sum', }).reset_index()
        if len(totals.index) == 0:
            return None
        # print(totals.head())
        return (totals['ISFAILED'][totals.index[0]],
                totals['ISSUCCESS'][totals.index[0]],
                totals['LOSTWALLTIME'][totals.index[0]])


    class IssueDescriptor(object):
        features = {}
        errorsStrings = {}
        timeWindow = {"start": None, "end": None}
        nFailedJobs = 0
        nSuccessJobs = 0
        lostWallTime = 0
        purity = 0
        isSplittable = True
        isHPC = False
        isGRID = False

        # We assume that input frame has the same time window as evaluated features
        def filterByIssue(self, frameWhole):
            mask = (frameWhole['ENDTIME'] >= self.timeWindow['start']) & (
                        frameWhole['ENDTIME'] <= self.timeWindow['end'])
            return frameWhole.loc[(frameWhole[list(self.features)] == pd.Series(self.features)).all(axis=1)].loc[mask]
        def filterByAntiPattern(self, frameWhole):
            mask = (frameWhole['ENDTIME'] >= self.timeWindow['start']) & (
                        frameWhole['ENDTIME'] <= self.timeWindow['end'])
            return frameWhole.loc[~((frameWhole[list(self.features)] == pd.Series(self.features)).all(axis=1))].loc[
                mask]
        def appendFeatures(self, parentIssue):
            self.features.update(parentIssue.features)


    def getErrorsForFrameSlice(self, frame):
        errors = {}
        for errfield in diagfields:
            if not frame[errfield].iloc[0] == 'Not specified':
                errors[errfield] = {frame[errfield].iloc[0]:len(frame.index)}
        return errors


    def scrubStatistics(self, frame_loc, hashval):

        # In this function we determine the principal issue related feature existing in
        # supplied dataframe
        # We count statistics of failed jobs associated with the problem and return correspondent object

        freq = self.errFrequency.loc[self.errFrequency['combinederrors'] == hashval].iloc[0]

        mask_f = (frame_loc['ENDTIME'] >= freq['ENDTIME']['min']) & (frame_loc['ENDTIME'] <= freq['ENDTIME']['max']) & (
                    frame_loc['combinederrors'] == hashval)
        mask_s = (frame_loc['ENDTIME'] >= freq['ENDTIME']['min'] - datetime.timedelta(minutes=20)) & (
                    frame_loc['ENDTIME'] <= freq['ENDTIME']['max'] + datetime.timedelta(minutes=20)) & (
                             frame_loc['JOBSTATUS'] == 'finished')

        try:
            (feature_names, feature_importances, interactions, X_tr_col) = self.analyseProblem(frame_loc.loc[mask_f | mask_s])
        except:
            return None
        feature_importances = list(filter(lambda x: x[0] > 10, zip(feature_importances, feature_names)))
        feature_importances = self.convertFeaturesImportances(feature_importances)
        feature_importances = self.createOrthogonalSets(frame_loc.loc[mask_f | mask_s], feature_importances)
        issues = []
        for features in feature_importances:
            featuresList = []
            for filt in features:
                featuresList.append(filt[0])
            if len(featuresList) > 0:
                mask = mask_f | mask_s
                for feature in features:
                    mask = mask & (frame_loc[filt[0]] == filt[1])
                groups = frame_loc.loc[mask].groupby(featuresList).agg(
                    {'ISSUCCESS': 'sum', 'ISFAILED': 'sum', 'LOSTWALLTIME': 'sum',
                     'ENDTIME': ('min', 'max')}).reset_index().sort_values(by=('ISFAILED', 'sum'), ascending=False)

            foundIssue = self.IssueDescriptor()
            foundIssue.features = {feature: groups[feature][groups.index[0]] for feature in featuresList}
            foundIssue.feature_importances = feature_importances
            foundIssue.nFailedJobs = groups[('ISFAILED', 'sum')][groups.index[0]]
            foundIssue.nSuccessJobs = groups[('ISSUCCESS', 'sum')][groups.index[0]]
            foundIssue.lostWallTime = groups[('LOSTWALLTIME', 'sum')][groups.index[0]]
            foundIssue.errorsStrings = self.getErrorsForFrameSlice(frame_loc.loc[mask_f])

            starttime = groups[('ENDTIME', 'min')][groups.index[0]]
            endtime = groups[('ENDTIME', 'max')][groups.index[0]]
            if starttime == endtime:
                starttime = starttime - datetime.timedelta(minutes=10)
                endtime = endtime + datetime.timedelta(minutes=10)

            foundIssue.timeWindow = {"start": starttime,
                                     "end": endtime}
            foundIssue.purity = foundIssue.nFailedJobs / (foundIssue.nFailedJobs + foundIssue.nSuccessJobs)
            issues.append(foundIssue)
        return issues


    def findIssues(self, frame_loc, listOfProblems=None):
        for errorHash in self.errorsToProcess:
            issue = self.scrubStatistics(frame_loc, errorHash)
            if issue:
                listOfProblems.extend(issue)


    def comparefeatures(self, features1, features2):
        numCommonFeat = 0
        for key, value in features1.items():
            if (features2.get(key, None) == value):
                numCommonFeat += 1
        if (len(features1) == len(features1) == numCommonFeat):
            return True
        if (numCommonFeat > 0) and ((numCommonFeat + 1 == len(features1)) or (numCommonFeat + 1 == len(features2))):
            return True
        return False

    def mergeissueinto(self, destination_issue, source_issue):
        destination_issue.nFailedJobs += source_issue.nFailedJobs
        destination_issue.nSuccessJobs += source_issue.nSuccessJobs
        destination_issue.lostWallTime += source_issue.lostWallTime
        destination_issue.errorsStrings = mergedicts(destination_issue.errorsStrings, source_issue.errorsStrings)
        destination_issue.timeWindow = {
            "start": min(destination_issue.timeWindow['start'], source_issue.timeWindow['start']),
            "end": max(destination_issue.timeWindow['end'], source_issue.timeWindow['end'])}
        return destination_issue

    def mergeissues(self, issue1, issue2):
        if issue1.nFailedJobs > issue2.nFailedJobs:
            return self.mergeissueinto(issue1, issue2)
        else:
            return self.mergeissueinto(issue2, issue1)

    def reduceIssues(self, frame_loc, issues):
        classes = {}
        overlapping = {}
        totals = {}
        for index, issue in enumerate(issues):
            frame = issue.filterByIssue(frame_loc).loc[frame_loc['ISFAILED'] == 1]
            totals[index] = len(frame.index)
            pandaids = frame['PANDAID'].tolist()
            classes[index] = set(pandaids)
        for i, pandaids in classes.items():
            for j in range(i + 1, len(classes) - 1):
                if len(pandaids & classes[j]):
                    overlapping.setdefault(i, {})[j] = len(pandaids & classes[j])
        issues_to_delete = []
        for key in totals:
            if key in overlapping:
                if totals[key] < max([totals[crossed_item] for crossed_item in overlapping[key]]):
                    issues_to_delete.append(key)
        reduced_issues = []
        for index, issue in enumerate(issues):
            if index not in issues_to_delete:
                reduced_issues.append(issue)

        mergelist = {}
        for index, issue in enumerate(reduced_issues):
            for j in range(index + 1, len(reduced_issues) - 1):
                features1 = issue.features
                features2 = reduced_issues[j].features
                if self.comparefeatures(features1, features2):
                    mergelist.setdefault(index, []).append(j)

        mergelist2 = []
        for key, value in mergelist.items():
            items = set([key] + value)
            mergelist2.append(items)

        for i in mergelist2:
            for item in mergelist2:
                intersection = item.intersection(i)
                if len(intersection) > 0:
                    i.update(item)
        mergelist2 = list(set(map(tuple, mergelist2)))

        mergeditemsindexes = set()
        mergeditems = []
        for items in mergelist2:
            for i, item in enumerate(items[1:], 1):
                reduced_issues[items[0]] = self.mergeissues(reduced_issues[items[0]], reduced_issues[item])
                mergeditemsindexes.add(item)
            mergeditems.append(reduced_issues[items[0]])
            mergeditemsindexes.add(items[0])

        for i, item in enumerate(reduced_issues):
            if i not in mergeditemsindexes:
                mergeditems.append(reduced_issues[i])
        return mergeditems