from django.db import models

class AnalysisSessions(models.Model):
    session_id = models.AutoField(primary_key=True, db_column='SESSION_ID')
    timewindow_start = models.DateTimeField(db_column='TIMEWINDOW_START')
    timewindow_end = models.DateTimeField(db_column='TIMEWINDOW_END')
    analysis_started = models.DateTimeField(db_column='ANALYSIS_STARTED', auto_now_add=True)
    analysis_finished = models.DateTimeField(db_column='ANALYSIS_FINISHED')
    class Meta:
        db_table = 'ATLAS_JOBS_BUSTER_ANSESS'


class WorkflowIssue(models.Model):
    issue_id = models.AutoField(primary_key=True, db_column='ISSUE_ID')
    session_id_fk = models.ForeignKey(AnalysisSessions, on_delete=models.CASCADE, db_column='SESSION_ID_FK')
    observation_started = models.DateTimeField(db_column='OBSERVATION_STARTED')
    observation_finished = models.DateTimeField(db_column='OBSERVATION_FINISHED')
    walltime_loss = models.IntegerField(null=True, db_column='WALLTIME_LOSS')
    nFailed_jobs = models.IntegerField(null=True, db_column='NFAILED_JOBS')
    nSuccess_jobs = models.IntegerField(null=True, db_column='NSUCCESS_JOBS')
    payload_type = models.IntegerField(null=True, db_column='PAYLOAD_TYPE')
    err_messages = models.TextField(null=True, db_column='ERR_MESSAGES')

    """
        _data = models.TextField(
                db_column='data',
                blank=True)
    
        def set_data(self, data):
            self._data = base64.encodestring(data)
    
        def get_data(self):
            return base64.decodestring(self._data)
    
        data = property(get_data, set_data)
    """

    class Meta:
        db_table = u'ATLAS_JOBS_BUSTER_ISSUE'


class WorkflowIssueMetadata(models.Model):
    meta_id = models.AutoField(primary_key=True, db_column='META_ID')
    issue_id_fk = models.ForeignKey(WorkflowIssue, on_delete=models.CASCADE, db_column='ISSUE_ID_FK')
    key = models.CharField(max_length=512)
    value = models.CharField(max_length=512)
    class Meta:
        db_table = 'ATLAS_JOBS_BUSTER_ISSUE_META'


class WorkflowIssueTicks(models.Model):
    tick_id = models.AutoField(primary_key=True, db_column='TICK_ID')
    issue_id_fk = models.ForeignKey(WorkflowIssue, on_delete=models.CASCADE, db_column='ISSUE_ID_FK')
    tick_time = models.DateTimeField(db_column='TICK_TIME')
    walltime_loss = models.IntegerField(null=True, db_column='WALLTIME_LOSS')
    nFailed_jobs = models.IntegerField(null=True, db_column='NFAILED_JOBS')
    class Meta:
        db_table = 'ATLAS_JOBS_BUSTER_ISSUE_TICKS'
