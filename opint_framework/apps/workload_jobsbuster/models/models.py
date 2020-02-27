from django.db import models
from opint_framework.core.models import IssueCategory, Action, Solution, ISSUE_STATUS
from opint_framework.core.models import Issue, IssueMetadata


class WorkflowIssue(models.Model):
    issue_id = models.BigIntegerField(primary_key=True, db_column='ISSUE_ID')
    observation_started = models.DateTimeField(db_column='OBSERVATION_STARTED')
    observation_finished = models.DateTimeField(db_column='OBSERVATION_FINISHED')
    walltime_loss = models.IntegerField(null=True, db_column='WALLTIME_LOSS')
    failures_counts = models.IntegerField(null=True, db_column='FAILURES_COUNTS')
    class Meta:
        db_table = u'ATLAS_JOBS_BUSTER_ISSUE'


class WorkflowIssueMetadata(IssueMetadata, models.Model):
    issue = models.ForeignKey(WorkflowIssue, on_delete=models.PROTECT)
    class Meta:
        db_table = u'ATLAS_JOBS_BUSTER_ISSUE_METADATA'


