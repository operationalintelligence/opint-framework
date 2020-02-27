from django.db import models
from opint_framework.core.models import IssueCategory, Action, Solution, ISSUE_STATUS
from opint_framework.core.models import Issue, IssueMetadata

class WorkflowIssue(Issue):
    class Meta:
        db_table = u'ATLAS_JOBS_BUSTER_ISSUE'


class WorkflowIssueMetadata(IssueMetadata):
    class Meta:
        db_table = u'ATLAS_JOBS_BUSTER_ISSUE_METADATA'
