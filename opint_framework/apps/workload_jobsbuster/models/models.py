from django.db import models
from opint_framework.apps.rucio_opint_backend.core.models import IssueCategory, Action, Solution, ISSUE_STATUS


class WorkflowIssue(models.Model):
    """
    Workflow Issue object.
    """

    message = models.CharField(max_length=1024)

    category = models.ForeignKey(IssueCategory, null=True, on_delete=models.PROTECT)
    action = models.ForeignKey(Action, null=True, verbose_name='Proposed Action', on_delete=models.SET_NULL)
    solution = models.ForeignKey(Solution, null=True, verbose_name='The solution given', on_delete=models.SET_NULL)
    status = models.CharField(max_length=12, choices=ISSUE_STATUS, default=ISSUE_STATUS.New)

    last_modified = models.DateTimeField(auto_now=True)

    def __str__(self):
        return str(self.id)


class WorkflowIssueMetadata(models.Model):
    """
    Key-value pairs for Issue metadata
    """
    issue = models.ForeignKey(WorkflowIssue, on_delete=models.PROTECT)
    key = models.CharField(max_length=512)
    value = models.CharField(max_length=512)

    class Meta:
        unique_together = (('issue', 'key', 'value'),)
