from django.db import models
from opint_framework.core.models import IssueCategory, Action, Solution, Issue, ISSUE_STATUS


class TransferIssue(Issue):
    """
    Transfer Issue object.
    """

    action = models.ForeignKey(Action, null=True, verbose_name='Proposed Action', on_delete=models.SET_NULL)

    src_site = models.CharField(max_length=128, blank=True)
    dst_site = models.CharField(max_length=128, blank=True)

    class Meta:
        unique_together = (('message', 'type', 'src_site', 'dst_site'), )
