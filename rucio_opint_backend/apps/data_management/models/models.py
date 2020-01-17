from django.db import models
from rucio_opint_backend.apps.core.models import IssueCategory, Action, Solution, ISSUE_STATUS


class TransferIssue(models.Model):
    """
    Transfer Issue object.
    """
    message = models.CharField(max_length=1024)

    category = models.ForeignKey(IssueCategory, null=True, on_delete=models.PROTECT)
    action = models.ForeignKey(Action, null=True, verbose_name='Proposed Action', on_delete=models.SET_NULL)
    solution = models.ForeignKey(Solution, null=True, verbose_name='The solution given', on_delete=models.SET_NULL)
    # amount = models.IntegerField(null=True, default=0)unt = models.IntegerField(null=True, default=0)
    type = models.CharField(max_length=128)
    status = models.CharField(max_length=12, choices=ISSUE_STATUS, default=ISSUE_STATUS.New)

    src_site = models.CharField(max_length=128, blank=True)
    dst_site = models.CharField(max_length=128, blank=True)

    last_modified = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = (('message', 'type', 'src_site', 'dst_site'), )
