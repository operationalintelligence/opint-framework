from django.db import models
from opint_framework.core.models import Issue


class TransferIssue(Issue):
    """
    Transfer Issue object.
    """
    src_site = models.CharField(max_length=128, blank=True)
    dst_site = models.CharField(max_length=128, blank=True)

    class Meta:
        unique_together = (('src_site', 'dst_site'), )
