from django.db import models

from .models import Issue


class WorkflowIssue(Issue):
    """
    Workflow Issue object.
    """

    workflow = models.CharField(max_length=128)

    class Meta:
        unique_together = (('message', 'workflow', 'type'), )
