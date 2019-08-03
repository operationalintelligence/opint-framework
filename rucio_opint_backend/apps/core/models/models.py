from django.db import models
from model_utils import Choices

ISSUE_STATUS = Choices('New', 'Ongoing', 'Resolved')
SITE_OPTIONS = Choices('src_site', 'dst_site', 'unknown')


class Issue(models.Model):
    """
    Rucio Issue object.
    """

    message = models.CharField(max_length=512)
    src_site = models.CharField(max_length=128)
    dst_site = models.CharField(max_length=128)

    category = models.ForeignKey(IssueCategory)

    amount = models.IntegerField(null=True, default=0)
    type = models.CharField(max_length=128)
    status = models.CharField(max_length=12, choices=ISSUE_STATUS, default=ISSUE_STATUS.New)

    last_modified = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = (('message', 'src_site', 'dst_site'), )


class IssueCategory(models.Model):
    """
    Rucio IssueCategory object.
    """

    amount = models.IntegerField(null=True, default=0)
    regex = models.CharField(max_length=512)
    cause = models.ForeignKey(IssueCause)
    last_modified = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = (('regex', 'reason'),)


class IssueCause(models.Model):
    """
    Rucio IssueCause object.
    """

    cause = models.CharField(max_length=128, unique=True)
    last_modified = models.DateTimeField(auto_now=True)

class Action(models.Model):
    """
    Action object.
    """

    action = models.CharField(max_length=128, unique=True)
    last_modified = models.DateTimeField(auto_now=True)


class Solution(models.Model):
    """
    Solution object.
    """

    category = models.ForeignKey(IssueCategory)
    proposed_action = models.ForeignKey(Action)
    solution = models.ForeignKey(Action)
    real_cause = models.ForeignKey(IssueCause)
    propability = models.FloatField(default=0, blank=True)
    score = models.BooleanField(default=False)

    affected_site = models.CharField(max_length=12, choices=SITE_OPTIONS, default=SITE_OPTIONS.unknown)
    last_modified = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = (('category', 'solution', 'real_cause', 'affected_site'),)  # TODO: This needs to be verified
