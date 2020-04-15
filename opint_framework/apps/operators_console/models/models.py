from django.db import models


class PredictionHistory(models.Model):
    """
    Workflow Prediction History model.
    """
    hid = models.BigAutoField(primary_key=True)
    name = models.CharField(max_length=255)
    good = models.FloatField()
    acdc = models.FloatField()
    resubmit = models.FloatField()
    timestamp = models.DateTimeField()


class LabelArchive(models.Model):
    """
    Labels Archive model.
    """
    name = models.CharField(max_length=255, unique=True, primary_key=True)
    label = models.IntegerField()


class DocsOneMonthArchive(models.Model):
    """
    Short-term Document Archive model.
    """
    name = models.CharField(max_length=255)
    document = models.TextField()
    timestamp = models.DateTimeField(auto_now=True)
