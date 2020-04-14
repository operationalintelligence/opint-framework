from django.db import models


class SampleModel(models.Model):
    """
    Sample Model.
    """
    sample_message = models.CharField(max_length=1024)
