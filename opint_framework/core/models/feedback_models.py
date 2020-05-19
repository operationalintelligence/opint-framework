from django.db import models


class Feedback(models.Model):
    """
    Feedback object.
    """

    model = models.CharField(max_length=512)
    object_pk = models.CharField(max_length=512)
    url = models.CharField(max_length=512, blank=True)
    comment = models.TextField(blank=True)
    extra_data = models.TextField(blank=True)
    created = models.DateTimeField(auto_now=True)

