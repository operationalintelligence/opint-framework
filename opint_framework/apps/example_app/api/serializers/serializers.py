from rest_framework import serializers
from opint_framework.apps.example_app.models import SampleModel


class SampleSerializer(serializers.ModelSerializer):
    class Meta:
        model = SampleModel
        fields = ['id', 'sample_message']
