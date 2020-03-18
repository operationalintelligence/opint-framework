from rest_framework import serializers
from opint_framework.apps.example_app.models import Sample


class SampleSerializer(serializers.ModelSerializer):
    class Meta:
        model = Sample
        fields = ['id', 'sample_message']