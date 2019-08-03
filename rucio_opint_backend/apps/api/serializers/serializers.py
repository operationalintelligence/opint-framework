from rest_framework import serializers
from rucio_opint_backend.apps.core.models import Issue


class IssueSerializer(serializers.ModelSerializer):
    class Meta:
        model = Issue
        fields = ['message', 'src_site', 'dst_site', 'category', 'amount', 'type', 'status', 'last_modified']
