from rest_framework import serializers
# from opint_framework.apps.workload_jobsbuster.models import WorkflowIssue, WorkflowIssueMetadata
from opint_framework.apps.data_management.models import TransferIssue


# class WorkflowIssueMetadataSerializer(serializers.ModelSerializer):
#     class Meta:
#         model = WorkflowIssueMetadata
#         fields = ['id', 'key', 'value']


class TransferIssueSerializer(serializers.ModelSerializer):
    class Meta:
        model = TransferIssue
        fields = ['id', 'message', 'src_site', 'dst_site', 'category', 'amount', 'type', 'status', 'last_modified']


# class WorkflowIssueSerializer(serializers.ModelSerializer):
#     metadata = WorkflowIssueMetadataSerializer(read_only=True, many=True, source='workflowissuemetadata_set')
#
#     class Meta:
#         model = WorkflowIssue
#         fields = ['id', 'message', 'category', 'status', 'metadata', 'last_modified']
