from rest_framework import viewsets, filters
from filters.mixins import FiltersMixin

from opint_framework.apps.data_management.models import TransferIssue

from opint_framework.apps.data_management.api.serializers import TransferIssueSerializer
from opint_framework.apps.data_management.api import issue_query_schema


class TransferIssueViewSet(FiltersMixin, viewsets.ModelViewSet):
    """
    API endpoint that allows Issues to be viewed or edited.
    """
    queryset = TransferIssue.objects.all()
    serializer_class = TransferIssueSerializer
    filter_backends = (filters.OrderingFilter,)
    ordering_fields = ('id', 'last_modified', 'src_site', 'dst_site')
    ordering = ('id',)
    filter_mappings = {
        'id': 'id',
        'message': 'message__icontains',
        'categories': 'category'
    }
    filter_validation_schema = issue_query_schema


# class WorkflowIssueViewSet(FiltersMixin, viewsets.ModelViewSet):
#     """
#     API endpoint that allows Issues to be viewed or edited.
#     """
#     queryset = WorkflowIssue.objects.all()
#     serializer_class = WorkflowIssueSerializer
#     filter_backends = (filters.OrderingFilter,)
#     ordering_fields = ('id', 'last_modified')
#     ordering = ('id',)
#     filter_mappings = {
#         'id': 'id',
#         'message': 'message__icontains',
#         'categories': 'category'
#     }
#     filter_validation_schema = issue_query_schema


# class IssueCauseViewSet(viewsets.ModelViewSet):
#     """
#     API endpoint that allows IssueCauses to be viewed or edited.
#     """
#     queryset = IssueCause.objects.all()
#     serializer_class = IssueCauseSerializer
