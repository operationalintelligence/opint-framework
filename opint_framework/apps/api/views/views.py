from rest_framework import viewsets, filters
from filters.mixins import FiltersMixin

from opint_framework.core.models import Action, IssueCategory, Solution
from opint_framework.apps.workload_jobsbuster.models import WorkflowIssue
from opint_framework.apps.data_management.models import TransferIssue

from opint_framework.apps.api.serializers import (TransferIssueSerializer, WorkflowIssueSerializer, ActionSerializer,
                                                  IssueCategorySerializer, SolutionSerializer)
from opint_framework.apps.api.validations import issue_query_schema


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


class WorkflowIssueViewSet(FiltersMixin, viewsets.ModelViewSet):
    """
    API endpoint that allows Issues to be viewed or edited.
    """
    queryset = WorkflowIssue.objects.all()
    serializer_class = WorkflowIssueSerializer
    filter_backends = (filters.OrderingFilter,)
    ordering_fields = ('id', 'last_modified')
    ordering = ('id',)
    filter_mappings = {
        'id': 'id',
        'message': 'message__icontains',
        'categories': 'category'
    }
    filter_validation_schema = issue_query_schema


# class IssueCauseViewSet(viewsets.ModelViewSet):
#     """
#     API endpoint that allows IssueCauses to be viewed or edited.
#     """
#     queryset = IssueCause.objects.all()
#     serializer_class = IssueCauseSerializer


class ActionViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows Actions to be viewed or edited.
    """
    queryset = Action.objects.all()
    serializer_class = ActionSerializer


class IssueCategoryViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows IssueCategories to be viewed or edited.
    """
    queryset = IssueCategory.objects.all().order_by('-last_modified')
    serializer_class = IssueCategorySerializer


class SolutionViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows Solutions to be viewed or edited.
    """
    queryset = Solution.objects.all()
    serializer_class = SolutionSerializer
