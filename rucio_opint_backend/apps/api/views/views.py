from rest_framework import viewsets, filters
from filters.mixins import FiltersMixin

from rucio_opint_backend.apps.core.models import Issue, IssueCause, Action, IssueCategory, Solution

from rucio_opint_backend.apps.api.serializers import (IssueSerializer, IssueCauseSerializer, ActionSerializer,
                                                      IssueCategorySerializer, SolutionSerializer)
from rucio_opint_backend.apps.api.validations import issue_query_schema


class IssueViewSet(FiltersMixin, viewsets.ModelViewSet):
    """
    API endpoint that allows Issues to be viewed or edited.
    """
    queryset = Issue.objects.all()
    serializer_class = IssueSerializer
    filter_backends = (filters.OrderingFilter,)
    ordering_fields = ('id', 'last_modified', 'src_site', 'dst_site')
    ordering = ('id',)
    filter_mappings = {
        'id': 'id',
        'message': 'message__icontains',
        'categories': 'category'
    }
    filter_validation_schema = issue_query_schema


class IssueCauseViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows Issues to be viewed or edited.
    """
    queryset = IssueCause.objects.all()
    serializer_class = IssueCauseSerializer


class ActionViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows Issues to be viewed or edited.
    """
    queryset = Action.objects.all()
    serializer_class = ActionSerializer


class IssueCategoryViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows Issues to be viewed or edited.
    """
    queryset = IssueCategory.objects.all().order_by('-last_modified')
    serializer_class = IssueCategorySerializer


class SolutionViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows Issues to be viewed or edited.
    """
    queryset = Solution.objects.all()
    serializer_class = SolutionSerializer
