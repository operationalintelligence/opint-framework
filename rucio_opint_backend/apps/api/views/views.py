from rest_framework import viewsets

from rucio_opint_backend.apps.core.models import Issue, IssueCause, Action, IssueCategory, Solution

from rucio_opint_backend.apps.api.serializers import (IssueSerializer, IssueCauseSerializer, ActionSerializer,
                                                      IssueCategorySerializer, SolutionSerializer)


class IssueViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows Issues to be viewed or edited.
    """
    queryset = Issue.objects.all().order_by('-last_modified')
    serializer_class = IssueSerializer


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
