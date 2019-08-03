from rest_framework import viewsets

from rucio_opint_backend.apps.core.models import Issue, IssueCause, Action

from rucio_opint_backend.apps.api.serializers import IssueSerializer, IssueCauseSerializer, ActionSerializer


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