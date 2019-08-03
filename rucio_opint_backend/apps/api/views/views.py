from rest_framework import viewsets

from rucio_opint_backend.apps.core.models import Issue

from rucio_opint_backend.apps.api.serializers import IssueSerializer


class IssueViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows Issues to be viewed or edited.
    """
    queryset = Issue.objects.all().order_by('-last_modified')
    serializer_class = IssueSerializer