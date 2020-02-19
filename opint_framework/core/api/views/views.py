from rest_framework import viewsets

from opint_framework.core.models import Action, IssueCategory, Solution
from opint_framework.core.api.serializers import (ActionSerializer, IssueCategorySerializer,
                                                  SolutionSerializer)


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
