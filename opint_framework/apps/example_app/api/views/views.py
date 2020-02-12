from rest_framework import viewsets

from opint_framework.apps.example_app.api.serializers import SampleSerializer
from opint_framework.apps.example_app.models import SampleModel


class SampleViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows SampleModel to be viewed or edited.
    """
    queryset = SampleModel.objects.all()
    serializer_class = SampleSerializer
