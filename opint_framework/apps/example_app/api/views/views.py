from rest_framework import viewsets
from rest_framework.response import Response

class SampleViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows SampleModel to be viewed or edited.
    """
    def index(request):
        return Response({"Result":"OK"})

