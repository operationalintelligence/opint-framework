from rest_framework import viewsets

import json
from django.http import HttpResponse
from opint_framework.core.utils.common import DateTimeEncoder

class SampleViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows SampleModel to be viewed or edited.
    """
    def index(request):
        return HttpResponse(json.dumps({"Result":"OK"}, cls=DateTimeEncoder), content_type='text/html')
