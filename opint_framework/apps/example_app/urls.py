from django.urls import path
from .api.views import SampleViewSet

urlpatterns = [

    # Could be accessed as http://127.0.0.1:8000/example_app/api/
    path("", SampleViewSet.index, name='index'),
]
