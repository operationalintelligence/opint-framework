from django.urls import path, include
from rest_framework import routers
from .api.views import SampleViewSet
import opint_framework.apps.example_app.api.views

urlpatterns = [

    # Could be accessed as http://127.0.0.1:8000/example_app/api/
    path("/", SampleViewSet.index, name='index'),
]

