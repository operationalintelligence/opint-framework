from django.urls import path, include
from rest_framework import routers
import opint_framework.apps.workload_jobsbuster.api.views.views as views
import opint_framework.apps.example_app.api.views


urlpatterns = [

    # Could be accessed as http://127.0.0.1:8000/example_app/api/
    path("", views.processTimeWindowData, name='index'),
]

