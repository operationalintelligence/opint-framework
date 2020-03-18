from django.urls import path, include

from rest_framework import routers

from opint_framework.apps.example_app.api.views import SampleViewSet

router = routers.DefaultRouter()
router.register(r'sample', SampleViewSet)

urlpatterns = [

    # Could be accessed as http://127.0.0.1:8000/example_app/api/
    path("home", SampleViewSet.index, name='sample-home'),
    path('', include(router.urls)),
]
