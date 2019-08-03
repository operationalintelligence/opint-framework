from django.urls import path, include

from rest_framework import routers

from .views import IssueViewSet


router = routers.DefaultRouter()
router.register(r'issues', IssueViewSet)

urlpatterns = [
    path('', include(router.urls)),
]
