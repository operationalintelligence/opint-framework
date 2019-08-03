from django.urls import path, include

from rest_framework import routers

from .views import IssueViewSet, IssueCauseViewSet, ActionViewSet


router = routers.DefaultRouter()
router.register(r'issues', IssueViewSet)
router.register(r'actions', ActionViewSet)
router.register(r'issuecauses', IssueCauseViewSet)

urlpatterns = [
    path('', include(router.urls)),
]
