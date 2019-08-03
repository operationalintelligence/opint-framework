from django.urls import path, include

from rest_framework import routers

from .views import IssueViewSet, IssueCauseViewSet, ActionViewSet, IssueCategoryViewSet, SolutionViewSet


router = routers.DefaultRouter()
router.register(r'issues', IssueViewSet)
router.register(r'actions', ActionViewSet)
router.register(r'issuecauses', IssueCauseViewSet)
router.register(r'issuecategories', IssueCategoryViewSet)
router.register(r'solutions', SolutionViewSet)

urlpatterns = [
    path('', include(router.urls)),
]
