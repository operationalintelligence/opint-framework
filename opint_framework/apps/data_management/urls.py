from django.urls import path, include

from rest_framework import routers

from opint_framework.apps.data_management.api.views import TransferIssueViewSet


router = routers.DefaultRouter()
router.register(r'issues/transfer', TransferIssueViewSet)
# router.register(r'issues/workflow', WorkflowIssueViewSet)

urlpatterns = [
    path('', include(router.urls)),
]
