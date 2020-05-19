from django.urls import path, include

from rest_framework import routers

# from opint_framework.core.api.views import (ActionViewSet, IssueCategoryViewSet,
#                                             SolutionViewSet)
from opint_framework.core.api.views import FeedbackViewSet

router = routers.DefaultRouter()
# router.register(r'actions', ActionViewSet)
# router.register(r'issuecauses', IssueCauseViewSet)
# router.register(r'issuecategories', IssueCategoryViewSet)
# router.register(r'solutions', SolutionViewSet)
router.register(r'feedback', FeedbackViewSet)

urlpatterns = [
    path('', include(router.urls)),
]
