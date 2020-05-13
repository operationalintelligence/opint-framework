from django.urls import path, include

from rest_framework import routers

from opint_framework.core.api.views import (ActionViewSet, IssueCategoryViewSet,
                                            SolutionViewSet)

router = routers.DefaultRouter()
router.register(r'actions', ActionViewSet)
# router.register(r'issuecauses', IssueCauseViewSet)
router.register(r'issuecategories', IssueCategoryViewSet)
router.register(r'solutions', SolutionViewSet)

urlpatterns = [
    path('', include(router.urls)),
]
