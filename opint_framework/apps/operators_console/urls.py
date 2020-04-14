from django.urls import path
from .api.views.views import index

urlpatterns = [
    path("", index, name='index'),
]
