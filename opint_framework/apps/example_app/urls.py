from django.urls import path
from .api.views.views import index

urlpatterns = [
    # Could be accessed as http://127.0.0.1:8000/example_app/api/
    path("", index, name='index'),
]
