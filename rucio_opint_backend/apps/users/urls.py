from django.urls import path, include
from .views import CERNLogin


urlpatterns = [
    path('rest-auth/', include('rest_auth.urls')),
    path('rest-auth/registration/', include('rest_auth.registration.urls')),
    path('rest-auth/cern/', CERNLogin.as_view())
]
