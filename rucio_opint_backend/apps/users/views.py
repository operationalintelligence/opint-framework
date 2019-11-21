from rest_auth.registration.views import SocialLoginView

from allauth.socialaccount.providers.oauth2.client import OAuth2Client
from allauth.socialaccount.providers.cern.views import CernOAuth2Adapter


class CERNLogin(SocialLoginView):
    adapter_class = CernOAuth2Adapter
    callback_url = 'https://rucio-opint-ui-dev.web.cern.ch/login/cern/success'
    client_class = OAuth2Client
