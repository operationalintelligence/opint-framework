import traceback

from rest_auth.registration.views import SocialLoginView

from allauth.socialaccount.providers.oauth2.client import OAuth2Client, OAuth2Error
from allauth.socialaccount.providers.cern.views import CernOAuth2Adapter


class HandledOAuth2Client(OAuth2Client):

    def get_access_token(self, code):
        try:
            super().get_access_token(code)
        except OAuth2Error:
            print(traceback.format_exc())
            return {'error': 'OAuth2Client Error'}


class CERNLogin(SocialLoginView):
    adapter_class = CernOAuth2Adapter
    callback_url = 'https://rucio-opint-ui-dev.web.cern.ch/login/cern/success'
    client_class = HandledOAuth2Client
