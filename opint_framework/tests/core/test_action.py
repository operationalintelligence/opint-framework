from datetime import datetime

from rest_framework import status
from rest_framework.reverse import reverse
from rest_framework.test import APITestCase

from opint_framework.core.models import Action
from opint_framework.apps.api.serializers import ActionSerializer


class ActionViewSetTestCase(APITestCase):
    list_url = reverse('action-list')

    def test_list_actions(self):
        res = self.client.get(self.list_url, format='json')
        self.assertEquals(res.status_code, status.HTTP_200_OK)

    def test_create_action(self):
        action_title = 'Test Action'
        data = {'action': action_title, 'last_modified': datetime.today()}
        res = self.client.post(self.list_url, data, format='json')

        ser = ActionSerializer(data=res.data)
        if ser.is_valid():
            action = Action(**ser.validated_data)

        self.assertEquals(res.status_code, status.HTTP_201_CREATED)
        self.assertEquals(action.action, action_title)

    def test_list_actions_populated(self):
        action_title = 'Test Action'
        action = Action(action=action_title, last_modified=datetime.today())
        action.save()

        res = self.client.get(self.list_url)
        ser = ActionSerializer(data=res.data)
        if ser.is_valid():
            action = Action(**ser.validated_data)

        self.assertEquals(res.status_code, status.HTTP_200_OK)
        self.assertEquals(action.action, action_title)
