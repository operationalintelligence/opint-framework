import json
from datetime import datetime

from rest_framework import status
from rest_framework.reverse import reverse
from rest_framework.test import APITestCase

from opint_framework.core.models import Action, IssueCategory, Solution
from opint_framework.apps.api.serializers import SolutionSerializer


class ActionViewSetTestCase(APITestCase):
    list_url = reverse('solution-list')

    def test_list_solutions(self):
        res = self.client.get(self.list_url, format='json')
        self.assertEquals(res.status_code, status.HTTP_200_OK)

    def test_create_solution(self):
        action = Action.objects.create(action='Test Action')
        category = IssueCategory.objects.create(regex='REGEX')

        data = {'solution': action.id, 'category': category.id}
        res = self.client.post(self.list_url, data, format='json')

        self.assertEquals(Solution.objects.count(), 1)

        solution = Solution.objects.first()

        self.assertEquals(res.status_code, status.HTTP_201_CREATED)
        self.assertEquals(solution.solution.action, 'Test Action')
        self.assertEquals(solution.category.regex, 'REGEX')

    def test_list_solutions_populated(self):
        action = Action.objects.create(action='Test Action')
        category = IssueCategory.objects.create(regex='REGEX')

        solution = Solution.objects.create(solution=action, category=category)

        res = self.client.get(self.list_url)

        self.assertEquals(res.status_code, status.HTTP_200_OK)
        self.assertEquals(len(res.data['results']), 1)

    def test_solution_detail(self):
        action = Action.objects.create(action='Test Action')
        category = IssueCategory.objects.create(regex='REGEX')

        solution = Solution.objects.create(solution=action, category=category)

        detail_url = reverse('solution-detail', args=[solution.pk])
        res = self.client.get(detail_url)

        self.assertEquals(res.status_code, status.HTTP_200_OK)

    def test_action_detail_invalid(self):
        detail_url = reverse('solution-detail', args=[9999])
        res = self.client.get(detail_url)

        self.assertEquals(res.status_code, status.HTTP_404_NOT_FOUND)