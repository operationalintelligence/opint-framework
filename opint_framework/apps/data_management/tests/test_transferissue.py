from rest_framework import status
from rest_framework.reverse import reverse
from rest_framework.test import APITestCase

from opint_framework.core.models import Action, Solution, IssueCategory
from opint_framework.apps.data_management.models import TransferIssue


class TransferIssueViewSetTestCase(APITestCase):
    list_url = reverse('transferissue-list')

    def test_list_transferissues(self):
        res = self.client.get(self.list_url, format='json')
        self.assertEquals(res.status_code, status.HTTP_200_OK)

    def test_create_transferissue(self):
        action = Action.objects.create(action='Test Action')
        category = IssueCategory.objects.create(regex='REGEX')
        solution = Solution.objects.create(solution=action)
        data = {'category': category.id,
                'solution': solution.id,
                'action': action.id,
                'src_site': 'src',
                'dst_site': 'dst',
                'message': 'msg',
                'type': 't1',
                }

        res = self.client.post(self.list_url, data, format='json')

        self.assertEquals(res.status_code, status.HTTP_201_CREATED)
        self.assertEquals(TransferIssue.objects.count(), 1)

        ti = TransferIssue.objects.first()

        self.assertEquals(ti.action, action)
        self.assertEquals(ti.solution.solution, ti.action)
        self.assertEquals(ti.category, category)
        self.assertEquals(ti.src_site, 'src')

    def test_list_solutions_populated(self):
        action = Action.objects.create(action='Test Action')
        category = IssueCategory.objects.create(regex='REGEX')
        solution = Solution.objects.create(solution=action)

        TransferIssue.objects.create(solution=solution, action=action, category=category)

        res = self.client.get(self.list_url)

        self.assertEquals(res.status_code, status.HTTP_200_OK)
        self.assertEquals(len(res.data['results']), 1)

    def test_solution_detail(self):
        action = Action.objects.create(action='Test Action')
        category = IssueCategory.objects.create(regex='REGEX')
        solution = Solution.objects.create(solution=action)

        ti = TransferIssue.objects.create(solution=solution, action=action, category=category)

        detail_url = reverse('transferissue-detail', args=[ti.pk])
        res = self.client.get(detail_url)

        self.assertEquals(res.status_code, status.HTTP_200_OK)

    def test_action_detail_invalid(self):
        detail_url = reverse('transferissue-detail', args=[9999])
        res = self.client.get(detail_url)

        self.assertEquals(res.status_code, status.HTTP_404_NOT_FOUND)
