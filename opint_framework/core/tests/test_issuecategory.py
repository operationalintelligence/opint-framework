from rest_framework import status
from rest_framework.reverse import reverse
from rest_framework.test import APITestCase

from opint_framework.core.models import IssueCategory


class IssueCategoryViewSetTestCase(APITestCase):
    list_url = reverse('issuecategory-list')

    def test_list_issuecategories(self):
        res = self.client.get(self.list_url, format='json')
        self.assertEquals(res.status_code, status.HTTP_200_OK)

    def test_create_issuecategory(self):
        data = {'regex': 'REGEX'}
        res = self.client.post(self.list_url, data, format='json')

        self.assertEquals(res.status_code, status.HTTP_201_CREATED)
        self.assertEquals(IssueCategory.objects.count(), 1)
        self.assertEquals(IssueCategory.objects.first().regex, 'REGEX')

    def test_list_issuecategory_populated(self):
        IssueCategory.objects.create(regex='REGEX')

        res = self.client.get(self.list_url)

        self.assertEquals(res.status_code, status.HTTP_200_OK)
        self.assertEquals(len(res.data['results']), 1)

    def test_issuecategory_detail(self):
        category = IssueCategory.objects.create(regex='REGEX')

        detail_url = reverse('issuecategory-detail', args=[category.pk])
        res = self.client.get(detail_url)

        self.assertEquals(res.status_code, status.HTTP_200_OK)

    def test_issuecategory_detail_invalid(self):
        detail_url = reverse('issuecategory-detail', args=[9999])
        res = self.client.get(detail_url)

        self.assertEquals(res.status_code, status.HTTP_404_NOT_FOUND)

    def test_delete_issuecategory(self):
        category = IssueCategory.objects.create(regex='REGEX')
        detail_url = reverse('issuecategory-detail', args=[category.pk])

        res = self.client.delete(detail_url, format='json')

        self.assertEquals(res.status_code, status.HTTP_204_NO_CONTENT)
        self.assertEquals(IssueCategory.objects.count(), 0)
