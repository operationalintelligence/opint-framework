import json
from rest_framework import status
from rest_framework.reverse import reverse
from rest_framework.test import APITestCase

from opint_framework.apps.example_app.models import Sample


class SampleViewSetTestCase(APITestCase):
    list_url = reverse('sample-list')

    def test_home_page(self):
        home_url = reverse('sample-home')

        res = self.client.get(home_url)
        self.assertEquals(res.status_code, status.HTTP_200_OK)
        self.assertEquals(json.loads(res.content)['Result'], 'OK')

    def test_list_samples(self):
        res = self.client.get(self.list_url, format='json')
        self.assertEquals(res.status_code, status.HTTP_200_OK)

    def test_create_samples(self):
        data = {'sample_message': 'TEST'}
        res = self.client.post(self.list_url, data, format='json')

        self.assertEquals(res.status_code, status.HTTP_201_CREATED)
        self.assertEquals(Sample.objects.count(), 1)
        self.assertEquals(Sample.objects.first().sample_message, 'TEST')

    def test_list_samples_populated(self):
        Sample.objects.create(sample_message='TEST')

        res = self.client.get(self.list_url)

        self.assertEquals(res.status_code, status.HTTP_200_OK)
        self.assertEquals(len(res.data['results']), 1)

    def test_sample_detail(self):
        sample = Sample.objects.create(sample_message='TEST')

        detail_url = reverse('sample-detail', args=[sample.pk])
        res = self.client.get(detail_url)

        self.assertEquals(res.status_code, status.HTTP_200_OK)

    def test_sample_detail_invalid(self):
        detail_url = reverse('sample-detail', args=[9999])
        res = self.client.get(detail_url)

        self.assertEquals(res.status_code, status.HTTP_404_NOT_FOUND)

    def test_delete_sample(self):
        sample = Sample.objects.create(sample_message='TEST')
        detail_url = reverse('sample-detail', args=[sample.pk])

        res = self.client.delete(detail_url, format='json')

        self.assertEquals(res.status_code, status.HTTP_204_NO_CONTENT)
        self.assertEquals(Sample.objects.count(), 0)
