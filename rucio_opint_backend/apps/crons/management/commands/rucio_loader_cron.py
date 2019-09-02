import time
import requests

from django.core.management.base import BaseCommand
from django.conf import settings

from rucio_opint_backend.apps.core.models import Issue, IssueCategory
from rucio_opint_backend.apps.utils.categorizer import categorize_issue


class Command(BaseCommand):
    help = 'Runs the Rucio fetching job'

    # function to retrieve the specific error message between two sites and with the event type
    def get_details(self, dst_site, src_site, event_type, size=1):
        end_time = int(time.time() * 1000)
        event = event_type + '-failed'
        start_time = end_time - 1 * 3600 * 1000
        data = '{"search_type":"query_then_fetch",' \
               '"ignore_unavailable":true,' \
               '"index":["monit_prod_ddm_enr_transfer_*"]}\n' \
               '{"size":0,"query":{"bool":{"filter":[{"range":{"metadata.timestamp":{"gte":"%i","lte":"%i","format":"epoch_millis"}}},' \
               '{"query_string":{"analyze_wildcard":true,"query":"data.event_type: %s AND data.dst_experiment_site:(\\"%s\\") AND' \
               'data.src_experiment_site:(\\"%s\\")"}}]}},"aggs":{"2":{"terms":{"field":"data.reason","size":%i,' \
               '"order":{"_count":"desc"},"min_doc_count":1},"aggs":{}}}}\n' % (start_time, end_time, event, dst_site, src_site, size)
        return data

    # function to retrieve inefficient transfers between sites, or ineffecient deletion on a destination site.
    def read_efficiency(self, activity, **kwargs):
        URL = 'https://monit-grafana.cern.ch/api/datasources/proxy/7730/query?db=monit_production_ddm_transfers' \
              '&q=SELECT%20sum(files_done),sum(files_total)%20FROM%20%22raw%22.%2F%5Eddm_transfer%24%2F%20WHERE%20%22' \
              'state%22%20%3D%20%27{}%27%20AND%20time%20%3E%3D%20now()%20-%201h%20GROUP%20BY%20%22' \
              'src_experiment_site%22%2C%20%22dst_experiment_site%22&epoch=ms'.format(activity)
        r = requests.get(URL, headers=kwargs.get('header'))
        result = []
        if r.status_code == 200:
            response = r.json()
            print response
            for link in response['results'][0]['series']:
                if link['values'][0][2] > 200:
                    if (100 * (link['values'][0][1] / link['values'][0][2])) < 20:
                        result.append(list(link.items()))
            return result
        else:
            print('could not read efficiency')

    # function that populates the error table with errors that cause a significant inefficiency between certain sites,
    # this function should be run as a cron job and should run once every hour
    def populate(self):
        for activity in ['transfer', 'deletion']:
            header = {'Authorization': 'Bearer %s' % getattr(settings, 'API_KEY', None)}
            r = self.read_efficiency(activity, header=header)
            for link in r:
                # following condition is a quick fix for when the read efficiency response is unordered.
                try:
                    isinstance(link[1][1][0][1], int)
                    sites = 3
                except KeyError:
                    sites = 1
                except IndexError:
                    sites = 3

                data = self.get_details(link[sites][1]['dst_experiment_site'], link[sites][1]['src_experiment_site'], activity)
                try:
                    r = requests.post('https://monit-grafana.cern.ch/api/datasources/proxy/8736/_msearch',
                                      headers=header, data=data)
                except Exception as e:
                    print('WARNING: error while fetching data', e)

                resp = r.json()
                if not resp['responses'][0]['aggregations']['2']['buckets']:
                    continue

                issue = {
                    'message': resp['responses'][0]['aggregations']['2']['buckets'][0]['key'],
                    'amount': resp['responses'][0]['aggregations']['2']['buckets'][0]['doc_count'],
                    'dst_site': link[sites][1]['dst_experiment_site'],
                    'src_site': link[sites][1]['src_experiment_site'],
                    'type': activity + '-failure'
                }
                self.register_issue(issue)

    def register_issue(self, issue):
        print("INFO: registering issue ", issue)
        obj, created = Issue.objects.get_or_create(message=issue.pop('message'),
                                                   src_site=issue.pop('src_site'),
                                                   dst_site=issue.pop('dst_site'),
                                                   type=issue.pop('type'),
                                                   defaults=issue)
        if not created:
            obj.last_modified = time.time()
            obj.save(update_fields=['last_modified'])
            # return
        category = categorize_issue(obj)
        print(obj)
        if not category:
            print("INFO: creating new category")
            category = IssueCategory(regex=obj.message, amount=obj.amount)
            category.save()
        else:
            print("INFO: assigning to existing category")
            category.amount += obj.amount
            category.save(update_fields=['amount'])

        obj.category = category
        obj.save(update_fields=['category'])

    def handle(self, *args, **options):
        print("Importing Rucio error data from monit-grafana")
        self.populate()
