import datetime
import requests

from opint_framework.core.scheduler import BaseAgent
from opint_framework.core.dataproviders import HDFSLoader
from opint_framework.apps.data_management.utils.tools import get_hostname
from opint_framework.apps.data_management.utils.register import register_transfer_issue


class FTSLoader(HDFSLoader):
    help = 'Runs the HDFS fetching job'
    base_path = '/project/monitoring/archive/fts/raw/complete'

    def construct_path_for_date(self, date):
        return self.base_path + F'/{date.year}/{date.month:02d}/{date.day:02d}/'

    @classmethod
    def resolve_issues(self, df):
        issues = df.filter(df.data.t_final_transfer_state_flag == 0) \
            .groupby(df.data.t__error_message.alias('t__error_message'),
                     df.data.src_hostname.alias('src_hostname'),
                     df.data.dst_hostname.alias('dst_hostname'),
                     df.data.dst_site_name.alias('dst_site_name'),
                     df.data.src_site_name.alias('src_site_name'),
                     df.data.tr_error_category.alias('tr_error_category')) \
            .count() \
            .collect()
        objs = []
        for issue in issues:
            issue_obj = {
                'message': issue['t__error_message'],
                'amount': issue['count'],
                'dst_site': issue['dst_site_name'],
                'src_site': issue['src_site_name'],
                'src_hostname': issue['src_hostname'],
                'dst_hostname': issue['dst_hostname'],
                # 'category': issue['tr_error_category'],
                'type': 'transfer-error'
            }
            objs.append(issue_obj)
        return objs

    @classmethod
    def resolve_sites(self, issues):
        cric_url = "http://wlcg-cric.cern.ch/api/core/service/query/?json&type=SE"
        r = requests.get(url=cric_url).json()
        site_protocols = {}
        for site, info in r.items():Couldn't read data from source.
            for se in info:
                for name, prot in se.get('protocols', {}).items():
                    site_protocols.setdefault(get_hostname(prot['endpoint']), site)

        for issue in issues:
            if not issue.get('src_site'):
                issue['src_site'] = site_protocols.get(get_hostname(issue.pop('src_hostname')), '')
            if not issue.get('dst_site'):
                issue['dst_site'] = site_protocols.get(get_hostname(issue.pop('dst_hostname')), '')

        return issues

    @classmethod
    def translate_data(self, data, **kwargs):
        """
        Translates the data from the source format to any desired format.
        Can be overwritten by parent.

        :return: (data translated to desired format)
        """
        data = self.resolve_issues(data)
        data = self.resolve_sites(data)
        return data

    def register_data(self):
        now = datetime.datetime.now()
        path = self.construct_path_for_date(now)
        data = self.load_data(type='JSON', spark_master='local[*]', spark_name='Issues', path=path)
        for entry in data:
            register_transfer_issue(entry)
