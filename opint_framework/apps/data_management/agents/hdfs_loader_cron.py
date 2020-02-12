import argparse
import datetime
import traceback

import requests
from pyspark.sql import SparkSession

from opint_framework.apps.data_management.utils import parse_date, get_hostname
from opint_framework.apps.data_management.utils.register import register_transfer_issue


class HDFSLoaderCron():
    help = 'Runs the HDFS fetching job'

    base_path = '/project/monitoring/archive/fts/raw/complete'

    def add_arguments(self, parser):
        parser.add_argument('-f', '--file', type=argparse.FileType('r'), help='File with files to be imported')
        parser.add_argument('-r', '--range', type=int, help='Range of days to be imported')
        parser.add_argument('-d', '--date', type=str, help='Day to be imported')

    def handle(self, *args, **options):
        print("Importing HDFS Data")
        self.populate(**options)

    def construct_path_for_date(self, date):
        return self.base_path + F'/{date.year}/{date.month:02d}/{date.day:02d}/'

    def pull_hdfs_dir(self, path, spark):
        try:
            fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(spark._jsc.hadoopConfiguration())
            list_status = fs.listStatus(spark._jvm.org.apache.hadoop.fs.Path(path))
            for file in [file.getPath().getName() for file in list_status]:
                self.pull_hdfs_json(path+file, spark)
        except Exception as e:
            print('Error listing files for ', path, e)

    def pull_hdfs_json(self, path, spark):
        try:
            res = spark.read.json(path)
            issues = self.resolve_issues(res)
            issues = self.resolve_sites(issues)
            for issue in issues:
                register_transfer_issue(issue)
        except Exception as e:
            print('Error loading data from', path, e)
            traceback.print_tb(e.__traceback__)

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

    def resolve_sites(self, issues):
        cric_url = "http://wlcg-cric.cern.ch/api/core/service/query/?json&type=SE"
        r = requests.get(url=cric_url).json()
        site_protocols = {}
        for site, info in r.items():
            for se in info:
                for name, prot in se.get('protocols', {}).items():
                    site_protocols.setdefault(get_hostname(prot['endpoint']), site)

        for issue in issues:
            if not issue.get('src_site'):
                issue['src_site'] = site_protocols.get(get_hostname(issue.pop('src_hostname')), '')
            if not issue.get('dst_site'):
                issue['dst_site'] = site_protocols.get(get_hostname(issue.pop('dst_hostname')), '')

        return issues

    def populate(self, **options):
        spark = SparkSession.builder.master("local[*]").appName("Issues").getOrCreate()
        if options.get('date'):
            path = self.construct_path_for_date(parse_date(options['date']))
            self.pull_hdfs_dir(path, spark)
        elif options.get('range'):
            today = datetime.datetime.today()
            date_list = [today - datetime.timedelta(days=x) for x in range(options['range'])]
            for date in date_list:
                path = self.construct_path_for_date(date)
                self.pull_hdfs_dir(path, spark)
        elif options.get('file'):
            for line in options.get('file'):
                self.pull_hdfs_json(line, spark)
        else:
            path = self.construct_path_for_date(datetime.datetime.today() - datetime.timedelta(days=1))
            self.pull_hdfs_dir(path, spark)
