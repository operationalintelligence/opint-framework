import argparse
import datetime
import requests

from pyspark.sql import SparkSession

from django.core.management.base import BaseCommand

from rucio_opint_backend.apps.utils.tools import parse_date, get_hostname
from rucio_opint_backend.apps.utils.register import register_transfer_issue


class Command(BaseCommand):
    help = 'Runs the HDFS fetching job'

    base_path = '/project/monitoring/archive/rucio/raw/events'

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
            self.register_issues(issues)
        except Exception as e:
            print('Error loading data from', path, e)

    def resolve_issues(self, df):
        issues = df.filter(df.data.event_type.isin(['transfer-failed', 'deletion-failed'])) \
            .groupby(df.data.reason.alias('reason'),
                     df.data.src_rse.alias('src_rse'),
                     df.data.dst_rse.alias('dst_rse'),
                     df.data.event_type.alias('event_type')) \
            .count() \
            .collect()
        return issues

    def register_issues(self, issues):
        issues = self.resolve_issues(issues)
        for issue in issues:
            issue_obj = {
                'message': issue['reason'],
                'amount': issue['count'],
                'dst_site': issue['dst_rse'].split('_')[0] if issue.dst_rse else '',
                'src_site': issue['src_rse'].split('_')[0] if issue.src_rse else '',
                'type': issue['event_type']
            }
            register_transfer_issue(issue_obj)

    def resolve_sites(self, issues):
        cric_url = "http://wlcg-cric.cern.ch/api/core/service/query/?json&type=SE"
        r = requests.get(url=cric_url).json()
        site_protocols = {}
        for site, info in r.iteritems():
            for name, prot in info.get('protocols', {}).iteritems():
                site_protocols.setdefault(get_hostname(prot['endpoint']), site)

        for issue in issues:
            if not issue['src_site']:
                issue['src_site'] = site_protocols.get(get_hostname(issue['src_endpoint']))
            if not issue['dst_site']:
                issue['dst_site'] = site_protocols.get(get_hostname(issue['dst_endpoint']))

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
