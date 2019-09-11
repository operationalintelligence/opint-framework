import argparse
import datetime

from pyspark.sql import SparkSession

from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = 'Runs the HDFS fetching job'

    def parse_date(self, date):
        for fmt in ('%Y-%m-%d', '%d.%m.%Y', '%d/%m/%Y'):
            try:
                return datetime.datetime.strptime(date, fmt)
            except ValueError:
                pass
        raise ValueError('no valid date format found')

    def add_arguments(self, parser):
        parser.add_argument('-f', '--file', type=argparse.FileType('r'), help='File with files to be imported')
        parser.add_argument('-r', '--range', type=int, help='Range of days to be imported')
        parser.add_argument('-d', '--date', type=str, help='Day to be imported')

    def handle(self, *args, **options):
        print("Importing HDFS Data")
        self.populate(**options)

    def populate(self, **options):
        spark = SparkSession.builder.master("local[*]").appName("Issues").getOrCreate()
        print(spark)
        if options.get('date'):
            date = self.parse_date(options['date'])
            print('Will read data for', date)
        if options.get('range'):
            today = datetime.datetime.today()
            date_list = [today - datetime.timedelta(days=x) for x in range(options['range'])]
            print('will read data for ')
            for date in date_list:
                print(date)
        if options.get('file'):
            for line in options.get('file'):
                print('wIll rad data from', line)
