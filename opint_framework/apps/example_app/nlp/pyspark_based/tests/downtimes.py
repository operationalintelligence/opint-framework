import datetime
from pyspark.sql import SparkSession
from pyspark.sql.types import StructField, StructType, StringType, TimestampType
from opint_framework.apps.example_app.nlp.pyspark_based.utils import *

spark = SparkSession.builder.master("local[*]").appName("test_downtimes_join").getOrCreate()
text = "\n###\tTesting join between FTS sample data and ATLAS CRIC api for downtimes\t###\n"
print("#"*len(text), text, "#"*len(text))

# create sample data
dwt_dict = [{'hostname': 'se.bfg.uni-freiburg.de',
             'start_time': datetime.datetime(2020, 12, 1, 9, 0, 0),
             'end_time': datetime.datetime(2020, 12, 2, 20, 0, 0)},
            {'hostname': 'se.bfg.uni-freiburg.de',
             'start_time': datetime.datetime(2021, 2, 1, 9, 0, 0),
             'end_time': datetime.datetime(2021, 2, 2, 20, 0, 0)},
            {'hostname': 'sedoor1.bfg.uni-freiburg.de',
             'start_time': datetime.datetime(2020, 12, 1, 9, 0, 0),
             'end_time': datetime.datetime(2020, 12, 1, 20, 0, 0)},
            {'hostname': 'lorienmaster.irb.hr',
             'start_time': datetime.datetime(2020, 11, 16, 17, 0, 0),
             'end_time': datetime.datetime(2020, 12, 15, 23, 0, 0)}]

schema = StructType([
    StructField('hostname', StringType(), False),
    StructField('start_time', TimestampType(), False),
    StructField('end_time', TimestampType(), False)
])
downtimes_df = spark.createDataFrame(dwt_dict, schema)

print("\n###\t-\tSample dump from ATLAS CRIC downtimes\n")
downtimes_df.show(truncate=False)

fts_dict = [{'tr_id': 'alfa-1',
             'src_hostname': 'se.bfg.uni-freiburg.de',
             'dst_hostname': 'storm.cnaf.infn.it',
             'tr_datetime_complete': datetime.datetime(2020, 12, 1, 13, 21, 0)},
            {'tr_id': 'alfa-2',
             'src_hostname': 'se.bfg.uni-freiburg.de',
             'dst_hostname': 'lorienmaster.irb.hr',
             'tr_datetime_complete': datetime.datetime(2020, 12, 1, 13, 21, 0)},
            {'tr_id': 'alfa-3',
             'src_hostname': 'dynafed-atlas.heprc.uvic.ca',
             'dst_hostname': 'storm.cnaf.infn.it',
             'tr_datetime_complete': datetime.datetime(2020, 12, 1, 13, 21, 0)},
            {'tr_id': 'alfa-4',
             'src_hostname': 'sedoor1.bfg.uni-freiburg.de',
             'dst_hostname': 'storm.cnaf.infn.it',
             'tr_datetime_complete': datetime.datetime(2020, 12, 1, 13, 21, 0)},
            ]

schema = StructType([
    StructField('tr_id', StringType(), False),
    StructField('src_hostname', StringType(), False),
    StructField('dst_hostname', StringType(), False),
    StructField('tr_datetime_complete', TimestampType(), False)
])
fts_df = spark.createDataFrame(fts_dict, schema)

print("\n###\t-\tSample FTS data\n")
fts_df.show(truncate=False)

#### test join
spark.conf.set("spark.sql.crossJoin.enabled", "true")

df = exclude_downtime(fts_df, spark)

print("###\t-\tJoin result should contain 1 row only, having dynafed-atlas as src_hostname\n")
df.show(truncate=False)

################################## CHECK EMPTY PROTOCOLS LIST IN CRIC DOWNTIMES
import json

cric_dump_path = "/home/luca/PycharmProjects/opint-framework/opint_framework/apps/example_app/nlp/pyspark_based/sample_data/wlcg-cric.cern.ch.json"
with open(cric_dump_path) as json_data:
    r = json.load(json_data)

prot_ddm = {}
prot_others = {}
for site, info in r.items():
    for n, entry in enumerate(info):
        if any([service in entry['affected_services'].lower() for service in ['webdav', 'xrootd', 'srm']]):
            prot_ddm[site] = {n: entry['protocols']}
        else:
            prot_others[site] = {n: entry['protocols']}

# how many sites in each group?
print("\nhow many sites in each group of downtimes?", f"\nstorage: {len(prot_ddm)},\tothers: {len(prot_others)}")  # 24 VS 14
# are some sites in both groups?
print("\nare some sites in both groups?", f"\nintersection: {set(prot_ddm.keys()).intersection(set(prot_others.keys()))}")  # --> apparently no
# are there empty protocols in ddm downtimes?
print("\nare there empty protocols in storage downtimes?")
for k, v in prot_ddm.items():
    if len(v[list(v.keys())[0]]) == 0:
        print(f"Site {k}:\n{v}\n")  # --> ATLAND, SAMPA, UPorto
# are there non-empty protocols in other downtimes?
print("\nare there non-empty protocols in other downtimes?")
check=0
for k, v in prot_others.items():
    if len(v[list(v.keys())[0]]) != 0:
        print(f"Site {k}:\n{v}\n\n")
        check += 1
if not check:
    print("All empty!")  # --> all empty
