from opint_framework.apps.example_app.nlp.pyspark_based.pyspark_nlp_adapter import pysparkNLPAdapter
from opint_framework.core.dataproviders.hdfs_provider import HDFSLoader

data_path = """/home/luca/PycharmProjects/opint-framework/opint_framework/apps/example_app/nlp/pyspark_based/sample_data/dump_1dec2020/*.json"""

# instantiate NLPAdapter object
pipeline = pysparkNLPAdapter(path_list=[data_path], vo=None, filter_T3=True,  # data
                             tks_col="stop_token_1",  # tokenization
                             w2v_model_path="results/w2v", w2v_mode="train", w2v_save_mode="overwrite",
                             emb_size=3, win_size=8, min_count=1, tks_vec="message_vector",  # word2vec
                             ft_col="features", kmeans_model_path="results/kmeans", kmeans_mode="train",
                             pred_mode="static", new_cluster_thresh=None, k_list=[2, 4],
                             distance="cosine", opt_initSteps=10, opt_tol=0.01, opt_maxIter=10,
                             log_path=None, n_cores=4,  # K_optim
                             tr_initSteps=30, tr_tol=0.001, tr_maxIter=30,  # train_kmeans
                             clust_col="prediction", wrdcld=True, timeplot=True)

pipeline.context['dataset'] = HDFSLoader().pull_hdfs_json(pipeline.context['path_list'], pipeline.context['spark'])

# retrieve just data
all_transfers = pipeline.context['dataset'].select("data.*")

# filter test_errors only
test_errors = all_transfers.filter(all_transfers["t_final_transfer_state_flag"] == 0)
if pipeline.context['vo'] is not None:
    test_errors = test_errors.filter(test_errors["vo"] == pipeline.context['vo'])

# add row id and select only relevant variables
test_errors = test_errors.select(f"{pipeline.context['id_col']}", "t__error_message", "src_hostname",
                                 "dst_hostname", f"{pipeline.context['timestamp_tr_x']}")

from opint_framework.apps.example_app.nlp.pyspark_based.utils import convert_timestamp_to_datetime, \
    convert_endpoint_to_site, add_tier_level, get_hostname

test_errors = convert_timestamp_to_datetime(test_errors)
test_errors = convert_endpoint_to_site(test_errors)
test_errors = add_tier_level(test_errors)

if pipeline.context['filter_T3']:
    test_errors = test_errors.filter((~test_errors.src_hostname.contains("TIER3")
                                      & ~test_errors.dst_hostname.contains("TIER3")))

df = test_errors.toPandas()
hostnames_FTS = set(df.src_hostname)
hostnames_FTS = hostnames_FTS.union(set(df.dst_hostname))
print(f"Found {len(hostnames_FTS)} different hostnames in FTS data.")

# retrieve downtimes from cric dump
import json
import datetime

cric_dump_path = "/home/luca/PycharmProjects/opint-framework/opint_framework/apps/example_app/nlp/pyspark_based/sample_data/wlcg-cric.cern.ch.json"
with open(cric_dump_path) as json_data:
    r = json.load(json_data)
host_downtime = []
for site, info in r.items():
    for entry in info:
        if entry['severity'].lower() == 'outage':
            if any([service in entry['affected_services'].lower() for service in ['webdav', 'xrootd', 'srm']]):
                for prot in entry['protocols']:
                    host_downtime.append(
                        {'hostname': get_hostname(prot['endpoint'], site),
                         'start_time': datetime.datetime.fromisoformat(entry['start_time']),
                         'end_time': datetime.datetime.fromisoformat(entry['end_time'])})
                if not entry['protocols']:
                    host_downtime.append(
                        {'hostname': site,
                         'start_time': datetime.datetime.fromisoformat(entry['start_time']),
                         'end_time': datetime.datetime.fromisoformat(entry['end_time'])})
# host_downtime.append({'hostname': 'pic', 'start_time': datetime.datetime.strptime('2020-10-20 22:50:00', '%Y-%m-%d %H:%M:%S'),
#                        'end_time': datetime.datetime.strptime('2020-10-20 22:55:00', '%Y-%m-%d %H:%M:%S')})
from pyspark.sql.types import StructField, StructType, StringType, TimestampType

schema = StructType([
    StructField('hostname', StringType(), False),
    StructField('start_time', TimestampType(), False),
    StructField('end_time', TimestampType(), False)
])

downtimes_df = pipeline.context["spark"].createDataFrame(host_downtime, schema).distinct()

cric_downtimes = set([entry["hostname"] for entry in host_downtime])
exclude_list = list(cric_downtimes.intersection(hostnames_FTS))
len(exclude_list)

import pyspark.sql.functions as F
pipeline.context['spark'].conf.set("spark.sql.crossJoin.enabled", "true")

df = test_errors.join(downtimes_df, test_errors.src_hostname == downtimes_df.hostname,
                      how='left')  # .join(downtimes_df, test_errors.dst_hostname==downtimes_df.hostname, how='left')
df = df.alias('l').join(downtimes_df.alias('r'), df.dst_hostname == downtimes_df.hostname, how='left').select(
    'tr_id', 't__error_message', 'src_hostname', 'dst_hostname', 'tr_datetime_complete', 'src_rcsite', 'dst_rcsite',
    'src_level', 'dst_level',
    F.when(~df.hostname.isNull(), df.hostname).otherwise(downtimes_df.hostname).alias("hostname"),
    F.when((df.start_time <= downtimes_df.start_time), df.start_time).otherwise(downtimes_df.start_time).alias(
        "start_time"),
    F.when((df.end_time >= downtimes_df.end_time), df.end_time).otherwise(downtimes_df.end_time).alias("end_time")
)

a = df.filter(~df.hostname.isNull())
b = df.filter(~df.start_time.isNull())
c = df.filter(~df.end_time.isNull())
assert a.count() == b.count() == c.count()
df.filter(~df.hostname.isNull()).show(30)

for i in range(len(exclude_list)):
    df.filter(
        (df.src_hostname.isin(exclude_list[i]) | df.dst_hostname.isin(exclude_list[i])) &
        (df.tr_datetime_complete <= df.end_time) &
        (df.tr_datetime_complete >= df.start_time)
    ).show()
    # first filter: hostname blacklisted
    df.filter(df.hostname.isin(exclude_list[i])).count()
    df.filter(df.hostname.isin(exclude_list[i])).show()

    # second filter: after start time
    df.where(
        (df.tr_datetime_complete >= downtimes_df.start_time)
    ).count()
    df.where(
        (df.tr_datetime_complete >= downtimes_df.start_time)
    ).show()
    # third filter: before end time
    df.where(
        (df.tr_datetime_complete <= downtimes_df.end_time)
    ).count()
    df.where(
        (df.tr_datetime_complete <= downtimes_df.end_time)
    ).show()
    # combine datetime filter: between start and end time
    df.where(
        (df.tr_datetime_complete >= downtimes_df.start_time) & (df.tr_datetime_complete <= downtimes_df.end_time)
    ).count()
    df.where(
        (df.tr_datetime_complete >= downtimes_df.start_time) & (df.tr_datetime_complete <= downtimes_df.end_time)
    ).show()



################################## CHECK EMPTY PROTOCOLS LIST IN CRIC DOWNTIMES

prot_ddm = {}
prot_others = {}
for site, info in r.items():
    for n, entry in enumerate(info):
        if any([service in entry['affected_services'].lower() for service in ['webdav', 'xrootd', 'srm']]):
            prot_ddm[site] = {n: entry['protocols']}
        else:
            prot_others[site] = {n: entry['protocols']}

# how many sites in each group?
len(prot_ddm), len(prot_others)  # 24 VS 14
# are some sites in both groups?
set(prot_ddm.keys()).intersection(set(prot_others.keys()))  # --> apparently no
# are there empty protocols in ddm downtimes?
for k, v in prot_ddm.items():
    print(f"Site {k}:\n{v}\n\n")  # --> ATLAND, SAMPA, UPorto
# are there non-empty protocols in other downtimes?
for k, v in prot_others.items():
    print(f"Site {k}:\n{v}\n\n")  # --> all empty
