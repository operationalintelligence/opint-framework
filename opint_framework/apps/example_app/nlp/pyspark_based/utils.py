
def convert_timestamp_to_datetime(df, tmstp_col="tr_timestamp_complete", dttm_col="tr_datetime_complete", out_fmt="yyyy-MM-dd HH:mm:ss"):
    """
    Convert timestamp column (nanoseconds, UTC) to datetime
    :param df: input pyspark dataframe
    :param tmstp_col: name of the timestamp (ns) column
    :param dttm_col: output name of the datetime column
    :param out_fmt: output datetim format
    :return: initial df with datetime column instead of timestamp
    """
    from pyspark.sql.functions import udf, to_timestamp
    import datetime

    # udf to convert the unix timestamp to datetime
    in_fmt = "%Y-%m-%d %H:%M:%S"
    get_timestamp = udf(lambda x: datetime.datetime.fromtimestamp(x / 1000.0).strftime(in_fmt))

    # apply this udf in the dataframe
    df = df.withColumn("datetime_str", get_timestamp(df[tmstp_col]))
    df = df.withColumn(dttm_col,to_timestamp(df['datetime_str'], out_fmt))
    df = df.select(df.columns[:-3] + [dttm_col])
    return df


def get_hostname(endpoint, default=''):
    """
    Extract hostname from the endpoint.
    Returns default if failed to extract.
    :return: hostname value
    """
    import re
    p = r'^(.*?://)?(?P<host>[\w.-]+).*'
    r = re.search(p, endpoint)

    return r.group('host') if r else default


def add_tier_level(dataset):
    """
    Attach tier level to the respective src/dst hostname.
    :return: dataset
    """
    import requests
    from pyspark.sql.functions import create_map, lit
    from itertools import chain

    # retrieve hostname-->rcsite mapping
    cric_url = "http://wlcg-cric.cern.ch/api/core/service/query/?json&type=SE"
    r = requests.get(url=cric_url).json()
    host_to_site_dict = {}
    for site, info in r.items():
        if 'protocols' in info:
            for name, prot in info.get('protocols', {}).items():
                host_to_site_dict.setdefault(get_hostname(prot['endpoint']), info.get('rcsite', site))

    # retrieve rcsite-->tier level mapping
    cric_url = "http://wlcg-cric.cern.ch/api/core/rcsite/query/?json"
    r = requests.get(url=cric_url).json()
    tier_levels = {site_name: site_values['rc_tier_level'] for site_name, site_values in r.items()}
    hostname_levels = {hostname: f"T{tier_levels.get(rcsite, None)}" for hostname, rcsite in host_to_site_dict.items()}

    mapping_expr = create_map([lit(x) for x in chain(*hostname_levels.items())])
    dataset = dataset.withColumn('src_level', mapping_expr[dataset['src_hostname']]) \
        .withColumn('dst_level', mapping_expr[dataset['dst_hostname']])
    return (dataset)


def convert_endpoint_to_site(dataset):
    """
    Attach site names to the respective src/dst hostname.
    :return: dataset
    """
    import requests
    from pyspark.sql.functions import create_map, lit
    from itertools import chain

    # retrieve hostname-->rcsite mapping
    cric_url = "http://wlcg-cric.cern.ch/api/core/service/query/?json&type=SE"
    r = requests.get(url=cric_url).json()
    host_to_site_dict = {}
    for site, info in r.items():
        if 'protocols' in info:
            for name, prot in info.get('protocols', {}).items():
                host_to_site_dict.setdefault(get_hostname(prot['endpoint']), info.get('rcsite', site))

    # apply mapping
    mapping_expr = create_map([lit(x) for x in chain(*host_to_site_dict.items())])
    dataset = dataset.withColumn('src_rcsite', mapping_expr[dataset['src_hostname']]) \
        .withColumn('dst_rcsite', mapping_expr[dataset['dst_hostname']])
    return (dataset)


def exclude_downtime(dataset, spark):
    """
    Retrieve sites in downtime from CRIC and filter them out from the dataset
    :return: dataset
    """
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
    downtimes_df = spark.createDataFrame(host_downtime, schema)

    dataset = dataset.join(downtimes_df, [
        (dataset.dst_hostname.isin(downtimes_df.hostname) | dataset.dst_hostname.isin(downtimes_df.hostname) |
         dataset.src_rcsite.isin(downtimes_df.hostname) | dataset.dst_rcsite.isin(downtimes_df.hostname)),
        dataset.tr_datetime_complete <= downtimes_df.end_time,
        dataset.tr_datetime_complete >= downtimes_df.start_time], how='inner')

    return (dataset)

