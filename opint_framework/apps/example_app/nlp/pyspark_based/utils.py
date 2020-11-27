def get_hostname(endpoint):
    """
    Extract hostname from the endpoint.
    Returns empty string if failed to extract.
    :return: hostname value
    """
    import re
    p = r'^(.*?://)?(?P<host>[\w.-]+).*'
    r = re.search(p, endpoint)

    return r.group('host') if r else ''


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
        if "protocols" in info:
            for name, prot in info.get('protocols', {}).items():
                host_to_site_dict.setdefault(get_hostname(prot['endpoint']), info.get("rcsite", site))

    # retrieve rcsite-->tier level mapping
    cric_url = "http://wlcg-cric.cern.ch/api/core/rcsite/query/?json"
    r = requests.get(url=cric_url).json()
    tier_levels = {site_name: site_values["rc_tier_level"] for site_name, site_values in r.items()}
    hostname_levels = {hostname: f"T{tier_levels.get(rcsite, None)}" for hostname, rcsite in host_to_site_dict.items()}

    mapping_expr = create_map([lit(x) for x in chain(*hostname_levels.items())])
    dataset = dataset.withColumn("src_level", mapping_expr[dataset["src_hostname"]]) \
        .withColumn("dst_level", mapping_expr[dataset["dst_hostname"]])
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
        if "protocols" in info:
            for name, prot in info.get('protocols', {}).items():
                host_to_site_dict.setdefault(get_hostname(prot['endpoint']), info.get("rcsite", site))

    # apply mapping
    mapping_expr = create_map([lit(x) for x in chain(*host_to_site_dict.items())])
    dataset = dataset.withColumn("src_rcsite", mapping_expr[dataset["src_hostname"]]) \
        .withColumn("dst_rcsite", mapping_expr[dataset["dst_hostname"]])
    return (dataset)


def exclude_downtime(dataset, spark):
    """
    Retrieve sites in downtime from CRIC and filter them out from the dataset
    :return: dataset
    """
    import requests

    # retrieve downtime
    cric_url = "https://wlcg-cric.cern.ch/api/core/downtime/query/?json"
    # TODO: fix SSL certificate issue
    r = requests.get(url=cric_url, cert="/home/luca/Desktop/certificates/cert_CERN.pem", verify=False).json()
    sites_downtime = []
    for site, info in r.items():
        for entry, values in info.items():
            if values["severity"].lower() == "outage":
                if any([substring in values["affected_services"].lower() for service in ["webdav", "xrootd", "srm"]]):
                    sites_downtime.append(
                        {"site": site, "start_time": values["start_time"], "end_time": values["end_time"]}
                    downtimes_df = spark.createDataFrame(sites_downtime)
                    dataset = dataset.join(downtimes_df, (
                                dataset.src_rcsite.contains(downtimes_df.site) | dataset.dst_rcsite.contains(
                            downtimes_df.site))) & (dataset.tr_datetime_complete <= downtimes_df.end_time) & (
                                      dataset.tr_datetime_complete >= downtimes_df.start_time), how = 'left_anti')
                    # mapping_expr = create_map([lit("active") for x in chain(*hostname_levels.items())])
                    # dataset = dataset.withColumn("src_level", mapping_expr[dataset["src_hostname"]]) \
                    #     .withColumn("dst_level", mapping_expr[dataset["dst_hostname"]])
        return (dataset)
