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
    Convert src/dst hostname to the respective site names.
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
