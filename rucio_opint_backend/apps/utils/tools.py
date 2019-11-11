import datetime
import re


def parse_date(date):
    for fmt in ('%Y-%m-%d', '%d.%m.%Y', '%d/%m/%Y'):
        try:
            return datetime.datetime.strptime(date, fmt)
        except ValueError:
            pass
    raise ValueError('no valid date format found')


def get_hostname(endpoint):
    """
    Extract hostname from the endpoint.
    Returns empty string if failed to extract.

    :return: hostname value
    """

    p = r'^(.*?://)?(?P<host>[\w.-]+).*'
    r = re.search(p, endpoint)

    return r.group('host') if r else ''
