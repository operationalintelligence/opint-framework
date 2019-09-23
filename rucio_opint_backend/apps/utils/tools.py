import datetime


def parse_date(date):
    for fmt in ('%Y-%m-%d', '%d.%m.%Y', '%d/%m/%Y'):
        try:
            return datetime.datetime.strptime(date, fmt)
        except ValueError:
            pass
    raise ValueError('no valid date format found')
