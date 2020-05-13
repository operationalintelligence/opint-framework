# General activation flag
IS_ACTIVATED = False

#The deployment setting which activate/deactivate all agents executing in the app
ENABLE_SCHEDULER = True

# The time period in sec when each agent is called. It is delay between starting time
# Only once instance of each agent could be executed at the same time
#
POLLING_TIME = {
    "sample_agent": 1,
}

API_PREFIX = "ddm/api"

DATABASES = {
    'default': {},
    'rucio_opint': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': 'rucio_opint',
    }
}
