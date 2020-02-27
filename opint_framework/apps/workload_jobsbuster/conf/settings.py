from pathlib import Path

# General activation flag
IS_ACTIVATED = True

# The time period in sec when each agent is called. It is delay between starting time
# Only once instance of each agent could be executed at the same time
#

POLLING_TIME = {
    "jobscollector": 1,
}

#The deployment setting which activate/deactivate all agents executing in the app
ENABLE_SCHEDULER = False

API_PREFIX = "jobsbuster/api"


datafilespath = "/tmp"

exec(open(str(Path.home())+"/private/db_settings.py").read())

DATABASES = {
    'default': {},
    'jobs_buster_jobs': {
        'ENGINE': 'django.db.backends.oracle',
        'HOST': 'localhost',
        'PORT': '10011',
        'NAME': 'adcr.cern.ch',
        'USER': DB_JOBS_USER, # Defined in private settings
        'PASSWORD': DB_JOBS_PASS # Defined in private settings
    },
    'jobs_buster_persistency': {
        'NAME': 'customer_data',
        'ENGINE': 'django.db.backends.oracle',
        'USER': 'cust',
        'PASSWORD': 'veryPriv@ate'
    }
}

