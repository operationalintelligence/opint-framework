from pathlib import Path

# General activation flag
IS_ACTIVATED = False

# The time period in sec when each agent is called. It is delay between starting time
# Only once instance of each agent could be executed at the same time
#

POLLING_TIME = {
    "jobsanalyzer": 10,
}

#The deployment setting which activate/deactivate all agents executing in the app
ENABLE_SCHEDULER = True

API_PREFIX = "jobsbuster/api"


datafilespath = "/tmp/"

DB_JOBS_USER = ''
DB_JOBS_PASS = ''
DB_PERS_USER = ''
DB_PERS_PASS = ''

db_settings = str(Path.home())+"/private/db_settings.py"
try:
    exec(open(db_settings).read())
except FileNotFoundError:
    print(f'!WARNING!: JobsBuster {db_settings} does not exist.')

DATABASES = {
    'default': {},
    'jobs_buster_jobs': {
        'ENGINE': 'django.db.backends.oracle',
        'HOST': 'adcr-s.cern.ch',
        'PORT': '10121',
        'NAME': 'adcr_pandamon.cern.ch',
        'USER': DB_JOBS_USER, # Defined in private settings
        'PASSWORD': DB_JOBS_PASS # Defined in private settings
    },
    'jobs_buster_persistency': {
        'ENGINE': 'django.db.backends.oracle',
        'NAME': 'int8r1-v.cern.ch:10121/int8r.cern.ch',
        'USER': DB_PERS_USER,
        'PASSWORD': DB_PERS_PASS
    }
}