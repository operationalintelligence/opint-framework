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


# Atlas settings
DB_HOST = 'localhost'
DB_PORT = '10011'
DB_SERV_NAME = 'adcr.cern.ch'

# To be imported from additional settings
DB_PASS = ''
DB_USER = ''

exec(open(str(Path.home())+"/private/db_settings.py").read())
datafilespath = "/tmp"

