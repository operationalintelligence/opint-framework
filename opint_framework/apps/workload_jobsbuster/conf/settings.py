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