#The deployment setting which activate/deactivate all agents executing in the app
IS_ACTIVATED = True

# The time period in sec when each agent is called. It is delay between starting time
# Only once instance of each agent could be executed at the same time
POLLING_TIME = 60

# if there are multiple agents with different polling time we may define polling time separately for each agent
POLLING_TIME  = {
    "sample_agent": 60,
}
