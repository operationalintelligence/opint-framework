import logging.config
import schedule
import threading
import time
import inspect
from importlib import util as importutil

from django.conf import settings
from opint_framework.core.utils.common import getAgentsShedule

# This is the main framework scheduler which controls main flow

if settings.LOG_PATH:
    logging.basicConfig(level=logging.DEBUG, filename=settings.LOG_PATH, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
else:
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


# Starts new thread to run agent
def run_threaded(job_func):
    job_thread = threading.Thread(target=job_func)
    job_thread.daemon = True
    job_thread.start()


def main():
    modulesToSchedule = getAgentsShedule()
    logging.debug("The following agents found: {}".format(modulesToSchedule))

    for agentname, polltime in modulesToSchedule.items():
        modulespec = importutil.find_spec(agentname)
        app_conf = importutil.module_from_spec(modulespec)
        modulespec.loader.exec_module(app_conf)
        clsmembers = inspect.getmembers(app_conf, inspect.isclass)
        logging.debug("We scheduling the following agent class: {}".format(clsmembers[-1][0]))
        agent = clsmembers[-1][1]()
        schedule.every(polltime).seconds.do(run_threaded, agent.execute)

    while 1:
        schedule.run_pending()
        time.sleep(1)


if __name__ == "__main__":
    main()
