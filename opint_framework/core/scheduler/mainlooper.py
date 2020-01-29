import os
import logging.config
import schedule
import threading
import time
import inspect

from django.conf import settings
import opint_framework.apps
from importlib import util as importutil

# This is the main framework scheduler which controls main flow

if settings.LOG_PATH:
    logging.basicConfig(level=logging.DEBUG, filename=settings.LOG_PATH, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
else:
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


# This function provides dict of modules to execute
def scanAvailableApps():

    modulesToSchedule = {}  #Pairs module names / poll time

    # We walk over installed apps in the top apps dir
    appsDirName = os.path.dirname(opint_framework.apps.__file__)
    for modulename in os.listdir(appsDirName):
        print(modulename)
        # We check existence of the settings file
        if not os.path.isfile(appsDirName + '/' + modulename + '/settings.py'):
            continue
        modulespec = importutil.find_spec("opint_framework.apps."+modulename+'.settings')
        if modulespec:
            app_conf = importutil.module_from_spec(modulespec)
            modulespec.loader.exec_module(app_conf)
            if hasattr(app_conf, 'ENABLE_SCHEDULER') and app_conf.IS_ACTIVATED and hasattr(app_conf, 'POLLING_TIME'):
                for agentname, polltime in app_conf.POLLING_TIME.items():
                    modulesToSchedule["opint_framework.apps."+modulename+'.agents.'+agentname] = polltime
    return modulesToSchedule


# Starts new thread to run agent
def run_threaded(job_func):
    job_thread = threading.Thread(target=job_func)
    job_thread.daemon = True
    job_thread.start()


def main():
    modulesToSchedule = scanAvailableApps()
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
