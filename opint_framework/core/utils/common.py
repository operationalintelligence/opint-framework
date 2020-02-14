import opint_framework.apps
import os
from importlib import util as importutil

def getURLStoFromApps(modules = None):
    """
    :param modules:
    :return: returns URLS modules of all active apps to add to the central configuration
    """
    appsUrlsFiles = {}
    if not modules: modules = scanActiveApps()
    appsDirName = os.path.dirname(opint_framework.apps.__file__)
    for modulename in modules:
        modulespec = importutil.find_spec("opint_framework.apps." + modulename + '.conf.settings')
        if modulespec:
            app_conf = importutil.module_from_spec(modulespec)
            modulespec.loader.exec_module(app_conf)
            if hasattr(app_conf, 'API_PREFIX') and app_conf.API_PREFIX:
                if os.path.isfile(appsDirName + '/' + modulename + '/urls.py'):
                    appsUrlsFiles[app_conf.API_PREFIX] = "opint_framework.apps."+modulename+".urls"
    return appsUrlsFiles


def getAgentsShedule(modules = None):
    """
    Determines active agents within list of provided applications
    :param modules: List of application names, if None scans
    :return: Dict of active agents modules names - period of scheduling in secs
    """
    if not modules: modules = scanActiveApps()
    modulesToSchedule = {}  #Pairs module names / poll time
    for module in modules:
        modulespec = importutil.find_spec("opint_framework.apps." + module + '.conf.settings')
        if modulespec:
            app_conf = importutil.module_from_spec(modulespec)
            modulespec.loader.exec_module(app_conf)
            if hasattr(app_conf, 'ENABLE_SCHEDULER') and app_conf.ENABLE_SCHEDULER and app_conf.IS_ACTIVATED and hasattr(app_conf, 'POLLING_TIME'):
                for agentname, polltime in app_conf.POLLING_TIME.items():
                    modulesToSchedule["opint_framework.apps." + module + '.agents.' + agentname] = polltime
    return modulesToSchedule


def scanActiveApps():
    """
    Scans all sub folders in the apps directory, checks configuration to evaluate active ones.
    :return: List of names of active applications
    """
    activeApps = []
    appsDirName = os.path.dirname(opint_framework.apps.__file__)
    for modulename in os.listdir(appsDirName):
        # We check existence of the settings file
        if not os.path.isfile(appsDirName + '/' + modulename + '/conf/settings.py'):
            continue
        modulespec = importutil.find_spec("opint_framework.apps."+modulename+'.conf.settings')
        if modulespec:
            app_conf = importutil.module_from_spec(modulespec)
            modulespec.loader.exec_module(app_conf)
            if app_conf.IS_ACTIVATED:
                activeApps.append(modulename)
    return activeApps

