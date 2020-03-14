import opint_framework.apps
import os
from importlib import util as importutil
import json
from datetime import datetime


def getURLStoFromApps():
    """
    :param modules:
    :return: returns URLS modules of all active apps to add to the central configuration
    """
    appsUrlsFiles = {}
    modulesWithURLs = getActiveAppSetting(['API_PREFIX'])
    for moduleName, urlConf in modulesWithURLs.items():
        if urlConf['API_PREFIX']:
            appsUrlsFiles[urlConf['API_PREFIX']] = "opint_framework.apps." + moduleName + ".urls"
    return appsUrlsFiles


def getAgentsShedule():
    """
    Determines active agents within list of provided applications
    :return: Dict of active agents modules names - period of scheduling in secs
    """
    modulesToSchedule = {}
    agentsSchedule = getActiveAppSetting(['ENABLE_SCHEDULER', 'POLLING_TIME'])
    for moduleName, pollingConf in agentsSchedule.items():
        if pollingConf['ENABLE_SCHEDULER']:
            for agentName, time in pollingConf['POLLING_TIME'].items():
                modulesToSchedule["opint_framework.apps." + moduleName + '.agents.' + agentName] = time
    return modulesToSchedule


def getActiveAppSetting(settingsToScan):
    modulesSettings = {}
    appsDirName = os.path.dirname(opint_framework.apps.__file__)
    for modulename in os.listdir(appsDirName):
        # We check existence of the settings file
        if not os.path.isfile(appsDirName + '/' + modulename + '/conf/settings.py'):
            continue
        modulespec = importutil.find_spec("opint_framework.apps."+modulename+'.conf.settings')
        if modulespec:
            app_conf = importutil.module_from_spec(modulespec)
            modulespec.loader.exec_module(app_conf)
            moduleSettings = {}
            if app_conf.IS_ACTIVATED:
                for setting in settingsToScan:
                    if hasattr(app_conf, setting):
                        moduleSettings[setting] = getattr(app_conf, setting)
                    else:
                        moduleSettings[setting] = None
                modulesSettings[modulename] = moduleSettings
    return modulesSettings


def getDataBasesForActiveApps():
    databases = {}
    activeAppsSettings = getActiveAppSetting(['DATABASES'])
    for app, settings in activeAppsSettings.items():
        if settings['DATABASES']:
            databases.update(settings['DATABASES'])
    return databases


class DateTimeEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, datetime):
            return o.isoformat()


def freeze(d):
    if isinstance(d, dict):
        return frozenset((key, freeze(value)) for key, value in d.items())
    elif isinstance(d, list):
        return tuple(freeze(value) for value in d)
    return d
