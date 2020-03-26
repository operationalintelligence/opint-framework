import cx_Oracle
import pandas as pd
import opint_framework.apps.workload_jobsbuster.conf.settings as settings
import pytz
class Connection(cx_Oracle.Connection):
    def cursor(self):
        cursor = super(Connection, self).cursor()
        cursor.arraysize = 10000
        cursor.execute("alter session set time_zone = 'UTC'")
        return cursor

def retreiveData(datefrom, dateto):
    datefrom = datefrom.astimezone(pytz.utc).replace(tzinfo=None).replace(microsecond=0)
    dateto = dateto.astimezone(pytz.utc).replace(tzinfo=None).replace(microsecond=0)
    dbsettings = settings.DATABASES['jobs_buster_jobs']

    query = """
        SELECT  ja.PANDAID, ja.STARTTIME, ja.ENDTIME, ja.ATLASRELEASE, ja.ATTEMPTNR, ja.AVGPSS,
                ja.AVGRSS, ja.AVGSWAP, ja.AVGVMEM, ja.BROKERAGEERRORCODE, ja.CLOUD, ja.CMTCONFIG, ja.COMPUTINGELEMENT, ja.COMPUTINGSITE, 
                ja.ACTUALCORECOUNT, ja.CPUCONSUMPTIONTIME, ja.CPUCONSUMPTIONUNIT, ja.CREATIONHOST, ja.CREATIONTIME, 
                ja.CURRENTPRIORITY, ja.DDMERRORCODE, ja.DDMERRORDIAG, ja.DESTINATIONSE, ja.DESTINATIONSITE, ja.EVENTSERVICE, ja.PILOTID, 
                ja.PILOTTIMING, ja.PROCESSINGTYPE, ja.PRODUSERID, ja.PRODUSERNAME, ja.REQID, ja.RESOURCE_TYPE, ja.SCHEDULERID, 
                ja.SPECIALHANDLING, ja.EXEERRORCODE, ja.EXEERRORDIAG, ja.GSHARE, ja.HOMEPACKAGE, ja.HS06, ja.HS06SEC, 
                ja.INPUTFILEBYTES, ja.INPUTFILEPROJECT, ja.INPUTFILETYPE, ja.JEDITASKID, ja.JOBDISPATCHERERRORCODE, 
                ja.JOBDISPATCHERERRORDIAG, ja.JOBMETRICS, ja.JOBNAME, ja.JOBSTATUS, ja.NEVENTS, ja.NUCLEUS, 
                ja.PILOTERRORCODE, ja.PILOTERRORDIAG, ja.SUPERRORCODE, ja.SUPERRORDIAG, ja.TASKBUFFERERRORCODE, ja.TASKBUFFERERRORDIAG, 
                ja.TOTRBYTES, ja.TOTWBYTES, ja.TRANSEXITCODE, ja.TRANSFORMATION, ja.WORKINGGROUP, ja.NINPUTDATAFILES, ja.BATCHID,
                ja.BROKERAGEERRORDIAG
            FROM ATLAS_PANDA.JOBSARCHIVED4 ja
            WHERE ja.JOBSTATUS in ('failed','finished') and ja.PRODSOURCELABEL='managed' and not ja.endtime is null and
                (
                  (ja.endtime > TO_DATE('{0}', 'YYYY-MM-DD HH24:MI:SS'))
                OR 
                  (ja.STARTTIME > TO_DATE('{1}', 'YYYY-MM-DD HH24:MI:SS'))
                )
                and ja.modificationtime >  TO_DATE('{0}', 'YYYY-MM-DD HH24:MI:SS')
        """.format(datefrom, dateto)

    dsn_tns = cx_Oracle.makedsn(dbsettings['HOST'], dbsettings['PORT'], service_name=dbsettings['NAME'])
    with Connection(dsn=dsn_tns, user=dbsettings['USER'], password=dbsettings['PASSWORD'], threaded=True) as db:
        frame = pd.read_sql(query, db)
    return frame

