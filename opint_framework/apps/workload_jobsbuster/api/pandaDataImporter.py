import cx_Oracle
import pandas as pd
import opint_framework.apps.workload_jobsbuster.conf.settings as settings
import os

class Connection(cx_Oracle.Connection):
    def cursor(self):
        cursor = super(Connection, self).cursor()
        cursor.arraysize = 10000
        cursor.execute("alter session set time_zone = 'UTC'")
        return cursor

def retreiveData(datefrom, dateto):
    os.chdir('/opt/oracle')

    query = """
        SELECT /*+ INDEX_RS_ASC(ja JOBS_JEDITASKID_PANDAID_IDX) */ ja.STARTTIME, ja.ENDTIME, ja.ATLASRELEASE, ja.ATTEMPTNR, ja.AVGPSS, 
            ja.AVGRSS, ja.AVGSWAP, ja.AVGVMEM, ja.BROKERAGEERRORCODE, ja.CLOUD, ja.CMTCONFIG, hh.COMPUTINGELEMENT, ja.COMPUTINGSITE, 
            ja.ACTUALCORECOUNT, ja.CPUCONSUMPTIONTIME, ja.CPUCONSUMPTIONUNIT, ja.CREATIONHOST, ja.CREATIONTIME, 
            ja.CURRENTPRIORITY, ja.DDMERRORCODE, ja.DDMERRORDIAG, ja.DESTINATIONSE, ja.DESTINATIONSITE, ja.EVENTSERVICE, ja.PILOTID, 
            ja.PILOTTIMING, ja.PROCESSINGTYPE, ja.PRODUSERID, ja.PRODUSERNAME, ja.REQID, ja.RESOURCE_TYPE, ja.SCHEDULERID, 
            ja.SPECIALHANDLING, ja.EXEERRORCODE, ja.EXEERRORDIAG, ja.GSHARE, ja.HOMEPACKAGE, ja.HS06, ja.HS06SEC, 
            ja.INPUTFILEBYTES, ja.INPUTFILEPROJECT, ja.INPUTFILETYPE, ja.JEDITASKID, ja.JOBDISPATCHERERRORCODE, 
            ja.JOBDISPATCHERERRORDIAG, ja.JOBMETRICS, ja.JOBNAME, ja.JOBSTATUS, ja.NEVENTS, ja.NUCLEUS, 
            ja.PILOTERRORCODE, ja.PILOTERRORDIAG, ja.SUPERRORCODE, ja.SUPERRORDIAG, ja.TASKBUFFERERRORCODE, ja.TASKBUFFERERRORDIAG, 
            ja.TOTRBYTES, ja.TOTWBYTES, ja.TRANSEXITCODE, ja.TRANSFORMATION, ja.WORKINGGROUP, ja.NINPUTDATAFILES, ja.BATCHID
        FROM ATLAS_PANDA.JOBSARCHIVED4 ja 
        JOIN ATLAS_PANDA.jedi_tasks ta 
        ON ta.jeditaskid=ja.jeditaskid  and ja.JOBSTATUS in ('failed','finished') and ta.tasktype='prod' and not ja.endtime is null and
            (
                  (ja.endtime > TO_DATE('{0}', 'YYYY-MM-DD HH24:MI:SS') and  ja.endtime < TO_DATE('{1}', 'YYYY-MM-DD HH24:MI:SS')) 
            )
        LEFT JOIN (
            SELECT hw.computingelement, hj.pandaid, hj.workerid, hj.harvesterid 
            from ATLAS_PANDA.harvester_workers hw, ATLAS_PANDA.harvester_rel_jobs_workers hj
            where hj.harvesterid = hw.harvesterid and hw.workerid = hj.workerid
        ) hh   
        on ja.pandaid  = hh.pandaid 
    """.format(datefrom, dateto)

    dsn_tns = cx_Oracle.makedsn(settings.DB_HOST, settings.DB_PORT, service_name=settings.DB_SERV_NAME)
    with Connection(dsn=dsn_tns, user=settings.DB_USER, password=settings.DB_PASS, threaded=True) as db:
        frame = pd.read_sql(query, db)
    return frame
