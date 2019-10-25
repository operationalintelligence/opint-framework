FROM python:3.6
ENV PYTHONUNBUFFERED 1
ENV APP_ROOT /usr/src/app
RUN apt-get update && apt-get -y install cron vim python-dev default-libmysqlclient-dev python3-dev
WORKDIR ${APP_ROOT}
ADD requirements.txt .
RUN pip install -r requirements.txt