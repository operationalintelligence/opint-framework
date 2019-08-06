FROM python:3.6
ENV PYTHONUNBUFFERED 1
RUN apt-get update && apt-get -y install cron vim python-dev default-libmysqlclient-dev python3-dev
RUN mkdir /code
RUN mkdir -p /code/migrations/core
RUN touch /code/migrations/core/__init__.py
WORKDIR /code
ADD requirements.txt /code/
RUN pip install -r requirements.txt
COPY . /code/
ENTRYPOINT /code/init.sh