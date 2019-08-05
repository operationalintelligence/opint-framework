FROM python:3
ENV PYTHONUNBUFFERED 1
RUN apt-get update && apt-get -y install cron vim
RUN mkdir /code
WORKDIR /code
#ADD requirements.txt /code/
#RUN pip install -r requirements.txt
VOLUME /code
ENTRYPOINT /code/init.sh