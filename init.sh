#!/bin/bash
export PYTHONPATH=${APP_ROOT}:${PYTHONPATH}
mkdir -p ${APP_ROOT}/migrations/core
touch ${APP_ROOT}/migrations/core/__init__.py
python manage.py makemigrations core
python manage.py migrate
python manage.py loaddata initial
echo "0 * * * * root /usr/bin/python3 /code/manage.py rucio_loader_cron >> /code/rucio_loader_output" >> /etc/crontab
service cron start
echo "from django.contrib.auth.models import User; User.objects.create_superuser('admin', 'admin@example.com', '${ADMIN_PASSWORD}')" | python manage.py shell
python3 manage.py runserver 0.0.0.0:8080



