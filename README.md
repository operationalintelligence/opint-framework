[![Build Status](https://travis-ci.com/operationalintelligence/rucio-opint-backend.svg?branch=master)](https://travis-ci.com/operationalintelligence/rucio-opint-backend)

## Description

Rucio OpInt backend.

## Installation

Fork the repo into your personal project and clone the project.
```commandline
cd ~/projects/rucio-opint/
git clone https://github.com/operationalintelligence/rucio-opint-backend
```

Create a new python3 virtual environment and activate it:
```commandline
virtualenv -p python3 ~/environments/rucio-opint-backend
source ~/environments/rucio-opint-backend/bin/activate
```


Install Python dependencies:
```commandline
cd ~/projects/rucio-opint/rucio-opint-backend
pip install -r requirements.txt
``` 

Export settings module:
```commandline
export DJANGO_SETTINGS_MODULE='rucio_opint_backend.apps.core.settings'
```

The following enviromental variables can be set:
API_KEY: The key for Monit Grafana's API
DB_PASS: The pass for the produciton database.
For Development you can enable sqlite from `rucio_opint_backend/apps/core/settings.py` 

Create DB:
```commandline
python manage.py makemigrations core
python manage.py migrate
```

Populate DB with initial data:
```commandline
python manage.py loaddata initial
```

## Running the app:
Navigate to the project's directory:
```commandline
cd ~/projects/rucio-opint-frontend
```
Run the django server:
```commandline
python manage.py runserver
```
