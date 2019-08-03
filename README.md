[![Build Status](https://travis-ci.com/Panos512/rucio-opint-backend-django.svg?token=9WE9v7qSHUzbjJweTfrs&branch=master)](https://travis-ci.com/Panos512/rucio-opint-backend-django)

## Description

Rucio OpInt backend.

## Installation

Fork the repo into your personal project and clone the project.
```commandline
cd ~/projects/rucio-opint-frontend
git clone https://github.com/operationalintelligence/rucio-opint-backend
```

Create a new python3 virtual environment and activate it:
```commandline
virtualenv -p python3 ~/environments/rucio-opint-backend
source ~/environments/rucio-opint-backend/bin/activate
```


Install Python dependencies:
```commandline
cd ~/projects/rucio-opint-backend
pip install -r requirements.txt
```
Create in a directory your personal `settings_local.py` file to store sensitive information

settings_local.py template:
```python
"""
Local (host dependent) sensitive settings
"""
import os


DEBUG = True

BASE_DIR = os.path.realpath(os.path.dirname(__file__))  # location of this config file

config = {

    'SECRET_KEY': '',  # FIX ME TO UNIQUE VALUE
    'API_KEY': '',  # API key for elasticsearch queries
    'DATABASES': {
        'default': {
          'ENGINE': 'django.db.backends.sqlite3',
          'NAME': os.path.join(BASE_DIR, 'rucioopint.sqlite3'),  # same location as this config file
        }
    },
    'TMP_DIR': os.path.join(BASE_DIR, 'cache'),
    'MIGRATIONS_STORE_PATH': os.path.join(BASE_DIR, 'migrations')
}
```

Update `PYTHONPATH` to include path to your `settings_local.py` file
```commandline
export PYTHONPATH=/path_to_settings_local_directory/:${PYTHONPATH};
```

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
