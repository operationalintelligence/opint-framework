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
virtualenv -p python3 ~/environments/journal
source ~/environments/journal/bin/activate
```


Install Python dependencies:
```commandline
cd ~/projects/journal
pip install -r requirements.txt
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
