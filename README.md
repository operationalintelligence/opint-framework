[![Build Status](https://travis-ci.com/operationalintelligence/rucio-opint-backend.svg?branch=master)](https://travis-ci.com/operationalintelligence/rucio-opint-backend)

## Description

OpInt Framework.

## Installation
There are 3 options for the Framework installation:
1. Minimal, required only for running Jupyter notebooks supplied with example applications.
2. Backend service intended for running agents. 
3. REST service for Web deployment

Any installation scenario starts from getting the source code:
```commandline
git clone https://github.com/operationalintelligence/opint-framework
```
Assuming that [Jupyter notebook is pre installed](https://jupyter.org/install) on the target system, notebooks could be 
ready to run. If a notebook needs extra dependency they could be installed in a standard manned depending on what 
package manager you are using (conda or pip).   

Create a new python3 virtual environment and activate it:
```commandline
python3 -m venv ./virtenv
source ./virtenv/bin/activate
```

Install the global (minimal) framework requirements:  
```commandline
pip install -r ./opint-framework/requirements.txt
``` 

_(optional)_ Depending on the usage scenario ones may want to install requirements for specific deployment and application intended to use, e.g:
```commandline
pip install -r ./opint-framework/opint_framework/apps/workload_jobsbuster/requirements.txt
```

The following two lines are needed to supply paths and settings to Django:
```commandline
export PYTHONPATH=$(pwd)/opint-framework
export DJANGO_SETTINGS_MODULE=opint_framework.conf.settings
```commandline

**At this point the agents could be executed:**
```commandline
python3 opint-framework/opint_framework/core/scheduler/mainlooper.py
``` 

If installation passed successfully, the following message will be displayed:
```commandline
Hello World from Sample Agent
``` 
and then FrameWork exits. To activate the infinite loop mode, when agents periodically called, 
the following parameter in the opint-framework/opint_framework/conf/settings.py
```commandline
DO_DEBUG_AGENTS = True
``` 

If the Operational Intelligence framework will be used for providing REST services, the Django dependencies should be also installed: 
```commandline
cd ~/projects/opint-framework/opint-framework
pip install -r requirements.txt
``` 

The following environmental variables can be set:
Run the django server:
```commandline
python3 opint-framework/manage.py runserver
```