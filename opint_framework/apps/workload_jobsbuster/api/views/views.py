from rest_framework.response import Response
from opint_framework.apps.workload_jobsbuster.models import WorkflowIssue, WorkflowIssueMetadata, AnalysisSessions
import pickle
import opint_framework.apps.workload_jobsbuster.conf.settings as settings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor, Pool, EFstrType
import tempfile
from rest_framework.decorators import api_view
import json
import datetime
import os
from django.utils import timezone

counter = 0

"""
API endpoint that allows SampleModel to be viewed or edited.
"""
@api_view(['GET'])
def processTimeWindowData(request):
    return Response({"Result":"OK"})

