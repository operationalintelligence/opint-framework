"""
Django settings for rucio_opint_backend project.

Generated by 'django-admin startproject' using Django 2.2.3.

For more information on this file, see
https://docs.djangoproject.com/en/2.2/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/2.2/ref/settings/
"""

import os
import sys

MIGRATIONS_STORE_MODULE = 'migrations'
MIGRATIONS_STORE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load local configuration
try:
    from settings_local import config, DEBUG
    DEBUG = str(os.environ.get('DJANGO_DEBUG', DEBUG)).lower() in ['true', '1']
    vars().update(config)
except ImportError as e:
    msg = "File settings_local.py not found or incomplete.\nError: %s" % e
    print('sys.path=', sys.path, 'PWD=', os.getcwd())
    raise Exception(msg)

# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/2.2/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = 'i-cj+m#t+!rv6x4t1(2r^zt@@p4&x7pv)=of0xh-a6w&vs-e(1'

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = []


# Application definition

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework',
    'filters',
    'corsheaders',
    'rucio_opint_backend.apps.core',
    'rucio_opint_backend.apps.api',
    'rucio_opint_backend.apps.crons',
    'rucio_opint_backend.apps.utils'
]

MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'rucio_opint_backend.apps.core.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'rucio_opint_backend.apps.core.wsgi.application'

# Password validation
# https://docs.djangoproject.com/en/2.2/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]


# Internationalization
# https://docs.djangoproject.com/en/2.2/topics/i18n/

LANGUAGE_CODE = 'en-us'

# Local time zone for this installation. Choices can be found here:
# http://en.wikipedia.org/wiki/List_of_tz_zones_by_name
TIME_ZONE = 'Europe/Zurich'

USE_I18N = True

USE_L10N = True

USE_TZ = True


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/2.2/howto/static-files/

STATIC_URL = '/static/'

MIGRATION_MODULES_LIST = ['core']
MIGRATION_MODULES = {}
MIGRATION_MODULES.update(dict([k, '%s.%s' % (MIGRATIONS_STORE_MODULE, k)] for k in MIGRATION_MODULES_LIST))

## check MIGRATIONS data dir
if '.' not in MIGRATIONS_STORE_MODULE and MIGRATIONS_STORE_MODULE:  ## create directory structure if need
    m = os.path.join(MIGRATIONS_STORE_PATH, MIGRATIONS_STORE_MODULE)
    if m and not os.path.exists(m):
        print(' ... prepare directory structure for MIGRATION files: MIGRATIONS_STORE_MODULE=%s, MIGRATIONS_STORE_PATH=%s'
              % (MIGRATIONS_STORE_MODULE, MIGRATIONS_STORE_PATH))
        os.makedirs(m, 0o755)
    pp = os.path.join(m, '__init__.py')
    if not os.path.exists(pp):
        with open(pp, 'a'):
            os.utime(pp, None)
        del m, pp

# RestFramework config
REST_FRAMEWORK = {
    'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.PageNumberPagination',
    'PAGE_SIZE': 100
}


# CORS config
CORS_ORIGIN_ALLOW_ALL = False
# Allow react dev server to query
CORS_ORIGIN_WHITELIST = ['http://localhost:8080']
