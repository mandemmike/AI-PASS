#!/bin/bash


python manage.py migrate --noinput
python manage.py loaddata fixtures/admin_user.json
exec "$@"



