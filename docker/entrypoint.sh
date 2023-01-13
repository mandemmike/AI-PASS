#!/bin/bash
python manage.py migrate --noinput --run-syncdb 
python manage.py loaddata fixtures/admin_user.json
exec "$@"
