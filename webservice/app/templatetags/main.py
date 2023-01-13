from datetime import datetime

from django import template
from django.conf import settings

register = template.Library()


@register.simple_tag
def project_title() -> str:
    return settings.PROJECT_TITLE


@register.filter(name='date_time')
def date(value: str):
    """2021-04-11T18:02:37.066850Z"""
    time = datetime.strptime(value, '%Y-%m-%dT%H:%M:%S.%fZ')
    # time = datetime.strptime(value, '%Y-%m-%dT%H:%M:%S.%f%z')
    return time.strftime('%b %dth, %Y')
