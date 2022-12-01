from django.shortcuts import render
from django.views.generic import TemplateView
from django.contrib import admin
from django.urls import path
from django.views.generic import TemplateView
from django.urls import include, path
from rest_framework import generics
from rest_framework.permissions import IsAuthenticated
from django.http import HttpResponse
from rest_framework.response import Response
from django.http import HttpResponse
from django.contrib.auth.models import User
from rest_framework.decorators import api_view
from rest_framework.decorators import action
from django.http import request
from rest_framework.permissions import AllowAny
from django.contrib.auth.decorators import login_required

from .forms import loginForm

class basicStuff(TemplateView):
    permission_classes = [AllowAny]
    template_name = "login.html"



