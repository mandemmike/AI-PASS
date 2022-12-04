from django.contrib.auth.models import User
from rest_framework import generics
from rest_framework.permissions import AllowAny
from .serializers import RegisterSerializer
from django.contrib.auth import authenticate, login as auth_login
from django.http import HttpResponse, request
from django.shortcuts import render


class RegisterView(generics.CreateAPIView):
    queryset = User.objects.all()
    permission_classes = (AllowAny,)
    serializer_class = RegisterSerializer
    template_name = 'homepage.html'


    def redirect(request):
        return render(request, 'homepage.html', status=201)
   

