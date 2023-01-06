from django.contrib.auth import authenticate, login
from django.contrib.auth.models import User
from django.shortcuts import HttpResponseRedirect
from rest_framework import generics
from rest_framework.permissions import AllowAny

from .serializers import RegisterSerializer


def registerUser(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = User.objects.create(username=username, password=password)
        user.set_password(password)
        user.save()

        authenticate(username=username, password=password)
        login(request, user)

        if user is not None:
            print('user authenticated and created')
            next = request.POST.get('next', '/')
            return HttpResponseRedirect(next)

        else:
            print('Failed to create user')
            return None




class RegisterView(generics.CreateAPIView):
    queryset = User.objects.all()
    permission_classes = (AllowAny,)
    serializer_class = RegisterSerializer


    def redirect(request):
        next = request.POST.get('next', '/')
        return HttpResponseRedirect(next)




