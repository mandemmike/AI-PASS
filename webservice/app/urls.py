from django.urls import path
from app import views
from .views import LoginUser
from django.urls import include, path
from django.conf.urls import url
from django.contrib import admin

app_name = 'main'

urlpatterns = [
    url('', views.index, name='index'),
    path('login', include('authentication.urls')),
    url('dataset/', views.AdminUI.as_view(), name='AdminUI'),
    path('admin/', admin.site.urls),

]
