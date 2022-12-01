from django.urls import path
from . import views


urlpatterns = [
    path('', views.ListCreateMovieAPIView.as_view(), name=''),
    path('<int:pk>/', views.RetrieveUpdateDestroyMovieAPIView.as_view(), name=''),
]