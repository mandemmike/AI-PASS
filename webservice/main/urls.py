from django.contrib import admin
from django.urls import path
from app import views
from django.conf import settings
from django.conf.urls.static import static
from authentication import views as v2

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.IndexView.as_view(), name='index'),
    path('admin-ui/', views.DatasetUploadView.as_view(), name='admin_ui'),
    path('login/', views.LoginUser, name='login'),
    path('register/', v2.registerUser, name='register'),
    path('logout/', views.logoutUser, name='logout')

]
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
