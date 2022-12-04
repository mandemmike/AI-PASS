from rest_framework import serializers
from .models import FacePicture
from django.contrib.auth.models import User


class FacePictureSerializer(serializers.ModelSerializer):  # create class to serializer model
    creator = serializers.ReadOnlyField(source='creator.username')

    class Meta:
        model = FacePicture
        fields = ['filename', 'age', 'created_at', 'year__gt', 'year__lt', 'creator__username']


class UserSerializer(serializers.ModelSerializer):  # create class to serializer user model
    movies = serializers.PrimaryKeyRelatedField(many=True, queryset=FacePicture.objects.all())

    class Meta:
        model = User
        fields = ('id', 'username', 'FacePicture')
