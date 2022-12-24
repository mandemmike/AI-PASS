from django.contrib.auth.models import User

from django.contrib.auth import authenticate

from rest_framework import serializers


class RegisterSerializer(serializers.ModelSerializer):
    username = serializers.CharField(required=True, max_length=100)
    password = serializers.CharField(required=True, max_length=100)

    class Meta:
        model = User
        fields = ['username', 'password']

    def create(self, validated_data):
        user = User.objects.create(**validated_data)

        user.save()

        authenticate(username=validated_data['username'], password=validated_data['password'])

        if user is not None:
            print('user authenticated')

            return user

        else:

            ## handle backend not authenicated user

            return None


""""
    def validate(self, attrs):
        if attrs['password'] != attrs['password2']:
            raise serializers.ValidationError({"password": "Password fields didn't match."})

        return attrs
"""
