from django_filters import rest_framework as filters
from .models import FacePicture


# We create filters for each field we want to be able to filter on

class FacePictureFilter(filters.FilterSet):
    filename = filters.CharFilter(lookup_expr='icontains')
    age = filters.NumberFilter()
    created_at = filters.NumberFilter()
    year__gt = filters.NumberFilter(field_name='year', lookup_expr='gt')
    year__lt = filters.NumberFilter(field_name='year', lookup_expr='lt')
    creator__username = filters.CharFilter(lookup_expr='icontains')

    class Meta:
        model = FacePicture
        fields = ['filename', 'age', 'created_at', 'year__gt', 'year__lt', 'creator__username']

