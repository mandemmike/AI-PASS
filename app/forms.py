from django import forms
from django.core.files.uploadedfile import InMemoryUploadedFile

from app.models import FaceRecognition, TrainingDatasetFile


class FaceRecognitionForm(forms.ModelForm):

    class Meta:
        model = FaceRecognition
        fields = ['image']


class DataSetUploadForm(forms.ModelForm):

    def clean_file(self):
        file: InMemoryUploadedFile = self.cleaned_data['file']

        if file.content_type != 'application/zip':
            raise forms.ValidationError('Only zip files')
        return file



    class Meta:
        model = TrainingDatasetFile
        fields = ['file']
