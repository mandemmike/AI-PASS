from django import forms
from django.core.files.uploadedfile import InMemoryUploadedFile

from app.models import FaceRecognition, TrainingDatasetFile, MLModel, CurrentModel, DatasetElementFormat


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


# class TrainedDatasetUploadForm(forms.ModelForm):
#     class Meta:
#         model = TrainedDataset
#         fields = ['file']


class ModelUploadForm(forms.ModelForm):
    class Meta:
        model = MLModel
        fields = "__all__"


class CurrentModelForm(forms.ModelForm):
    class Meta:
        model = CurrentModel
        fields = "__all__"

class DatasetElementForm(forms.ModelForm):
    class Meta:
        model = DatasetElementFormat
        fields = '__all__'
