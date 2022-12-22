from django import forms
from django.core.files.uploadedfile import InMemoryUploadedFile

from app.models import FaceRecognition, TrainingDatasetFile, MLModel


class FaceRecognitionForm(forms.ModelForm):
    class Meta:
        model = FaceRecognition
        fields = ['image']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['image'].widget.attrs.update({'class': 'form-control'})


class DataSetUploadForm(forms.ModelForm):

    def clean_file(self):
        file: InMemoryUploadedFile = self.cleaned_data['file']

        if file.content_type != 'application/zip':
            raise forms.ValidationError('Only zip files')
        return file

    class Meta:
        model = TrainingDatasetFile
        fields = ['file']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['file'].widget.attrs.update({'class': 'form-control'})


# class TrainedDatasetUploadForm(forms.ModelForm):
#     class Meta:
#         model = TrainedDataset
#         fields = ['file']


class ModelUploadForm(forms.ModelForm):
    class Meta:
        model = MLModel
        fields = ['id', 'name', 'format', 'file']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print(self.fields)
        self.fields['file'].widget.attrs.update({'class': 'form-control'})

class SelectModelForm(forms.Form):
    selected = forms.IntegerField(min_value=1)

    def save(self, request):
        print(self.cleaned_data, 'SelectForm')
        print(self.cleaned_data)
        selected_model_id = self.cleaned_data['selected']
        ml_model = MLModel.objects.get(id=selected_model_id)
        ml_model.is_active = True
        ml_model.save(update_fields=['is_active'])
        MLModel.objects.exclude(id=ml_model.id).update(is_active=False)
        print(ml_model)


class EvaluateModelForm(SelectModelForm):

    def save(self, request):
        print(self.cleaned_data, 'EvaluateForm')
        selected_model_id = self.cleaned_data['selected']
        ml_model = MLModel.objects.get(id=selected_model_id)
        print(ml_model)
