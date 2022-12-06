from django.contrib import admin

from app.models import FaceRecognition, TrainingDatasetFile, TrainedDataset

# Register your models here.

admin.site.register(FaceRecognition)


@admin.register(TrainingDatasetFile)
class TrainingDatasetFile(admin.ModelAdmin):
    list_display = ('file', 'id', 'dataset')

@admin.register(TrainedDataset)
class TrainedDataset(admin.ModelAdmin):
    pass



