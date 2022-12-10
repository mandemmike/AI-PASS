from django.contrib import admin

from app.models import FaceRecognition, TrainingDatasetFile, TrainedDataset, MLModel


@admin.register(FaceRecognition)
class FaceRecognitionAdmin(admin.ModelAdmin):
    pass


@admin.register(MLModel)
class MLModelAdmin(admin.ModelAdmin):
    list_display = ('name', 'format', 'is_active')


@admin.register(TrainingDatasetFile)
class TrainingDatasetFileAdmin(admin.ModelAdmin):
    list_display = ('file', 'id', 'dataset')


@admin.register(TrainedDataset)
class TrainedDatasetAdmin(admin.ModelAdmin):
    pass
