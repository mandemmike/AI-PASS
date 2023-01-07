from django.contrib import admin

from app.models import FaceRecognition, TrainingDatasetFile, TrainedDataset, MLModel, EvaluatedModelData, Dataset

@admin.register(EvaluatedModelData)
class EvaluatedModelDataAdmin(admin.ModelAdmin):
    pass

@admin.register(FaceRecognition)
class FaceRecognitionAdmin(admin.ModelAdmin):
    pass

@admin.register(MLModel)
class MLModelAdmin(admin.ModelAdmin):
    list_display = ('name', 'format', 'is_active')
    # inlines = [EvaluatedModelDataInline]


@admin.register(TrainingDatasetFile)
class TrainingDatasetFileAdmin(admin.ModelAdmin):
    list_display = ('file', 'id', 'dataset')


@admin.register(TrainedDataset)
class TrainedDatasetAdmin(admin.ModelAdmin):
    pass

@admin.register(Dataset)
class TrainedDatasetAdmin(admin.ModelAdmin):
    pass
