from django.db import models


class FaceRecognition(models.Model):
    record_date = models.DateTimeField(auto_now_add=True)
    image = models.ImageField(upload_to='images/')
    #recognized_image = models.ImageField(upload_to='ml_output/')

    def __str__(self):
        return str(self.record_date)


class MLModel(models.Model):
    name = models.CharField("Model's name", max_length=50)
    format = models.CharField("Model's format", max_length=50)
    timestamp = models.DateTimeField(auto_now_add=True)
    file = models.FileField(upload_to='models/')
    # is_active = models.BooleanField(default=False)

    def __str__(self) -> str:
        return str(self.model_timestamp)


class CurrentModel(models.Model):
    name = models.CharField("Model's name", max_length=50)
    format = models.CharField("Model's format", max_length=50)
    timestamp = models.DateTimeField(auto_now_add=True)
    file = models.FileField(upload_to='currentModel/')

    def __str__(self) -> str:
        return str(self.model_timestamp)


class TrainedDataset(models.Model):
    # predicted_age = models.FloatField()
    # predicted_gender = models.FloatField()
    pickle_file = models.FileField(null=True)


class TrainingDatasetFile(models.Model):
    file = models.FileField()
    dataset = models.OneToOneField(TrainedDataset, on_delete=models.CASCADE, related_name="dataset_file",
                                   null=True, blank=True)

class DatasetElementFormat(models.Model):
    filename = models.CharField(max_length=124)

    def isValid():
        return True

