from django.db import models
from django.db.models import JSONField


class FaceRecognition(models.Model):
    record_date = models.DateTimeField(auto_now_add=True)
    image = models.ImageField(upload_to='images/')
    results = JSONField(default='-')

    # recognized_image = models.ImageField(upload_to='ml_output/')

    def __str__(self):
        return str(self.record_date)


class EvaluatedModelData(models.Model):
    perfomance = models.FloatField()
    accuracy = models.FloatField()
    loss = models.FloatField()

    @classmethod
    def create(cls, perfomance, accuracy, loss):


        evaluatedModelData = cls(perfomance=perfomance, accuracy=accuracy, loss=loss)

        return evaluatedModelData

class Dataset(models.Model):
    file = models.CharField(unique=True, max_length=100)
    filename = models.CharField(max_length=50, unique=True)
    timestamp = models.DateTimeField(auto_now_add=True)


    @classmethod
    def create(cls, file, filename):
        dataset = cls(file=file, filename=filename)

        return dataset

class MLModel(models.Model):
    class MLFormat(models.TextChoices):
        H5 = ('h5', 'H5')
        H5_R = ('h5r', 'H5R')
        PICKLE = ('pkl', 'Pickle')


    name = models.CharField("Name", max_length=50)
    format = models.CharField("Format", max_length=50, choices=MLFormat.choices)
    timestamp = models.DateTimeField(auto_now_add=True)
    file = models.FileField(upload_to='models/')
    is_active = models.BooleanField(default=False)
    evaluated_data = models.OneToOneField(
        EvaluatedModelData,
        on_delete=models.CASCADE,
        related_name='ml_model',
        null=True,
        blank=True
    )

    def __str__(self):
        return self.name

    @classmethod
    def create(cls, name, file, evaluated_data, format, is_active):

        mlModel = cls(file=file, name=name, evaluated_data=evaluated_data, format=format, is_active=is_active)

        return mlModel

class TrainedDataset(models.Model):
    file = models.FileField(upload_to='dataset/data', default='default.zip')
    filename = models.CharField("filename", max_length=100, default="filename")

class TrainingDatasetFile(models.Model):
    file = models.FileField()
    dataset = models.OneToOneField(
        TrainedDataset, on_delete=models.CASCADE, related_name="dataset_file", null=True, blank=True
    )
