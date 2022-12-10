from django.db import models


class FaceRecognition(models.Model):
    record_date = models.DateTimeField(auto_now_add=True)
    image = models.ImageField(upload_to='images/')

    # recognized_image = models.ImageField(upload_to='ml_output/')

    def __str__(self):
        return str(self.record_date)


class MLModel(models.Model):
    class MLFormat(models.TextChoices):
        H5 = ('h5', 'H5')
        PICKLE = ('pickle', 'Pickle')

    name = models.CharField("Model's name", max_length=50)
    format = models.CharField("Model's format", max_length=50, choices=MLFormat.choices)
    timestamp = models.DateTimeField(auto_now_add=True)
    file = models.FileField(upload_to='models/')
    is_active = models.BooleanField(default=False)

    def __str__(self):
        return self.name


class TrainedDataset(models.Model):
    pickle_file = models.FileField(null=True)


class TrainingDatasetFile(models.Model):
    file = models.FileField()
    dataset = models.OneToOneField(
        TrainedDataset, on_delete=models.CASCADE, related_name="dataset_file", null=True, blank=True
    )
