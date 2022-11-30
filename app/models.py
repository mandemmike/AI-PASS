from django.db import models


class FaceRecognition(models.Model):
    record_date = models.DateTimeField(auto_now_add=True)
    image = models.ImageField(upload_to='images/')

    def __str__(self):
        return str(self.record_date)


class TrainedDataset(models.Model):
    # predicted_age = models.FloatField()
    # predicted_gender = models.FloatField()
    pickle_file = models.FileField()

class TrainingDatasetFile(models.Model):
    file = models.FileField()
    dataset = models.OneToOneField(TrainedDataset, on_delete=models.CASCADE, related_name="dataset_file",
                                   null=True, blank=True)




