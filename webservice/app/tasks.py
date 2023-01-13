import os
import zipfile

from django.conf import settings

from main.celery import app
from .models import TrainingDatasetFile


@app.task()
def dataset_preparation(dataset_id: int):
    instance: TrainingDatasetFile = TrainingDatasetFile.objects.get(id=dataset_id)
    media_path = settings.MEDIA_ROOT
    os_path = os.path.join(media_path, 'images', str(instance.id))
    os.makedirs(os_path, exist_ok=True)

    with zipfile.ZipFile(instance.file.path, 'r') as zip_ref:
         zip_ref.extractall(os_path)

