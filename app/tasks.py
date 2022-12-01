import time

from main.celery import app
from .models import TrainingDatasetFile
import zipfile
from django.conf import settings
import os
from app.services import Resnet

@app.task()
def dataset_preparation(dataset_id: int):
    instance: TrainingDatasetFile = TrainingDatasetFile.objects.get(id=dataset_id)
    media_path = settings.MEDIA_ROOT
    os_path = os.path.join(media_path, 'images', str(instance.id))
    os.makedirs(os_path, exist_ok=True)

    with zipfile.ZipFile(instance.file.path, 'r') as zip_ref:
         zip_ref.extractall(os_path)

    train_dataset(instance.id, os_path)

def train_dataset(dataset_id:int, train_path: str):
    # time.sleep(10)
    print(dataset_id)
    print(train_path)
    service = Resnet(train_path)
    train_data, val_data, preproc = service.split_dataset()
    learner = service.run_learning(train_data, val_data)
    predictor = service.get_predictor(learner, preproc)
    service.save_predictor_data(predictor, db_record_id=dataset_id)
    return predictor


