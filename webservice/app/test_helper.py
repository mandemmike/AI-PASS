from pathlib import Path
import numpy as np
import pandas as pd
from django.conf import settings
from django.urls import include, path
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 128
SHAPE = (224, 224, 3)
max_age = 116


def preprocess_input_facenet(image_):
    preprocessed = tf.keras.applications.resnet50.preprocess_input(
        x=image_)

    return preprocessed


def parse_dataset(path):
    image_gen = ImageDataGenerator(preprocessing_function=preprocess_input_facenet)

    data_folder = Path(path)

    filenames = list(map(lambda x: x.name, data_folder.glob('*.jpg')))

    np.random.seed(10)
    np.random.shuffle(filenames)
    gender_mapping = {0: 'Male', 1: 'Female'}
    race_mapping = dict(list(enumerate(('White', 'Black', 'Asian', 'Indian', 'Others'))))
    age_labels, gender_labels, race_labels, correct_filenames = [], [], [], []

    for filename in filenames:
        if len(filename.split('_')) != 4:
            print(f"Bad filename {filename}")
            continue

        age, gender, race, _ = filename.split('_')
        correct_filenames.append(filename)
        age_labels.append(age)
        gender_labels.append(gender)
        race_labels.append(race)

    age_labels = np.array(age_labels, dtype=np.float32)
    max_age = 116

    data = {'img_name': correct_filenames,
            'age': age_labels / max_age,
            'race': race_labels,
            'gender': gender_labels}
    df = pd.DataFrame(data)
    df = df.astype({'race': 'int32', 'gender': 'int32'})
    """
    val_generator = image_gen.flow_from_dataframe(
        dataframe=df,
        class_mode='multi_output',
        x_col='img_name',
        y_col=['gender', 'race', 'age'],
        directory=str(data_folder),
        target_size=IMAGE_SIZE,
        batch_size=24,
        shuffle=True)
    """

    return df


def generate_images(df, batch_num):
    image_gen = ImageDataGenerator(preprocessing_function=preprocess_input_facenet)

    val_generator = image_gen.flow_from_dataframe(
        dataframe=df,
        class_mode='multi_output',
        x_col='img_name',
        y_col=['gender', 'race', 'age'],
        directory=str('./dataset/test_data'),
        target_size=IMAGE_SIZE,
        batch_size=batch_num,
        shuffle=True)

    return val_generator
