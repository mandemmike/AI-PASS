#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# import numpy as np
# from keras_preprocessing.image import img_to_array, load_img
# from tensorflow.python.keras.models import load_model

# Model saved with Keras model.save()
# MODEL_PATH = './ModelTrainingService2/models/latest.h5'

# Load your trained model
# new_model = load_model(MODEL_PATH)


# Necessary
# print('Model loaded. Start serving...')


# def loadImage(filepath):
#     test_img = load_img(filepath, target_size=(198, 198))
#     test_img = img_to_array(test_img)
#     test_img = np.expand_dims(test_img, axis=0)
#     test_img /= 255
#     return test_img
#
#
# def model_predict(img_path):
#     global new_model
#     age_pred, gender_pred = new_model.predict(loadImage(img_path))
#     img = load_img(img_path)
#
#     max = -1
#     count = 0
#     # print('Chances of belonging in any category :')
#     xx = list(age_pred[0])
#     for i in age_pred[0]:
#         if i > max:
#             max = i
#             temp = count
#         count += 1
#
#     if temp == 0:
#         age = '0-24 yrs old'
#     if temp == 1:
#         age = '25-49 yrs old'
#     if temp == 2:
#         age = '50-74 yrs old'
#     if temp == 3:
#         age = '75-99 yrs old'
#     if temp == 4:
#         age = '91-124 yrs old'
#
#     if gender_pred[0][0] > gender_pred[0][1]:
#         gender = 'male'
#     else:
#         gender = 'female'
#
#     return age, gender
# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------

def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'main.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()