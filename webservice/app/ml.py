import ktrain
import numpy as np
import cv2
import pickle
import pandas as pd

import tensorflow as tf
from main import settings
import os
from app.models import MLModel
from keras.models import load_model
import keras.utils
import keras.preprocessing
from keras.preprocessing.image import ImageDataGenerator
import imageio

from tensorflow.keras.models import load_model

STATIC_DIR = settings.STATIC_DIR


def loadImage(filepath):
    test_img = keras.utils.load_img(filepath, target_size=(224, 224, 3))
    test_img = keras.preprocessing.image.image_utils.img_to_array(test_img)
    test_img = np.expand_dims(test_img, axis=0)
    test_img /= 255

    return test_img


# face detection
face_detector_model = cv2.dnn.readNetFromCaffe(os.path.join(STATIC_DIR, 'models/deploy.prototxt.txt'),
                                               os.path.join(STATIC_DIR,
                                                            'models/res10_300x300_ssd_iter_140000.caffemodel'))
# feature extraction
face_feature_model = cv2.dnn.readNetFromTorch(
    os.path.join(STATIC_DIR, 'models/openface.nn4.small2.v1.t7'))

# face recognition
face_recognition_model = pickle.load(open(os.path.join(STATIC_DIR, 'models/machinelearning_face_person_identity.pkl'),
                                          mode='rb'))
# emotion recognition model
emotion_recognition_model = pickle.load(open(os.path.join(
    STATIC_DIR, 'models/machinelearning_face_emotion.pkl'), mode='rb'))

# TODO: connect age and gender models here

# age estimation model
age_estimation_model = pickle.load(
    open(os.path.join(STATIC_DIR, 'models/machinelearning_face_emotion.pkl'), mode='rb'))

gender_estimation_model = pickle.load(
    open(os.path.join(STATIC_DIR, 'models/machinelearning_face_emotion.pkl'), mode='rb'))


def get_current_model() -> MLModel:
    try:
        ml_model = MLModel.objects.get(is_active=True)
    except (MLModel.DoesNotExist,):
        raise RuntimeError('Please upload a model')
    return ml_model


def get_estimation_model():
    ml_model = get_current_model()
    file_path = os.path.join(settings.MEDIA_ROOT, ml_model.file.path)
    if ml_model.format == MLModel.MLFormat.PICKLE:
        with open(file_path, mode='rb') as file:
            return pickle.load(file)
    elif ml_model.format == MLModel.MLFormat.H5_R:
        return load_model(ml_model.file.path)
    elif ml_model.format == MLModel.MLFormat.H5:
        return load_model(ml_model.file.path)


def preprocess_input_facenet(image_):
 
    preprocessed = tf.keras.applications.resnet50.preprocess_input(
    x=image_)

    return preprocessed



def pipeline_model(path):
    print(path)
    modelformat = get_current_model().format

    if modelformat == MLModel.MLFormat.H5_R:
        age_max = 116
       
        image_gen = ImageDataGenerator(preprocessing_function=preprocess_input_facenet)

       
        image = imageio.imread(path)
        df = pd.DataFrame({'filename': [path], 'label': [1]})
        df['image'] = [image]
        print(df)
        gen = image_gen.flow_from_dataframe(df, 
                              x_col='filename',
                              y_col=['label'],
                              target_size=(224, 224),
                              class_mode='raw', 
                              batch_size=1)
        input_img, label = next(gen)

        model_pred = load_model(str(get_current_model().file))
       
        img = cv2.imread(path)
        output_img = img.copy()
        cv2.imwrite('./media/ml_output/process.jpg', output_img)
        cv2.imwrite('./media/ml_output/roi_1.jpg', img)
        output = model_pred.predict(input_img)
        print(output)
        h5age = int(output*age_max)
        predicted = round(h5age)
        print(get_current_model().file)
        print(predicted)
        machinlearning_results = dict(
            age=[], gender=[], count=[])
        machinlearning_results['age'].append(predicted)

        return machinlearning_results

    if modelformat == MLModel.MLFormat.H5:
        # pipeline model
        h5model = get_estimation_model()
        
        h5img = loadImage(path)
        img = cv2.imread(path)
        output_img = img.copy()
        cv2.imwrite('./media/ml_output/process.jpg', output_img)
        cv2.imwrite('./media/ml_output/roi_1.jpg', img)
        output = h5model.predict(h5img)
        print(get_current_model().file)
        print(output)
        h5age = np.argmax(output[0])
        h5gender = np.argmax(output[1])
        if h5age == 0:
            age = '0-24 yrs old'
        if h5age == 1:
            age = '25-49 yrs old'
        if h5age == 2:
            age = '50-74 yrs old'
        if h5age == 3:
            age = '75-99 yrs old'
        if h5age == 4:
            age = '100-124 yrs old'
        print(age)
        print(h5gender)
        if h5gender == 0:
            gender = 'Male'
        if h5gender == 1:
            gender = 'Female'
        machinlearning_results = dict(
            age=[], gender=[], count=[])
        machinlearning_results['age'].append(age)
        machinlearning_results['gender'].append(gender)
        machinlearning_results['count'].append(1)
    elif modelformat == MLModel.MLFormat.H5_R:
        h5r_model = get_estimation_model()
        h5img = loadImage(path)
        reloaded_predictor = ktrain.load_predictor(h5r_model)
        reloaded_predictor.predict_filename(h5img)
        img = cv2.imread(path)
        output_img = img.copy()
        cv2.imwrite('./media/ml_output/process.jpg', output_img)
        cv2.imwrite('./media/ml_output/roi_1.jpg', img)
        output = h5r_model.predict(h5img)
        h5age = np.argmax(output)
        predicted = round(h5age)
        print(get_current_model().file)
        print(predicted)
        machinlearning_results = dict(
            age=[], gender=[], count=[])
        machinlearning_results['age'].append(predicted)
    elif modelformat == MLModel.MLFormat.PICKLE:
        img = cv2.imread(path)
        image = img.copy()
        h, w = img.shape[:2]
        # face detection
        img_blob = cv2.dnn.blobFromImage(
            img, 1, (300, 300), (104, 177, 123), swapRB=False, crop=False)
        face_detector_model.setInput(img_blob)
        detections = face_detector_model.forward()

        # machcine results
        machinlearning_results = dict(age=[], count=[])
        count = 1
        if len(detections) > 0:
            for i, confidence in enumerate(detections[0, 0, :, 2]):
                if confidence > 0.5:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    startx, starty, endx, endy = box.astype(int)

                    cv2.rectangle(image, (startx, starty),
                                  (endx, endy), (0, 255, 0))

                    # feature extraction
                    face_roi = img[starty:endy, startx:endx]
                    face_blob = cv2.dnn.blobFromImage(
                        face_roi, 1 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=True)
                    face_feature_model.setInput(face_blob)
                    vectors = face_feature_model.forward()
                    age = get_estimation_model().predict(vectors)[0]
                    #gender = gender_estimation_model.predict_proba(vectors).max()

                    cv2.imwrite(os.path.join(settings.MEDIA_URL,
                                             'ml_output/process.jpg'), image)
                    cv2.imwrite(os.path.join(settings.MEDIA_URL,
                                             'ml_output/roi_{}.jpg'.format(count)), face_roi)

                    machinlearning_results['count'].append(count)
                    machinlearning_results['age'].append(age)
                    #machinlearning_results['gender'].append(gender)

                    count += 1

    return machinlearning_results
