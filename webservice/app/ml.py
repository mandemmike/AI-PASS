import numpy as np
import cv2
import pickle
from main import settings
import os
from app.models import MLModel
from keras.models import load_model
import keras.utils
import keras.preprocessing

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
    elif ml_model.format == MLModel.MLFormat.H5:
        return load_model(ml_model.file.path)


def pipeline_model(path):
    modelformat = get_current_model().format
    if modelformat == MLModel.MLFormat.H5:
        # pipeline model
        h5model = get_estimation_model()
        h5img = loadImage(path)
        cv2.imwrite(os.path.join(settings.MEDIA_URL,
                                 '/mloutput/process.jpg'), h5img)
        output = h5model.predict(h5img)
        print(output)
        h5age = np.argmax(output[0])
        h5gender = np.argmax(output[1])
        print(h5age)
        print(h5gender)
        machinlearning_results = dict(
            age=[], gender=[], count=[])
        machinlearning_results['age'].append(h5age)
        machinlearning_results['gender'].append(h5gender)
        machinlearning_results['count'].append(1)
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
        machinlearning_results = dict(age=[], gender=[], count=[])
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
                    gender = gender_estimation_model.predict_proba(
                        vectors).max()

                    cv2.imwrite(os.path.join(settings.MEDIA_URL,
                                             'mloutput/process.jpg'), image)
                    cv2.imwrite(os.path.join(settings.MEDIA_URL,
                                             'mloutput/roi_{}.jpg'.format(count)), face_roi)

                    machinlearning_results['count'].append(count)
                    machinlearning_results['age'].append(age)
                    machinlearning_results['gender'].append(gender)

                    count += 1

    return machinlearning_results
