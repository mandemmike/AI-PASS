import datetime
import ssl
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input

from app.models import EvaluatedModelData
from app.models import MLModel

ssl._create_default_https_context = ssl._create_unverified_context
import zipfile
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split


from tensorflow.keras.applications.resnet50 import ResNet50

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 128
SHAPE = (224, 224, 3)
max_age = 116


def preprocess_input_facenet(image_):

    preprocessed = tf.keras.applications.resnet50.preprocess_input(
    x=image_)

    return preprocessed

def getEvaluate(model_instance):

    image_gen = ImageDataGenerator(preprocessing_function=preprocess_input_facenet)

    data_folder = Path('./dataset/test_data')

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
    max_age = age_labels.max()

    data = {'img_name': correct_filenames,
            'age': age_labels / max_age,
            'race': race_labels,
            'gender': gender_labels}
    df = pd.DataFrame(data)
    df = df.astype({'race': 'int32', 'gender': 'int32'})

    val_generator = image_gen.flow_from_dataframe(
        dataframe=df,
        class_mode='multi_output',
        x_col='img_name',
        y_col=['gender', 'race', 'age'],
        directory=str(data_folder),
        target_size=IMAGE_SIZE,
        batch_size=24,
        shuffle=True)

    results = model_instance.evaluate(val_generator, steps=5)

    print(results)
    return results
    #eval = data={'perfomance': results[0], 'loss': results[6], 'accuracy': results[4]}
    #eval = EvaluatedModelData.create(perfomance=float(results[0]), accuracy=float(results[4]), loss=float(results[6]))
   # eval.save()




def decode_label(age):
    return int(age*max_age)

def show_face(image, age):
    plt.imshow(image)
    age = decode_label(age)
    plt.title('Age: {age}')
    plt.show()




def DynamicTraining(request, path_file_zip, Filename, epochs, steps_epochs):


    vggface_model = ResNet50()



    print(str(path_file_zip))
    with zipfile.ZipFile(path_file_zip, 'r') as zip_file:

        zip_file.extractall('./dataset/data/')

        data_folder = Path('./dataset/data/' + Filename)



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
    max_age = age_labels.max()

    data = {'img_name': correct_filenames,
            'age': age_labels / max_age,
            'race': race_labels,
            'gender': gender_labels}
    df = pd.DataFrame(data)
    df = df.astype({'race': 'int32', 'gender': 'int32'})


    df_train, df_val = train_test_split(df, test_size=0.3)

    print('Training:', len(df_train))
    print('Validation:', len(df_val))

    row = df.iloc[np.random.randint(len(df))]
    img = plt.imread(str(data_folder / row["img_name"]))




    ## Restnet50 pretrained weights

    base_model = tf.keras.Model([vggface_model.input], vggface_model.get_layer(index=-1).output)
    base_model.trainable = False

    input_layer = Input(shape=SHAPE)

    base_model_main = base_model(input_layer)

    out_base = Dense(64, name='added_new_layer')(base_model_main)


    race_output = Dense(5, activation='softmax', name='race')(out_base)
    age_output = Dense(1, activation='sigmoid', name='age')(out_base)
    gender_output = Dense(1, activation='sigmoid', name='gender')(out_base)

    outputs=[gender_output, race_output, age_output]

    model = tf.keras.Model(inputs=input_layer, outputs=outputs)

    model.summary()



    image_gen = ImageDataGenerator(preprocessing_function=preprocess_input_facenet)


    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)


    loss_bc = tf.keras.losses.BinaryCrossentropy()
    loss_scce = tf.keras.losses.SparseCategoricalCrossentropy()
    loss_mse = tf.keras.losses.MeanSquaredError()


    metric_bc = tf.keras.metrics.BinaryAccuracy()
    metric_scce = tf.keras.metrics.SparseCategoricalAccuracy()
    metric_mae = tf.keras.metrics.MeanAbsoluteError()

    model.compile(optimizer=optimizer,
              loss={'gender': loss_bc, 'race': loss_scce, 'age': loss_mse},
              metrics={'gender': metric_bc, 'race': metric_scce, 'age': metric_mae})


    CLASS_MODE = 'multi_output'
    TARGET_SIZE = IMAGE_SIZE



    # Preparing validation and training data for model training
    ############################
    ############################
    ############################
    train_generator = image_gen.flow_from_dataframe(
        dataframe=df_train,
        class_mode='multi_output',
        x_col='img_name',
        y_col=['gender', 'race', 'age'],
        directory=str(data_folder),
        target_size=TARGET_SIZE,
        batch_size=128,
        shuffle=True)


    val_generator = image_gen.flow_from_dataframe(
        dataframe=df_val,
        class_mode='multi_output',
        x_col='img_name',
        y_col=['gender', 'race', 'age'],
        directory=str(data_folder),
        target_size=TARGET_SIZE,
        batch_size=128,
        shuffle=True)



    path = Path("media/models")
    path.mkdir(exist_ok=True)



    # Model output config as well as metric to determine model saving
    ############################
    ############################
    ############################
    cpt_filename = 'media/models/' + Filename + '.h5'


    metric = 'age_mean_absolute_error' # Saving the best model according to the min(MAE)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(cpt_filename,
                                                monitor=metric,
                                                verbose=1,
                                                save_best_only=True,
                                                mode='min')

    log_dir = '../tblogs/multitask_three_out_' + datetime.datetime.now().strftime('%y-%m-%d_%H-%M')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0)

    log_dir




    # Training
    ############################
    ############################
    ############################

    EPOCHS = int(epochs)
    steps = len(df_val)//BATCH_SIZE

    model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    steps_per_epoch=int(steps_epochs),
    callbacks=[tensorboard_callback, checkpoint],
    validation_steps=steps)

    results = model.evaluate(val_generator, steps=steps)
    print(results)
    print('Training done')
    #eval = data={'perfomance': results[0], 'loss': results[6], 'accuracy': results[4]}
    eval = EvaluatedModelData.create(perfomance=float(results[0]), accuracy=float(results[4]), loss=float(results[6]))
    eval.save()

    model = MLModel.create(file=str(cpt_filename), format='h5r',name=Filename,is_active=False, evaluated_data=eval)

    return model



