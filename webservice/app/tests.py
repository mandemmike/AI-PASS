from django.test import TestCase
from pathlib import Path
import numpy as np
import pandas as pd
import os
import ssl
from django.urls import include, path
import sys


ssl._create_default_https_context = ssl._create_unverified_context

from django.conf import settings

filenames = []
count = 0
percent = 0
verified = 0
notClear = 0

# Create your tests here.
class DatasetSourceVerification(TestCase):

    def setUp(self):
        global filenames
        global length
        DIR_DATA = '/Data/UTKFace'
        dir = str(settings.BASE_DIR)

        data_folder = Path(dir + DIR_DATA)

        filenames = list(map(lambda x: x.name, data_folder.glob('*.jpg')))
        length = len(filenames)

        np.random.seed(10)
        np.random.shuffle(filenames)
        

    def testDatasetFormat(self):

        global filenames 
        global count
        global percent 
        global verified
        global notClear
        global length
        for filename in filenames:
        
            count += 1
            splittedFilename = filename.split('_')
            ##sys.exit()

            if len(splittedFilename) != 4:
                notClear += 1
            
            else:
                if int(splittedFilename[0]) >= 0 and int(splittedFilename[0]) < 175:
                    pass
                    
                else:  
                    notClear += 1

                if int(splittedFilename[1]) >= 0 and int(splittedFilename[1]) < 2:
                    pass
                    
                else:
                    notClear += 1
                if int(splittedFilename[2]) >= 0 and int(splittedFilename[2]) < 5:
                    verified +=1
                    pass
                else:
                    notClear += 1
        
        percent = verified / count
        percent = percent * 100
        print(str(percent) + ' percent of the labels correct')
        self.assertTrue(percent >= 99)


        


#Unit test for CNN model

# Test that the function correctly extracts the age, gender, and race information from the file names:

# def test_parse_dataset_extracts_correct_info(self):
#     df = parse_dataset('/path/to/dataset')
#     self.assertEqual(df['age'][0], expected_age)
#     self.assertEqual(df['gender'][0], expected_gender)
#     self.assertEqual(df['race'][0], expected_race)
                

# Test that the function correctly handles files with the wrong file extension:

# def test_parse_dataset_handles_wrong_file_extension(self):
#     df = parse_dataset('/path/to/dataset', ext='png')
#     self.assertEqual(len(df), 0)

           


# Test that the function correctly handles a non-existent dataset path:
# def test_parse_dataset_handles_nonexistent_path(self):
#     df = parse_dataset('/path/to/nonexistent/dataset')
#     self.assertEqual(len(df), 0)



# Test that the generate_images function correctly yields the expected number of batches:
# def test_generate_images_correct_number_of_batches(self):
#     num_batches = 0
#     for _ in generate_images(train_idx, is_training=True, batch_size=16):
#         num_batches += 1
#     self.assertEqual(num_batches, expected_num_batches)


# Test that the generate_images function correctly yields the expected shape and type for the images:
# def test_generate_images_correct_shape_and_type(self):
#     for images, _ in generate_images(train_idx, is_training=True, batch_size=16):
#         self.assertEqual(images.shape, (batch_size, image_width, image_height, 3))
#         self.assertEqual(images.dtype, np.float32)


# Test that the generate_images function correctly yields the expected shape and type for the labels:
# def test_generate_images_correct_labels_shape_and_type(self):
#     for _, labels in generate_images(train_idx, is_training=True, batch_size=16):
#         self.assertEqual(labels[0].shape, (batch_size, 5))
#         self.assertEqual(labels[0].dtype, np.float32)
#         self.assertEqual(labels[1].shape, (batch_size, 2))
#         self.assertEqual(labels[1].dtype, np.float32)

# unit tests for the hidden_layers method:
# def test_hidden_layers(self):
#         # Test with input of shape (10, 10, 3)
#         inputs = Input(shape=(10, 10, 3))
#         output = hidden_layers(inputs)
#         model = Model(inputs, output)

#         # Ensure the output has the expected shape
#         self.assertEqual(model.output_shape, (None, 1, 1, 256))

#         # Ensure the model can be trained
#         model.compile(loss="binary_crossentropy", optimizer="adam")
#         model.fit(np.random.random((128, 10, 10, 3)), np.random.random((128, 1, 1, 256)))


# unit tests for the build_gender_branch method:
# def test_build_gender_branch(self):
#         # Test with input of shape (10, 10, 3)
#         inputs = Input(shape=(10, 10, 3))
#         output = build_gender_branch(inputs)
#         model = Model(inputs, output)

#         # Ensure the output has the expected shape
#         self.assertEqual(model.output_shape, (None, 2))

#         # Ensure the model can be trained
#         model.compile(loss="binary_crossentropy", optimizer="adam")
#         model.fit(np.random.random((128, 10, 10, 3)), np.random.random((128, 2)))

# unit tests for the build_age_branch method:
# def test_build_age_branch(self):
#         # Test with input of shape (10, 10, 3)
#         inputs = Input(shape=(10, 10, 3))
#         output = build_age_branch(inputs)
#         model = Model(inputs, output)

#         # Ensure the output has the expected shape
#         self.assertEqual(model.output_shape, (None, 5))

#         # Ensure the model can be trained
#         model.compile(loss="categorical_crossentropy", optimizer="adam")
#         model.fit(np.random.random((128, 10, 10, 3)), np.random.random((128, 5)))