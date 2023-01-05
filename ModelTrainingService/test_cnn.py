import os
import sys

import numpy as np
import pandas as pd
import unittest
import random
from unittest import TestCase
# import glob
from cnn_model import UtkFaceDataGenerator  # , UtkMultiOutputModel
from cnn_model import parse_dataset, parse_info_from_file  # , loadImage

dataset_path = '../webservice/dataset/test_data/'
ds_list = os.listdir(dataset_path)
img_path = dataset_path + ds_list[0]
training_split = 0,8

class TestParseDataset(TestCase):

    # Test for parse_dataset(): Testing if the dataframe is returned expected number of features
    def test_ParseDataset_BuildDataframe_ReturnNumberOfColumns_Successful(self):
        df = parse_dataset(dataset_path)
        actual = df.shape[1]
        expected = 4
        self.assertEqual(expected, actual)

    # Test for parse_dataset(): Testing if the dataframe gives the right feature names
    def test_ParseDataset_BuildDataframe_CheckNamesOfColumns_Successful(self):
        expected = ['age', 'gender', 'race', 'file']
        df = parse_dataset(dataset_path)
        actual = list(df.columns)
        self.assertListEqual(expected, actual)

    # Test for parse_info_from_file(): Testing if the dataframe split the right features
    def test_ParseInfoFromFile_SplitDataframe_CheckIfEachFeatureSplit_Successful(self):
        dataset_dict = {
            'age_id': {'0-24': 0, '25-49': 1, '50-74': 2, '75-99': 3, '100-124': 4},
            'race_id': {0: 'white', 1: 'black', 2: 'asian', 3: 'indian', 4: 'others'},
            'gender_id': {0: 'male', 1: 'female'}
        }
        dataset_dict['age_temp'] = dict((a, i) for i, a in dataset_dict['age_id'].items())  # (Age: id)
        dataset_dict['gender_temp'] = dict((g, i) for i, g in dataset_dict['gender_id'].items())  # (Gender: id)
        dataset_dict['race_temp'] = dict((r, i) for i, r in dataset_dict['race_id'].items())  # (Race: id)

        for img in ds_list:
            splitted = img.split('_')
            expected = (
                int(splitted[0]), dataset_dict['gender_id'][int(splitted[1])],
                dataset_dict['race_id'][int(splitted[2])])
            actual = parse_info_from_file(img)
            self.assertTupleEqual(expected, actual)


class TestUtkFaceDataGenerator(TestCase):

    # # Test for generate_split_indexes(): Testing if the training, validation, test dataset are splitted
    def test_generate_split_indexes(self):
        df_count = len(ds_list)
        expected_train_percentage = 63  # 64% training dataset
        expected_valid_percentage = 17  # 16% validation dataset
        expected_test_percentage = 20  # 20% test dataset
        total_percentage = expected_train_percentage + expected_valid_percentage + expected_test_percentage

        training_idx, valid_idx, test_idx = UtkFaceDataGenerator.generate_split_indexes(UtkFaceDataGenerator)
        # tra_idx, val_idx, tst_idx = training_idx, valid_idx, test_idx
        print('train_idx: ', training_idx)
        actual_train_perc = round((len(training_idx) / df_count) * 100)
        actual_valid_perc = round((len(valid_idx) / df_count) * 100)
        actual_test_perc = round((len(test_idx) / df_count) * 100)

        self.assertEqual(expected_train_percentage, actual_train_perc)
        self.assertEqual(expected_valid_percentage, actual_valid_perc)
        self.assertEqual(expected_test_percentage, actual_test_perc)
        self.assertEqual(total_percentage, 100)

    # Test for preprocess_image(): Testing if the image have the right shape
    def test_preprocess_image_shape(self):
        expected = (64, 64, 3)

        im = UtkFaceDataGenerator.preprocess_image(img_path)
        actual = im.shape
        self.assertTupleEqual(expected, actual)

    # Test for preprocess_image(): Testing if the image have the right datatype
    def test_preprocess_image_datatype(self):
        expected = np.asarray(np.where(64, 64, 3))

        actual = UtkFaceDataGenerator.preprocess_image(img_path)
        self.assertEqual(type(expected), type(actual))

    # Test for generate_images(): Testing if a single image batch is generated
    def test_generateImages_getAnImage_succesful(self):
        batch_size = 4

        permitation_no = np.random.permutation(len(ds_list))
        train_idx = permitation_no[:20]
        images = UtkFaceDataGenerator.generate_images(UtkFaceDataGenerator,
                                                      train_idx,
                                                      is_training=True,
                                                      batch_size=batch_size)
        expected = (64, 64, 3)
        for i in images:
            actual = i[0][0].shape
            j = 0
            for img in ds_list:
                if train_idx[j] == j:
                    j += 1
                    self.assertEqual(expected, actual)
            break

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testPipeline']
    unittest.main()
