import os
import numpy as np
import pandas as pd
import unittest
from unittest import TestCase
# import glob
from cnn_model import UtkFaceDataGenerator  #, UtkMultiOutputModel
from cnn_model import parse_dataset, parse_info_from_file
# from ModelTrainingService.cnn import loadImage


class TestUtkFaceDataGenerator(TestCase):

    # # Test for generate_split_indexes(): Testing if the training, validation, test dataset are splitted
    def test_generate_split_indexes(self):
        df_count = 23708
        expected_train_percentage = 64
        expected_valid_percentage = 16
        expected_test_percentage = 20

        self.trainset, self.validset, self.testset = UtkFaceDataGenerator.generate_split_indexes(UtkFaceDataGenerator)

        actual_train_perc = round((len(self.trainset) / df_count) * 100)
        actual_valid_perc = round((len(self.validset) / df_count) * 100)
        actual_test_perc = round((len(self.testset) / df_count) * 100)

        self.assertEqual(expected_train_percentage, actual_train_perc)
        self.assertEqual(expected_valid_percentage, actual_valid_perc)
        self.assertEqual(expected_test_percentage, actual_test_perc)

    # Test for preprocess_image(): Testing if the image have the right shape
    def test_preprocess_image_shape(self):
        expected = (64, 64, 3)
        path = 'testing-dataset/13_0_0_20170104013342923.jpg.chip.jpg'
        im = UtkFaceDataGenerator.preprocess_image(path)
        actual = im.shape
        self.assertTupleEqual(expected, actual)

    # Test for preprocess_image(): Testing if the image have the right datatype
    def test_preprocess_image_datatype(self):
        expected = np.asarray(np.where(64, 64, 3))

        path = 'testing-dataset/13_0_0_20170104013342923.jpg.chip.jpg'
        actual = UtkFaceDataGenerator.preprocess_image(path)
        self.assertEqual(type(expected), type(actual))


class TestParseDataset(TestCase):

    # Test for parse_dataset(): Testing if the dataframe is returned expected number of features
    def test_ParseDataset_BuildDataframe_ReturnNumberOfColumns_Successful(self):
        dataset_path = 'testing-dataset/'

        df = parse_dataset(dataset_path)
        actual = df.shape[1]
        expected = 4
        self.assertEqual(expected, actual)

    # Test for parse_dataset(): Testing if the dataframe gives the right feature names
    def test_ParseDataset_BuildDataframe_CheckNamesOfColumns_Successful(self):
        dataset_path = 'testing-dataset/'

        expected = ['age', 'gender', 'race', 'file']
        df = parse_dataset(dataset_path)
        actual = list(df.columns)
        self.assertListEqual(expected, actual)

    # Test for parse_info_from_file(): Testing if the dataframe split the right features
    def test_ParseInfoFromFile_SplitDataframe_CheckIfEachFeatureSplit_Successful(self):
        ds_path = 'testing-dataset'
        ds_list = os.listdir(ds_path)

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
            int(splitted[0]), dataset_dict['gender_id'][int(splitted[1])], dataset_dict['race_id'][int(splitted[2])])
            actual = parse_info_from_file(img)
            self.assertTupleEqual(expected, actual)


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testPipeline']
    unittest.main()
