import os
import sys
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import unittest
import random
from unittest import TestCase
# import glob
from cnn_model import UtkFaceDataGenerator  # , UtkMultiOutputModel
from cnn_model import parse_dataset, parse_info_from_file  # , loadImage

dataset_path = '../webservice/dataset/test_data/'
ds_list = os.listdir(dataset_path)
img_path = dataset_path + ds_list[0]
training_split = 0, 8
expected_train_percentage = 63  # 63% training dataset
expected_valid_percentage = 17  # 17% validation dataset
expected_test_percentage = 20  # 20% test dataset


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
        total_percentage = expected_train_percentage + expected_valid_percentage + expected_test_percentage

        training_idx, valid_idx, test_idx = UtkFaceDataGenerator.generate_split_indexes(UtkFaceDataGenerator)
        # tra_idx, val_idx, tst_idx = training_idx, valid_idx, test_idx
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

    def test_generate_images(self):
        batch_size = 4

        permitation_no = np.random.permutation(len(ds_list))
        train_idx = permitation_no[:20]
        images = UtkFaceDataGenerator.generate_images(UtkFaceDataGenerator,
                                                      train_idx,
                                                      is_training=True,
                                                      batch_size=batch_size)

        expected = (64, 64, 3)
        return_img_arr = []
        expected_img_arr_bins = []
        actual_img_arr_bins = []

        idx = 0
        for i in images:

            actual_img_arr_bins.append(i[0][idx])
            actual_img_shape = i[0][idx].shape
            idx += 1

            if idx == 4:
                idx = 0
            # an assertion for the image shape
            self.assertEqual(expected, actual_img_shape)

            batch_list = i[2]

            for j in ds_list:
                for batch_item in batch_list:
                    slash = "\\"
                    splitted = batch_item.split(slash)
                    actual_file_name = splitted[1]

                    if actual_file_name == j:
                        # append() each img from images to return img arr
                        return_img_arr.append(actual_file_name)

                        # append() each corresponding img arr in our local dataset
                        img_bin = cv2.imread(dataset_path + actual_file_name)
                        img_bin = cv2.resize(img_bin, (64, 64))
                        img_bin = cv2.cvtColor(img_bin, cv2.COLOR_BGR2RGB)
                        img_bin = np.array(img_bin) / 255.0
                        expected_img_arr_bins.append(img_bin)

        print('expected_img_arr_bins,', len(expected_img_arr_bins), 'actual_img_arr_bins', len(actual_img_arr_bins))
        self.assertListEqual(expected_img_arr_bins, actual_img_arr_bins)

        # Above we need the file to compare with original array by indexes
        # Get the unique items from file to an array
        # Get the percentage of the list count to be equal to expected percentages.


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testPipeline']
    unittest.main()
