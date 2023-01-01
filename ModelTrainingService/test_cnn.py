import os
import numpy as np
import unittest
import glob
from unittest import TestCase
from ModelTrainingService.cnn import UtkFaceDataGenerator, UtkMultiOutputModel
from ModelTrainingService.cnn import parse_dataset
from ModelTrainingService.cnn import loadImage


class TestUtkFaceDataGenerator(unittest.TestCase):

    def setUp(self):
        self.UtkFaceDataGenerator = UtkFaceDataGenerator()

class TestFunc(TestUtkFaceDataGenerator):

    def test_generate_split_indexes(self):
        self.success()

    def test_preprocess_image(self):

        input_image = np.array([[1., 1.], [1., 1.]])
        input_mask = 1
        expected_image = np.array([[0.00562123, 0.00562123], [0.00562123, 0.00562123]])

        result = self.unet._preprocess_image(input_image)
        self.assertEquals(expected_image, result[0])

    def test_generate_images(self):
        self.success()

    def success(self):
        pass

class TestUtkMultiOutputModel(unittest.TestCase):

    def setUp(self):
        self.UtkMultiOutputModel = UtkMultiOutputModel()

class TestCNNFunc(UtkMultiOutputModel):

    def test_hidden_layers(self):
        self.success()

    def test_build_race_branch(self):
        self.success()

    def test_build_gender_branch(self):
        self.success()

    def test_build_age_branch(self):
        self.success()

    def test_assemble_full_model(self):
        self.success()

    def success(self):
        pass


class Test(unittest.TestCase):

    # def setUp(self) -> None:
    #     print('Running on ', os.getcwd())
    #
    #     self.model = load_model()

    def test_parse_dataset(self):
        self.success()

    def test_load_image(self):
        self.success()

    def test_foo(tb):
        foo = tb.get("foo")
        assert foo(2) == 3
    def success(self):
        pass


    # def test_load_model_invalid_path(self):
    #     self.assertRaises(FileNotFoundError, model, 'invalid_path', 'invalid_path')

    # def test_data_leakage(train, validate, test):
    #     concat_dataset = np.concatenate((train.data, validate.data, test.data), axis=0)
    #     train, validate, test = train.data.size()[0], validate.data.size()[0], test.data.size()[0]
    #     assert concat_dataset.shape()[0] == train + validate + test


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testPipeline']
    unittest.main()
