from django.test import TestCase
from pathlib import Path
import numpy as np
from django.conf import settings
from app.test_helper import parse_dataset
from app.test_helper import generate_images

filenames = []
count = 0
percent = 0
verified = 0
notClear = 0


class GeneralTest(TestCase):

    def setUp(self):
        global filenames
        global length
        DIR_DATA = '/Data/UTKFace'
        dir = str(settings.BASE_DIR)

        data_folder = Path('./dataset/test_data')

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
                    verified += 1
                    pass
                else:
                    notClear += 1

        percent = verified / count
        percent = percent * 100
        print(str(percent) + ' percent of the labels correct')
        self.assertTrue(percent >= 99)

    def ml_pipeline(self):

        pass

    def test_parse_dataset(self):

        data_folder = ('./dataset/test_data')

        df = parse_dataset(data_folder)

        filename = df['img_name'][0]
        splittedFilename = filename.split('_')

        expected_age = int(splittedFilename[0])
        label_age = round(float(df['age'][0]) * 116)
        expected_gender = int(splittedFilename[1])
        label_gender = int(df['gender'][0])
        self.assertEqual(label_age, expected_age)
        self.assertEqual(label_gender, expected_gender)

    def test_parse_dataset_handles_nonexistent_path(self):
        df = parse_dataset('/path/to/nonexistent/dataset')
        self.assertEqual(len(df), 0)

    def test_generate_images_correct_number_of_batches(self):
        data_folder = ('./dataset/test_data')
        df = parse_dataset(data_folder)

        expected_batch_size = 12
        num_batches = 0
        val_gen = generate_images(df, expected_batch_size)

        print(num_batches)
        self.assertEqual(num_batches, expected_batch_size)
