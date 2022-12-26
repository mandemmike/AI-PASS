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


        



                

           









