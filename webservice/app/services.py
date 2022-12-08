import os
import ktrain
from ktrain import vision as vis
import matplotlib.pyplot as plt
import pickle
import re


class Resnet:

    def __init__(self, path: str):
        self.datadir = os.path.join(path, 'UTK')
        self.pattern = self._image_pattern()


    def _image_pattern(self):
        pattern = r'([^/]+)_\d+_\d+_\d+.jpg$'
        p = re.compile(pattern)
        #r = p.search('/hello/world/40_1_0_20170117134417715.jpg')
        #print("Extracted Age:%s" % (int(r.group(1))))
        return p

    def split_dataset(self, horizontal_flip: bool = True):
        data_aug = vis.get_data_aug(horizontal_flip=horizontal_flip)
        (train_data, val_data, preproc) = vis.images_from_fname(
            self.datadir,
            pattern=self.pattern,
            data_aug=data_aug,
            is_regression=True,
            random_state=42,
        )
        return train_data, val_data, preproc

    def run_learning(self, train_data, val_data):
        model = vis.image_regression_model('pretrained_resnet50', train_data, val_data)
        learner = ktrain.get_learner(
            model=model,
            train_data=train_data,
            val_data=val_data,
            workers=8,
            use_multiprocessing=False,
            batch_size=64
        )
        learner.lr_find(max_epochs=2)
        learner.lr_plot()
        learner.fit_onecycle(1e-4, 3)
        learner.fit_onecycle(1e-4, 2)
        return learner

    def save_predictor_data(self, predictor, db_record_id: int):
        trained_model_file = f'{db_record_id}_age_estimation.pkl'
        pickle.dump(predictor, open(trained_model_file, 'wb'))



    def get_predictor(self, learner, preproc, save_model=True):

        predictor = ktrain.get_predictor(learner.model, preproc)
        # if save_model:
        #     pickle.dump(predictor, open(self.trained_model_file, 'wb'))
        return predictor

    # def predict(self, predictor):
    #        fname =  + '/' + fname
    #        predicted = round(predictor.predict_filename(fname)[0])
    #        vis.show_image(fname)
    #        print('predicted:%s' % (predicted))
    #        return predicted




