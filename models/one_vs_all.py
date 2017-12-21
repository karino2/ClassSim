from keras.preprocessing.image import Iterator
from keras import backend
from keras.preprocessing.image import img_to_array, load_img
import random
import numpy as np

# image resize size.
SIZE=224

class OneVsAllFilesIterator(Iterator):
    def __init__(self, true_files, false_files, image_data_generator, target_size=(256, 256), batch_size=32, shuffle=True, seed=None):

        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)

        # assume channel last.
        self.data_format = 'channels_last'
        assert backend.image_data_format() == self.data_format
        self.image_shape = self.target_size + (3,)

        self.filenames = []
        self.filenames.extend(true_files)
        self.classes = np.ones(len(true_files))
        self.filenames.extend(false_files)
        self.classes = np.append(self.classes, np.zeros(len(false_files)))

        self.n = len(self.filenames)

        super(OneVsAllFilesIterator, self).__init__(self.n, batch_size, shuffle, seed)
    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros((len(index_array),) + self.image_shape, dtype=backend.floatx())
        for i, j in enumerate(index_array):
            fname = self.filenames[j]
            img = load_img(fname,
                           grayscale=False,
                           target_size=self.target_size)
            x = img_to_array(img, data_format=self.data_format)
            x = self.image_data_generator.random_transform(x)
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x
        batch_y = self.classes[index_array].astype(backend.floatx())
        return batch_x, batch_y
    def next(self):
        """For python 2.x.
        # Returns
            The next batch.
        """
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)

from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
import json
import os
import glob
from models.modelutils import choose_best_val_acc_path

import collections

FilesPair = collections.namedtuple("FilesPair", ["trues", "falses"])
TrValFiles = collections.namedtuple('TrValFiles', ['trainings', 'valids'])

class OneVsAllModelTrainer:
    def __init__(self, train_datagen, valid_datagen):
        self.train_datagen = train_datagen
        self.valid_datagen = valid_datagen
    def set_model(self, model):
        self.model = model
    def set_savepath(self, model_save_path):
        self.model_save_path= model_save_path
        self.file_path = self.model_save_path + "-{epoch:02d}-{val_acc:.3f}.h5"
        self.checkpoint = ModelCheckpoint(
            self.file_path
            , monitor='val_acc'
            , verbose=1
            , save_best_only=False
            , mode='max'
        )
        self.callbacks_list = [self.checkpoint]
    def set_dataset(self, trvals):
        self.trvals = trvals
    def set_dataset_files(self, true_trainings, false_trainings, true_valids, false_valids):
        trs = FilesPair(trues=true_trainings, falses=false_trainings)
        vals = FilesPair(trues=true_valids, falses = false_valids)
        trval = TrValFiles(trs, vals)
        self.set_dataset(trval)
    def validation_generator(self, batch_size, target_size):

        # false_sampled = random.sample(false_valids, len(true_valids))
        # temporary use whole false validation data.
        vals = self.trvals.valids
        false_sampled = vals.falses
        return OneVsAllFilesIterator(vals.trues, false_sampled, self.valid_datagen, target_size=target_size, batch_size=batch_size)
    def save_result(self, history):
        # use epoch 99 as special (last saved model).
        self.model.save_weights("{0}-99-{1:.3f}.h5".format(self.model_save_path, history.history['val_acc'][-1]))
    def list_checkpoints_except_best(self):
        pat = "{}-*.h5".format(self.model_save_path)
        paths = list(glob.iglob(pat))
        best = choose_best_val_acc_path(paths)
        return [path for path in paths if path != best]
    def remove_checkpoint(self):
        list(map(os.remove, self.list_checkpoints_except_best()))
    def train_model(self, eachepochs=5, batch_size=16, target_size=(SIZE, SIZE)):
        with open("{0}.json".format(self.model_save_path), 'w') as f:
            json.dump(json.loads(self.model.to_json()), f) # model.to_json() is a STRING of json

        trs = self.trvals.trainings

        validgen = self.validation_generator(batch_size, target_size)
        traingen = OneVsAllFilesIterator(trs.trues, random.sample(trs.falses, len(trs.trues)),  self.train_datagen, target_size=target_size, batch_size= batch_size)

        history = self.model.fit_generator(
            generator=traingen
            #, steps_per_epoch= 100
            , steps_per_epoch= traingen.n/batch_size
            , epochs=eachepochs
            , verbose=1
            , validation_data=validgen
            , validation_steps=validgen.n/batch_size
            # , validation_steps=10
            , callbacks=self.callbacks_list
        )

        self.save_result(history)




