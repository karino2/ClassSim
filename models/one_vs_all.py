from keras.preprocessing.image import Iterator
from keras import backend
from keras.preprocessing.image import img_to_array, load_img
import random
import numpy as np



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
    def next(self):
        """For python 2.x.
        # Returns
            The next batch.
        """
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        batch_x = np.zeros((current_batch_size,) + self.image_shape, dtype=backend.floatx())
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




from itertools import filterfalse
import random
from operator import itemgetter
from models.processor import DataSet

class WeakPointCollector:
    def __init__(self):
        self.dataset = DataSet()
    def collect_weaks(self, model, falsefiles, sample_size, batch_size=32):
        '''
        Predict all false-labeled files and sort as highest (wrong) score orders, return pathes.
        '''
        sampled_falses = random.sample(list(falsefiles), sample_size)
        sampled_array = self.dataset.files_to_dataset(sampled_falses)
        prediction = model.predict(sampled_array, batch_size)
        tupples = [(pair[0][0], pair[0][1], pair[1]) for pair in zip(prediction, sampled_falses)]
        tupples.sort(key=itemgetter(1), reverse=True)
        # store intermediate data for debug
        self.sampled_falses, self.sampled_array, self.prediction, self.tupples = sampled_falses, sampled_array, prediction, tupples
        return list(map(itemgetter(2), tupples))


from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import optimizers
import os
from keras.layers.normalization import BatchNormalization
from models.modelutils import load_model, load_best_model, load_best_model_if_exist

# similar to ModelCompiler, but dup for a while.
class OneVsAllModelCompiler:
    def __init__(self, extractor_path):
        self.basemodel_layer_num = 311 #corresponding to len(base_model.layers)
        self.extractor_path = extractor_path
        self._extractor = None
    @property
    def extractor(self):
        if self._extractor:
            return self._extractor
        emodel = load_model(self.extractor_path)
        self._extractor = Model(inputs=emodel.input, outputs=emodel.layers[-3].output)
        return self._extractor
    def build_initial_model(self):
        base_model = self.extractor
        x = base_model.output
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(2, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        model._initial = True
        return model
    def codemodel_to_fullmodel(self, codemodel):
        base_model = self.extractor
        x = base_model.output
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(2, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        model.layers[-2].set_weights(codemodel.layers[-2].get_weights())
        model.layers[-1].set_weights(codemodel.layers[-1].get_weights())
        return model
    def generate_compiled_model(self, base_model_path):
        '''
        input : base_model_path - model_prefix of your trained model. If file not exist, create from imagenet.
        outpu : compiled model
        '''
        if os.path.exists("{0}.json".format(base_model_path)):
            model = load_best_model(base_model_path)
        else:
            model = self.build_initial_model()
        self.setup_trainable(model)
        self.compile(model)
        return model
    def setup_trainable(self, model):
        for layer in model.layers[:self.basemodel_layer_num]:
            layer.trainable = False
            # use extractor's batchnorm params
            if type(layer) == BatchNormalization:
                layer._per_input_updates = {}
        for layer in model.layers[self.basemodel_layer_num:]:
            layer.trainable = True
    def compile(self, model, optimizer = optimizers.Adam(lr=0.0001, decay=0.01)):
        #Model compile
        #optimizer = optimizers.Adagrad(lr=0.0025, epsilon=1e-08, decay=0.01)
        #optimizer = optimizers.SGD(lr=0.001, momentum=0.001, decay=0.001, nesterov=True)
        #optimizer = optimizers.SGD(lr=0.000001)
        # optimizer = optimizers.SGD(lr=0.0001)
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=["accuracy"])

import pandas as pd
from .model_binder import ModelBinder


class ManualCascader:
    def __init__(self, key_thrds_pair, binder):
        self.key_thrds_pair = key_thrds_pair
        self.keys = [pair[0] for pair in key_thrds_pair]
        self.otherclass = len(self.key_thrds_pair)
        self.binder = binder
    def load_all_models(self):
        self.binder.load_all_models(self.keys)
    def predict_raw(self, files):
        self.df = self.binder.predict(files)
        return self.df
    def row2class(self, rowdf):
        for i, pair in enumerate(self.key_thrds_pair):
            key, thrd = pair
            if rowdf[key] >= thrd:
                return i
        return self.otherclass
    def df2classes(self, df):
        res = []
        for i in range(len(df)):
            rowdf = df.iloc[i, :]
            res.append(self.row2class(rowdf))
        return res

    def predict(self, files):
        df = self.predict_raw(files)
        return self.df2classes(df)
    def predict_arrs(self, arrs):
        df = self.binder.predict_arrs(arrs)
        self.df = df
        return self.df2classes(df)
    def class2catname(self, classidx):
        if classidx == self.otherclass:
            return "9999"
        return self.keys[classidx]

from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
import json


import collections

FilesPair = collections.namedtuple("FilesPair", ["trues", "falses"])
TrValFiles = collections.namedtuple('TrValFiles', ['trainings', 'valids'])

#Image resize size
SIZE = 224




class OneVsAllModelTrainerBase:
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


class OneVsAllModelTrainer(OneVsAllModelTrainerBase):
    def __init__(self, train_datagen, valid_datagen):
        OneVsAllModelTrainerBase.__init__(self, train_datagen, valid_datagen)
        self.retrainings = 1 # this value is never used, but keep this sentence for document purpose.
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


class OneVsAllModelTrainerWeakpointCollector(OneVsAllModelTrainerBase):
    def __init__(self, train_datagen, valid_datagen):
        OneVsAllModelTrainerBase.__init__(self, train_datagen, valid_datagen)
        self.retrainings = 2
        self.sample_size_for_weak_collect = 5000
    def train_model(self, eachepochs=5, batch_size=16, target_size=(SIZE, SIZE)):
        with open("{0}.json".format(self.model_save_path), 'w') as f:
            json.dump(json.loads(self.model.to_json()), f) # model.to_json() is a STRING of json

        trs = self.trvals.trainings

        validgen = self.validation_generator(batch_size, target_size)
        collector = WeakPointCollector()

        for retry in range(self.retrainings):
            print("retry: {}".format(retry))

            print("=== start collecting weakpoints ===")
            weakfalses = collector.collect_weaks(self.model, trs.falses, self.sample_size_for_weak_collect, batch_size)
            traingen = OneVsAllFilesIterator(trs.trues, weakfalses[0:len(trs.trues)],  self.train_datagen, target_size=target_size, batch_size= batch_size)

            print("=== start training ===")
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

        # return collector for investigation purpose.
        return collector


from models.category import split_files, VirtualCategories
from models.modelutils import choose_best_val_acc_path
import random
import glob
import keras.backend.tensorflow_backend
from operator import itemgetter

class MultipleModelTrainer:
    def __init__(self, base_model_name, basedir, trainer, extractor_path):
        self.base_model_name = base_model_name
        self.basedir = basedir
        self.extractor_path = extractor_path

        self.compiler = OneVsAllModelCompiler(extractor_path)
        self.trainer = trainer
    def setup_filedict(self, train_files_dict, valid_files_dict):
        self.train_files_dict = train_files_dict
        self.valid_files_dict = valid_files_dict
        self.valid_files_dict_org = self.valid_files_dict
    def sample_valid_files(self, samplenum, seed=123):
        random.seed(seed)
        dic = {}

        # to reproduce smampling, we fix sort order.
        keyvals = sorted(list(self.valid_files_dict_org.items()), key=itemgetter(0))
        for key, val in keyvals:
            if(len(val) < samplenum):
                dic[key] = val
            else:
                dic[key] = random.sample(val, samplenum)
        self.valid_files_dict = dic
    def _model_path(self, target_key):
        return os.path.join(self.basedir, "{}_{}".format(self.base_model_name, target_key))
    def _split_by_set(self, target_key, false_keyset, files_dict):
        trues = files_dict[target_key]
        falses = [path for key in false_keyset for path in files_dict[key]]
        return FilesPair(trues, falses)
    def _split_files(self, targetkey, files_dict):
        return FilesPair(*split_files(targetkey, files_dict))
    def train_sub(self, target_key, highcat_keyset, eachepochs=10, retrainings=1, removecheckpoint=True):
        self.trainer.retrainings = retrainings
        falseset = highcat_keyset - set(target_key)
        trs = self._split_by_set(target_key, falseset, self.train_files_dict)
        vals = self._split_by_set(target_key, falseset, self.valid_files_dict)
        trvals = TrValFiles(trs, vals)
        self._train_one_core("h1_"+target_key, trvals, eachepochs, removecheckpoint)
    def train_one(self, target_key, eachepochs=10, retrainings=1, removecheckpoint=True):
        self.trainer.retrainings = retrainings
        trs = self._split_files(target_key, self.train_files_dict)
        vals = self._split_files(target_key, self.valid_files_dict)
        trvals = TrValFiles(trs, vals)
        self._train_one_core(target_key, trvals, eachepochs, removecheckpoint)
    def _clear_backend_session(self):
        import tensorflow as tf
        if keras.backend.tensorflow_backend._SESSION:
            tf.reset_default_graph()
            keras.backend.tensorflow_backend._SESSION.close()
            keras.backend.tensorflow_backend._SESSION = None

    def _train_one_setup(self, model_key, trvals):
        model_save_path = self._model_path(model_key)

        # this cause error. I guess it is because ImageDataGenerator becomes stale.
        # self._clear_backend_session()

        model = self.compiler.generate_compiled_model(model_save_path)
        self.trainer.set_model(model)
        self.trainer.set_savepath(model_save_path)
        self.trainer.set_dataset(trvals)


    def _train_one_core(self, model_key, trvals, eachepochs, removecheckpoint):
        self._train_one_setup(model_key, trvals)

        self.trainer.train_model(eachepochs=eachepochs)
        if removecheckpoint:
            self.trainer.remove_checkpoint()

    def remove_checkpoint(self, model_key):
        # utility method for cleaup interrupted case
        self.trainer.set_savepath(self._model_path(model_key))
        self.trainer.remove_checkpoint()

    def train_categories(self, targetkeys, eachepochs=10, retrainings=1):
        for key in targetkeys:
            print("Start training {}".format(key))

            self.train_one(key, eachepochs, retrainings)

def setup_filedict_by_vc(multi, vc, traindir, validdir):
    multi.setup_filedict(vc.list_dict(traindir), vc.list_dict(validdir))

def create_multiple_model_trainer_with_weakpoint_collect(base_model_name, basedir, traingen, validgen, extractor_path):
    return MultipleModelTrainer(base_model_name, basedir, OneVsAllModelTrainerWeakpointCollector(traingen, validgen), extractor_path)

def create_multiple_model_trainer(base_model_name, basedir, traingen, validgen, extractor_path):
    return MultipleModelTrainer(base_model_name, basedir, OneVsAllModelTrainer(traingen, validgen), extractor_path)

from models.codeextractor import load_codemodel_from_fullmodel_path, DNNCodeExtractor
from itertools import repeat
import random
from keras import optimizers
from models.modelutils import load_best_model

class ModelEvaluator:
    def __init__(self, valid_files_dict,  base_model_name, basedir, extractor_path):
        self.valid_files_dict = valid_files_dict
        self.base_model_name = base_model_name
        self.basedir = basedir
        self.extractor_path = extractor_path
        self.extractor = DNNCodeExtractor.create_from(extractor_path)
    def model_path(self, target_key):
        return os.path.join(self.basedir, "code_{}_{}".format(self.base_model_name, target_key))

    def evaluate_one(self, target_key):
        truevalids, falsevalids_all = split_files(target_key, self.valid_files_dict)
        # falsevalids = random.sample(falsevalids_all, len(truevalids))
        # falsevalids = falsevalids_all[0:len(truevalids)]
        falsevalids = falsevalids_all
        model_save_path = self.model_path(target_key)

        codemodel = load_best_model(model_save_path)

        truebtlnks  = self.extractor.map_code(truevalids)
        falsebtlnks = self.extractor.map_code(falsevalids)

        # truecodepaths = self.extractor.map(truevalids)
        #falsecodepaths = self.extractor.map(falsevalids)

        #truebtlnks = self.extractor.load_all(truecodepaths)
        #falsebtlnks = self.extractor.load_all(falsecodepaths)

        btlnks = list(truebtlnks) + list(falsebtlnks)
        y = list(repeat(1, len(truevalids))) + list(repeat(0, len(falsevalids)))

        codemodel.compile(optimizer=optimizers.Adam(lr=0.0001, decay=0.01), loss='sparse_categorical_crossentropy', metrics=["accuracy"])
        self.deb = np.array(btlnks), np.array(y)
        self.deb2 = codemodel

        return codemodel.evaluate(np.array(btlnks), np.array(y))


# for backward compatibility
from .two_level_classifier import TwoLevelClassifier
