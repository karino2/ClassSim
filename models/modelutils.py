
import unittest
def run_test(tcase):
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(tcase)
    unittest.TextTestRunner().run(suite)


import json
from keras.models import model_from_json
import re
import glob
import itertools

def load_model_from(jsonpath, weightpath):
    with open(jsonpath, 'r') as f:
        model_json = json.dumps(json.load(f)) # Need to convert json to str
    model = model_from_json(model_json)
    model.load_weights(weightpath)
    return model

def load_model(base_model_path):
    return load_model_from("{0}.json".format(base_model_path), "{0}.h5".format(base_model_path))

def save_model_to(model, jsonpath, weightpath):
    with open(jsonpath, 'w') as f:
        json.dump(json.loads(model.to_json()), f) # model.to_json() is a STRING of json
    model.save_weights(weightpath)


def save_model(model, base_model_path):
    jsonpath = "{0}.json".format(base_model_path)
    weightpath = '{0}.h5'.format(base_model_path)
    save_model_to(model, jsonpath, weightpath)

def peek_val_acc(path):
    m = re.search(r'([0-9]\.[0-9][0-9][0-9])\.h5$', path)
    if m:
        return float(m.group(1))
    return 0

def choose_best_val_acc_path(paths):
    paths = list(paths)
    accs = list(map(peek_val_acc, paths))
    argmax = accs.index(max(accs))
    return paths[argmax]

def choose_best_val_acc_path_from_base(base_model_path):
    paths = ["{0}.h5".format(base_model_path)] + list(glob.iglob("{0}-*.h5".format(base_model_path)))
    return choose_best_val_acc_path(paths)

def load_best_model_if_exist(base_model_path):
    if not os.path.exists("{0}.json".format(base_model_path)):
        return None
    return load_best_model(base_model_path)

def load_best_model(base_model_path):
    weight_path = choose_best_val_acc_path_from_base(base_model_path)
    return load_model_from("{0}.json".format(base_model_path), weight_path)

def dir2filedict(basedir):
    res = {}
    for f in glob.iglob("{}/*/*".format(basedir), recursive=True):
        cat = os.path.basename(os.path.dirname(f))
        res.setdefault(cat, []).append(f)
    return res

### it should have been better for sorted always, but keep old method so that prevoius result can be reproduced.
def dir2filedict_sorted(path):
    dic = dir2filedict(path)
    return {key: sorted(dic[key]) for key in dic.keys()}


def split_files(target_key, files_dict):
    truetrains = files_dict[target_key]
    falsetrains = [file for key, files in files_dict.items() if key != target_key
                    for file in files]
    return truetrains, falsetrains


TRAIN_VALID_RATIO=0.9
import random

from sklearn.model_selection import train_test_split

def split_fdict(fdict, test_size=0.2, random_state=1):
    trdict = {}
    valdict = {}
    cats = sorted(fdict.keys())
    idx = 0
    # random_state is for reproduce.
    # split always the same way may have some problem.
    # So I add idx to this seed for each split to make seed unique.
    for cat in cats:
        paths = sorted(fdict[cat])
        tr, val = train_test_split(paths, test_size=test_size, random_state=random_state+idx)
        trdict[cat] = tr
        valdict[cat] = val
        idx+=1
    return trdict, valdict

from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import optimizers
import os

class ModelCompiler:
    def __init__(self, basedir, only_top_layer=False):
        self.basedir = basedir
        self.basemodel_layer_num = 311 #corresponding to len(base_model.layers)
        self.only_top_layer = only_top_layer
    @property
    def imagenet_path(self):
        return os.path.join(self.basedir, "imagenet_model")
    def ensure_imagenet(self):
        imagenet_json = "{}.json".format(self.imagenet_path)
        if os.path.exists(imagenet_json):
            return
        self.download_imagenet(self.imagenet_path)
    def download_imagenet(self, path):
        model = InceptionV3(weights='imagenet', include_top=False)        
        
        with open("{0}.json".format(path), 'w') as f:
            json.dump(json.loads(model.to_json()), f) # model.to_json() is a STRING of json
        model.save_weights('{0}.h5'.format(path))
    def build_initial_model(self):
        self.ensure_imagenet()
        base_model = load_model(self.imagenet_path)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(2, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
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
        if self.only_top_layer:
            for layer in model.layers[:self.basemodel_layer_num]:
                layer.trainable = False
            for layer in model.layers[self.basemodel_layer_num:]:
                layer.trainable = True
        else:
            for layer in model.layers:
                layer.trainable = True
    def compile(self, model, optimizer = optimizers.Adam(lr=0.0001, decay=0.01)):
        #Model compile
        #optimizer = optimizers.Adagrad(lr=0.0025, epsilon=1e-08, decay=0.01)
        #optimizer = optimizers.SGD(lr=0.001, momentum=0.001, decay=0.001, nesterov=True)
        #optimizer = optimizers.SGD(lr=0.000001)
        # optimizer = optimizers.SGD(lr=0.0001)
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=["accuracy"])

   

class TestModelUtils(unittest.TestCase):
    def setUp(self):
        pass
    def test_imgpath2codepath(self):
        input = test_data = [ 'trained_model/old/model138_1vsAll_111-17-0.453.h5',
 'trained_model/old/model138_1vsAll_111-19-0.509.h5',
 'trained_model/old/model138_1vsAll_111.h5',
 'trained_model/old/model138_1vsAll_111-01-0.463.h5',
 'trained_model/old/model138_1vsAll_111-07-0.486.h5']
        expect ='trained_model/old/model138_1vsAll_111-19-0.509.h5'

        actual = choose_best_val_acc_path(input)
        self.assertEqual(expect, actual)
