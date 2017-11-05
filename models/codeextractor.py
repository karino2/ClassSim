from keras.applications.inception_v3 import InceptionV3
from keras.models import Model, model_from_json
from keras.layers import Dense, GlobalAveragePooling2D
from keras import optimizers
from keras.models import Sequential
from models.modelutils import load_model, save_model, choose_best_val_acc_path_from_base, load_best_model, save_model_to

import os.path
from models.processor import DataSet
import pickle


def compile_extractor_model():
    '''
    outpu : compiled model
    '''
    assert False, "This extractor is obsolete, use 178 classifier as extractor."
    base_model = InceptionV3(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    return Model(inputs=base_model.input, outputs=x)


def fullmodel2extractor(model):
    return Model(inputs=model.input, outputs=model.layers[-3].output)


def fullmodel2codemodel(fullmodel):
    model = Sequential()
    model.add(Dense(1024, activation='relu', input_shape=(2048, )))
    model.add(Dense(2, activation='softmax'))
    model.layers[-2].set_weights(fullmodel.layers[-2].get_weights())
    model.layers[-1].set_weights(fullmodel.layers[-1].get_weights())
    return model


def load_extractor_from_fullmodel_path(path):
    return fullmodel2extractor(load_model(path))


def load_codemodel_from_fullmodel_path(path):
    return fullmodel2codemodel(load_model(path))


def fullmodel_file_to_codemodel_file(frompath, topath):
    save_model(load_codemodel_from_fullmodel_path(frompath), topath)


def to_fullmodel_path(basedir, basename, targetkey):
    return os.path.join(basedir, "{}_{}".format(basename, targetkey))


def to_codemodel_path(basedir, basename, targetkey):
    return os.path.join(basedir, "code_{}_{}".format(basename, targetkey))


def fullmodel_file_to_codemodel_file_by_basename(basedir, basename, targetkey):
    fullmodelpath = to_fullmodel_path(basedir, basename, targetkey)
    codemodelpath = to_codemodel_path(basedir, basename, targetkey)
    fullmodel_file_to_codemodel_file(fullmodelpath, codemodelpath)


def fullmodelprefix_to_best_code_full_path(fromprefix):
    bestweight = choose_best_val_acc_path_from_base(fromprefix)
    fulljson = "{0}.json".format(fromprefix)

    dirname, filename = os.path.dirname(bestweight), os.path.basename(bestweight)
    codeweight = os.path.join(dirname, "code_" + filename)
    codejson = os.path.join(dirname, "code_{0}.json".format(os.path.basename(fromprefix)))
    return (codejson, codeweight, fulljson, bestweight)


def best_code_path(basedir, basename, targetkey):
    codemodelprefix = to_codemodel_path(basedir, basename, targetkey)
    bestweight = choose_best_val_acc_path_from_base(codemodelprefix)
    codejson = "{0}.json".format(codemodelprefix)
    return (codejson, bestweight)


def best_code_full_path(basedir, basename, targetkey):
    fullmodelprefix = to_fullmodel_path(basedir, basename, targetkey)
    return fullmodelprefix_to_best_code_full_path(fullmodelprefix)


def best_fullmodel_file_to_codemodel_file(fromprefix):
    tojson, toweight, _, _ = fullmodelprefix_to_best_code_full_path(fromprefix)

    fullmodel = load_best_model(fromprefix)
    codemodel = fullmodel2codemodel(fullmodel)

    save_model_to(codemodel, tojson, toweight)


def best_fullmodel_file_to_codemodel_file_by_basename(basedir, basename, targetkey):
    fullmodelprefix = to_fullmodel_path(basedir, basename, targetkey)
    best_fullmodel_file_to_codemodel_file(fullmodelprefix)


class DNNCodeExtractor:
    def __init__(self, extractor=None):
        self.extractor = extractor
        self.ds = DataSet()
        self.chunk_size = 3000

    @classmethod
    def create(klass):
        return DNNCodeExtractor(compile_extractor_model())

    @classmethod
    def create_from(klass, path):
        return DNNCodeExtractor(load_extractor_from_fullmodel_path(path))

    def imgpath2codepath(self, fpath):
        return fpath + ".dat"

    def calc_chunked_bottlenecks(self, chunked_paths):
        datas = self.ds.files_to_dataset(chunked_paths)
        return self.arrs2bottlenecks(datas)

    def arrs2bottlenecks(self, arrs):
        return self.extractor.predict(arrs)

    def calc_bottlenecks(self, flist):
        chunked_paths = self.ds.chunked(flist, self.chunk_size)
        return [codes for chunk in chunked_paths for codes in self.calc_chunked_bottlenecks(chunk)]

    def save_one(self, btlnk, path):
        with open(path, "wb") as f:
            pickle.dump(btlnk, f)

    def save_all(self, bottlenecks, files):
        assert len(bottlenecks) == len(files)
        list(map(lambda pair: self.save_one(*pair), zip(bottlenecks, files)))

    def load_one(self, path):
        with open(path, "rb") as f:
            return pickle.load(f)

    def load_all(self, files):
        return map(self.load_one, files)

    def map(self, flist):
        # for backward compat
        return self.map_paths(flist)

    def map_path(self, flist):
        '''
        image paths -> DNNCode paths
        If code file does not exist, create it here.
        '''
        flist = list(flist)
        codepaths = map(self.imgpath2codepath, flist)
        targets = [img for img in flist if not os.path.exists(self.imgpath2codepath(img))]
        bottlenecks = self.calc_bottlenecks(targets)
        self.deb = bottlenecks
        self.save_all(bottlenecks, list(map(self.imgpath2codepath, targets)))

        # for summary
        self.input_num = len(flist)
        self.generate_num = len(targets)

        return codepaths

    def map_code(self, flist):
        '''
        image paths -> DNNCode
        If code file does not exist, create it here.
        '''
        return self.load_all(self.map_path(flist))

    def summary(self):
        return "{} out of {} code files are generated.".format(self.generate_num, self.input_num)


import unittest


class TestDNNCodeExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = DNNCodeExtractor(None)

    def test_imgpath2codepath(self):
        input = "test/1234.jpg"
        expect = "test/1234.jpg.dat"

        actual = self.extractor.imgpath2codepath(input)
        self.assertEqual(expect, actual)
