import numpy as np
from models.modelutils import load_best_model_if_exist
import os


class ModelBinder:
    def __init__(self, base_model_name, basedir, extractor):
        self.base_model_name = base_model_name
        self.basedir = basedir
        self.extractor = extractor
        self._models = {}
        self.verbose = True

    @classmethod
    def dup_from(cls, binder):
        return ModelBinder(binder.base_model_name, binder.basedir, binder.extractor)

    def model_path(self, target_key):
        return os.path.join(self.basedir, "{}_{}".format(self.base_model_name, target_key))

    def get_or_load_model(self, target_key):
        if target_key in self._models:
            return self._models[target_key]
        self.notify("load {}".format(target_key))
        self._models[target_key] = load_best_model_if_exist(self.model_path(target_key))
        return self._models[target_key]

    def notify(self, msg):
        if self.verbose:
            print(msg)

    def load_all_models(self, keys):
        list(map(self.get_or_load_model, keys))

    def files2bottlenecks(self, files):
        return np.array(list(self.extractor.map_code(files)))

    def predict(self, files):
        btlnks = self.files2bottlenecks(files)
        preddict = self.bottlenecks2predict(btlnks)
        preddict["filepaths"] = files
        return pd.DataFrame(preddict)

    def bottlenecks2predict(self, btlnks):
        models = self._models
        return {key: models[key].predict(btlnks)[:, 1] for key in models.keys()}

    def predict_arrs(self, arrs):
        btlnks = self.arrs2bottlenecks(arrs)
        preddict = self.bottlenecks2predict(btlnks)
        return pd.DataFrame(preddict)

    def add_argmax_max(self, df):
        catkeys = list(self._models.keys())
        df['argmax'] = df[df.columns[:len(catkeys)]].idxmax(axis=1)
        df['max'] = df[df.columns[0:len(catkeys)]].max(axis=1)

    def arrs2bottlenecks(self, arrs):
        return self.extractor.arrs2bottlenecks(arrs)
