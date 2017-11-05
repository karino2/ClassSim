import pandas as pd
import numpy as np


class TwoLevelClassifier:
    def __init__(self, catspecs, h2binder, h1binder, otherlabel="OTHER"):
        """
            catspecs: list of dict which key is category key (eg. 100) and value is three tupple which indicate (threshold1, threshold2, categorname). category name is returned as a result of prediction.
            binder: ModelBinder
        """
        self._categories = {tup[0]: (tup[1], tup[2], tup[3]) for tup in catspecs}
        self._catkeys = [tup[0] for tup in catspecs]
        self._h2binder = h2binder
        self._h1binder = h1binder
        self._OTHERCLASS = -1
        self._OTHERLABEL = otherlabel

    def load_all(self):
        catkeys = self._catkeys
        self._h2binder.load_all_models(catkeys)
        self._h1binder.load_all_models(["h1_" + cat for cat in catkeys])

    def _predict_second(self, targetkey, btlnks, firstdf):
        catspec = self._categories[targetkey]
        thrshold = catspec[0]
        filtered = firstdf[firstdf[targetkey] > thrshold]

        if len(filtered) == 0:
            return None

        fbtlnks = btlnks[filtered.index, :]
        model = self._h1binder.get_or_load_model('h1_' + targetkey)

        # no second level classifier, all score is already enough.
        if model == None:
            return pd.DataFrame({targetkey: np.ones(len(filtered.index)), 'orgindex': filtered.index})

        res = model.predict(fbtlnks)
        scores = res[:, 1]

        return pd.DataFrame({targetkey: scores, 'orgindex': filtered.index})

    def _predict_bottlenecks_df(self, btlnks):
        firstdf = pd.DataFrame(self._h2binder.bottlenecks2predict(btlnks))

        resultdf = pd.DataFrame(np.zeros(firstdf.shape))
        resultdf.columns = firstdf.columns

        for targetkey in self._catkeys:

            df = self._predict_second(targetkey, btlnks, firstdf)
            if df is not None:
                resultdf.loc[df['orgindex'], targetkey] = df[targetkey].values

        return resultdf

    def _predict_bottlenecks_cattupple(self, btlnks):
        df = self._predict_bottlenecks_df(btlnks)
        self._df = df
        return self._df2classes(df)

    def predict_arrs(self, arrs):
        btlnks = self._h2binder.arrs2bottlenecks(arrs)
        return self._predict_bottlenecks_cattupple(btlnks)

    def predict_files(self, files):
        btlnks = self._h2binder.files2bottlenecks(files)
        return self._predict_bottlenecks_cattupple(btlnks)

    def predict_files_df(self, files):
        btlnks = self._h2binder.files2bottlenecks(files)
        tups = self._predict_bottlenecks_cattupple(btlnks)
        return pd.DataFrame({
            "filepaths": files,
            "category": [tup[0] for tup in tups],
            "label": [tup[1] for tup in tups]
        })

    def _row2class(self, rowdf):
        for catkey, catspec in self._categories.items():
            if rowdf[catkey] >= catspec[1]:
                return (catkey, catspec[2])
        return (self._OTHERCLASS, self._OTHERLABEL)

    def _df2classes(self, df):
        res = []
        for i in range(len(df)):
            rowdf = df.iloc[i, :]
            res.append(self._row2class(rowdf))
        return res
