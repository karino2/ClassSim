import numpy as np
import pandas as pd
import re

def load_category_dict(path):
    catdf = pd.read_csv(path)
    return {idx:catdf[catdf['idx']==idx]['name'].values[0] for idx in catdf['idx']}

def lookup_index(catdict, catname):
    return [k for  k, v in catdict.items() if v == catname][0]

def category_matcher(catindex):
    target_pat = re.compile("(^{}/)|(.*/{}/)".format(catindex, catindex))
    return lambda f: None != target_pat.match(f)


def _merge_categories(df, thrshldcallable):
    handled = set()
    catlist = []
    
    def collect_recursive(targetcat):
        handled.add(targetcat)
        cands = df.index[thrshldcallable(df[targetcat])]
        reses = [cand for cand in cands if cand not in handled and cand != targetcat]
        list(map(lambda x: handled.add(x), reses))
        return [targetcat] + [one for oneres in reses
                    for one in collect_recursive(oneres)]
        
    
    for cat in df:
        if cat not in handled:
            samecats = collect_recursive(cat)
            catlist.append(samecats)
    return catlist


def merge_categories_below(df, thrshd):
    return _merge_categories(df, lambda targetdf: targetdf < thrshd)

def merge_categories_above(df, thrshd):
    return _merge_categories(df, lambda targetdf: targetdf > thrshd)

import os
import glob
import pickle

class VirtualCategories:
    def __init__(self, catstore):
        '''
        catstore: dict of names, keys for virtual categories. Created by virtual_categories.ipynb
        '''
        self.catkeys = catstore['keys']
        self.namesdict = catstore['names']
    @classmethod
    def from_file(klass, catpath):
        '''
        catpath: file path for pickled catstore
        basename: see __init__
        '''
        with open(catpath, "rb") as f:
            catstore = pickle.load(f)
        return VirtualCategories(catstore)
    def keys(self):
        return self.catkeys
    def items(self):
        return ((key, self.namesdict[key]) for key in self.keys())
    def name(self, key):
        return self.namesdict[key]
    def _glob(self, path):
        return glob.iglob("{}/**/*.jpg".format(path), recursive=True)
    def rawkey_to_path(self, rawkey, basepath):
        return os.path.join(basepath, rawkey)
    def files(self, key, basedir):
        return (file for rawkey in key.split("_")
                     for file in self._glob(self.rawkey_to_path(rawkey, basedir)))
    def list_dict(self, basedir):
        return {key: list(self.files(key, basedir)) for key in self.keys()}

def split_files(target_key, files_dict):
    truetrains = files_dict[target_key]
    falsetrains = [file for key, files in files_dict.items() if key != target_key
                    for file in files]
    return truetrains, falsetrains
