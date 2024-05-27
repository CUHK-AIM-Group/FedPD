import os
import pickle
from itertools import chain
import csv
import collections
import numpy as np
import ot
import sys

def get_timestamp_other():
    import time
    import datetime
    ts = time.time()
    # %f allows granularity at the micro second level!
    timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H-%M-%S_%f')
    return timestamp

class dotdict(dict):
    """ dot.notation access to dictionary attributes """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def mkdir(path):
    os.makedirs(path, exist_ok=True)
    # if not os.path.exists(path):
    #     os.makedirs(path)

def pickle_obj(obj, path, mode = "wb", protocol=pickle.HIGHEST_PROTOCOL):
    '''
    Pickle object 'obj' and dump at 'path' using specified
    'mode' and 'protocol'
    Returns time taken to pickle
    '''

    import time
    st_time = time.perf_counter()
    pkl_file = open(path, mode)
    pickle.dump(obj, pkl_file, protocol=protocol)
    end_time = time.perf_counter()
    return (end_time - st_time)

def dict_union(*args):
    return dict(chain.from_iterable(d.items() for d in args))

def isnan(x):
    return x != x

