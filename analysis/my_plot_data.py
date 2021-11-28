
from collections import defaultdict
import pandas as pd
# import numpy as np
import copy
import random

class MyPlotData():
    def __init__(self):
        self.fields = set()
        self.data = []
        pass
    def add_data_point(self, **kwargs):
        self.fields |= set(kwargs.keys())
        self.data.append(kwargs)
        # for key, value in kwargs.iteritems():
        #     self.data.append()
    def add(self, **kwargs):
        self.add_data_point(**kwargs)
    def add_key_value(self, k, v):
        self.fields.add(k)
        for d in self.data:
            d[k] = v
        return self
    def append(self, other, replace=[], sample=False, sample_seed=0):
        self.fields |= other.fields
        data = copy.deepcopy(other.data)
        for kv in replace:
            k, v = kv
            for d in data:
                d[k] = v
        if isinstance(sample, float):
            sample = int(sample*len(data))
        random.seed(sample_seed)
        if sample and sample < len(data):
            for e in random.sample(data, sample):
                self.data.append(e)
        else:
            self.data.extend(data)
    def get_histogram(self, count_var=None):
        assert count_var is not None
        hist = defaultdict(int)
        # fields = [k for k in self.fields - hist_var]
        # fields = [k for k in self.fields]
        key = count_var
        for entry in self.data:
            # if hist_var not in entry:
            #     continue
            if key is None:
                key = tuple([v for v in entry])
            val = entry[key]
            hist[val] += 1
        return hist
    def to_histogram(self, count_var=None, hist_key='hist_key', hist_val='hist_val'):
        assert count_var is not None
        hist = self.get_histogram(count_var)
        ret = MyPlotData()
        for k, v in hist.items():
            args = {
                f'{hist_key}': k,
                f'{hist_val}': v,
            }
            ret.add(**args)
        return ret

        # # fields = [k for k in self.fields - hist_var]
        # fields = [k for k in self.fields]
        # key = count_var
        # for entry in self.data:
        #     # if hist_var not in entry:
        #     #     continue
        #     if key is None:
        #         key = tuple([v for v in entry])
        #     val = entry[key]
        #     hist[val] += 1
        # return hist

        # return hist
        # ret = MyPlotData()
        # for k in hist:
        #     uv = zip(fields, k)
        #     uv.append((count_name, hist[k]))
        #     data_point = {}
        #     for u, v in uv:
        #         data_point[u] = v
        #     # data_point = {}
        #     # for f in fields:
        #     #     data_point[f] = 
        #     ret.add_data_point(**data_point)

    def to_pdf(self, count_var, cumulative=False, fixed_scale=False):
        asdf
        sum = 0
        new_mpd = MyPlotData()
        new_mpd.append(self)
        for entry in new_mpd.data:
            if count_var not in entry:
                continue
            sum += entry[count_var]
        if fixed_scale:
            sum = fixed_scale
        last = 0
        # for entry in self.data:
        for i in range(len(new_mpd.data)):
            if count_var not in new_mpd.data[i]:
                continue
            before = new_mpd.data[i][count_var]
            new_mpd.data[i][count_var] = (new_mpd.data[i][count_var]+last) / sum
            if cumulative:
                last += before
        return new_mpd
    def to_cdf(self, count_var):
        return self.to_pdf(count_var, cumulative=True)

    def add_pdf(self, key, append_key='pdf', cumulative=False):
        sum = 0
        for entry in self.data:
            if key not in entry:
                continue
            sum += entry[key]
        append_key = key + "_" + append_key
        last = 0
        for entry in self.data:
            if key not in entry:
                continue
            before = entry[key]
            entry[append_key] = (before+last)/sum
            if cumulative:
                last += before
        self.fields.add(append_key)
        return self
    def add_cdf(self, key, append_key='cdf'):
        return self.add_pdf(key, cumulative=True, append_key='cdf')



    def to_dict(self):
        data_dict = {}
        for f in self.fields:
            data_dict[f] = []
        for item in self.data:
            for f in self.fields:
                if f not in item:
                    data_dict[f].append(None)
                else:
                    data_dict[f].append(item[f])
        return data_dict
    def save_data(self):
        '''For saving purposes'''
        return (self.fields, self.data)
    def load_data(self, data):
        '''For saving purposes'''
        self.fields, self.data = data
    def to_dataframe(self):
        data_dict = self.to_dict()
        return pd.DataFrame.from_dict(data_dict)
