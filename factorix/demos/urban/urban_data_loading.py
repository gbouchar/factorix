
from collections import defaultdict
from random import shuffle
from copy import deepcopy
import numpy as np
from naga.shared.dictionaries import IndexerWithCounts

from factorix import FACTORIX_DIR


def load_area_aspects_rawdata(aspects=None, fold='train', verbose=True):
    dir = FACTORIX_DIR + '/demos/urban/'
    if aspects is None:
        aspects = {'multicultural', 'waterside', 'posh'}
    with open(dir + 'aspects/classification_london_ya_binary_uni_min_5.txt', 'r') as f:
        x1_tuples = [eval(t) for t in f.read().split()]
    with open(dir + 'aspects/aspect_word.txt', 'r') as f:
        x2_tuples = [eval(t) for t in f.read().split()]
        # print(x2_tuples)
        # print([t[0].split(':') for t, v in x2_tuples])
        x2_tuples = [(t, v) for t, v in x2_tuples if t[0].split(':')[1] in aspects]
    y_tuples = []
    for a in aspects:
        with open(dir + 'aspects/classification_london_aspects_' + a + 'fold_0_' + fold + '.txt', 'r') as f:
            y_tuples += [eval(t) for t in f.read().split()]
    if verbose:
        print('%d area-aspect pairs loaded' % len(y_tuples))
    return x1_tuples, x2_tuples, y_tuples


def _group(tuples, dim, max_len, replacement_key):
    dic = defaultdict(list)
    for t, v in tuples:
        t2 = list(t)
        t2[dim] = replacement_key
        dic[t[dim]].append((tuple(t2), v))
    for k, v in dic.items():
        if len(v) > max_len:
            v2 = deepcopy(v)
            shuffle(v2)
            dic[k] = v2[:max_len]
    return dic


def _tuples_to_value(yt):
    dic = {}
    for k, v in yt:
        dic[(k[0], k[1])] = v
    return dic


def _make_data(aspects, fold, verbose, max_n_words_per_area, max_n_words_per_aspect, rep_keys=(0, 1)):
    x1t, x2t, yt = load_area_aspects_rawdata(aspects, fold, verbose)
    y_values = _tuples_to_value(yt)
    g1 = _group(x1t, dim=0, max_len=max_n_words_per_area, replacement_key=rep_keys[0])
    g2 = _group(x2t, dim=0, max_len=max_n_words_per_aspect, replacement_key=rep_keys[1])
    data = []
    for k1, v1 in g1.items():
        for k2, v2 in g2.items():
            if (k1, k2) in y_values:
                y = y_values[(k1, k2)]
                data.append((v1 + v2, [(rep_keys, y)]))
    return data


def _index_data(data_str, voc: IndexerWithCounts=None):
    data = []
    voc = voc or IndexerWithCounts()
    for d_str in data_str:
        d = []
        for obs in d_str:
            d.append([(voc.ints(*t), v) for t, v in obs])
        data.append(d)
    return tuple(data), voc


def load_area_aspects_data(aspects, fold, verbose=True,
                           max_n_words_per_area=np.inf, max_n_words_per_aspect=np.inf, vocab=None):
    unk_keys = ('Area:UNK', 'Aspect:UNK')
    if vocab is None:
        vocab = IndexerWithCounts(unk_keys)

    data_str0 = _make_data(aspects, fold, verbose, max_n_words_per_area, max_n_words_per_aspect, unk_keys)
    return _index_data(data_str0, vocab)
    # [print(d1, '\n', d2) for d1, d2 in zip(data_str0, data0)]

if __name__ == "__main__":
    dat, vc = load_area_aspects_data(None, 'train', max_n_words_per_area=10, max_n_words_per_aspect=10)
    for e in dat:
        print(e)
    print(vc)
    print(len(dat))
