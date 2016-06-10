
from collections import defaultdict
from random import shuffle
from copy import deepcopy
import numpy as np
from naga.shared.dictionaries import IndexerWithCounts
from factorix import FACTORIX_DIR


def load_area_aspects_rawdata(verbose=True, x1=True, x2=True, y=True):
    dir = FACTORIX_DIR + '/demos/urban/'
    if not x1:
        x1_tuples = []
    else:
        with open(dir + 'aspects/area_feature.txt', 'r') as f:
            x1_tuples = [eval(t) for t in f.read().split()]
        if verbose:
            print('%d area-feature pairs loaded' % len(x1_tuples))
    if not x2:
        x2_tuples = []
    else:
        with open(dir + 'aspects/aspect_feature.txt', 'r') as f:
            x2_tuples = [eval(t) for t in f.read().split()]
        if verbose:
            print('%d aspect-feature pairs loaded' % len(x2_tuples))
    if not y:
        y_tuples = []
    else:
        with open(dir + 'aspects/area_aspect.txt', 'r') as f:
            y_tuples = [eval(t) for t in f.read().split()]
    if not y:
        y_tuples = []
    else:
        with open(dir + 'aspects/area_demographics.txt', 'r') as f:
            y_tuples += [eval(t) for t in f.read().split()]
        if verbose:
            print('%d area-aspect pairs loaded' % len(y_tuples))

    tuples = x1_tuples + x2_tuples + y_tuples
    return extract_types(tuples)

def extract_types(tuples):
    new_tuples = []
    for t, v in tuples:
        new_t = tuple([tuple(idx.split(':')) for idx in t])
        new_tuples.append((new_t, v))
    return new_tuples

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


from collections import Counter

def make_entities_by_type(data):
    entities_by_type = defaultdict(list)
    for t, v in data:
        for typ, ent in t:
            entities_by_type[typ].append(ent)
    for k in entities_by_type:
        entities_by_type[k] = sorted(Counter(entities_by_type[k]).items(), key=lambda x: -x[1])
    return entities_by_type

def random_split(data, proportion=None, n=None):
    if proportion is None:
        n_train = min(len(data), n)
    else:
        n_train = int(len(data) * proportion)
    perm = np.random.permutation(len(data))
    return [data[i] for i in perm[:n_train]], [data[i] for i in perm[n_train:]]

def story_generator(data, target_types, prop_test=0.25, prop_positives=0.5, nb_questions=2):
    entities_by_type = make_entities_by_type(data)
    print('\nTotal:')
    [print(k, len(v)) for k, v in entities_by_type.items()]
    for k in entities_by_type:
        entities_by_type[k] = random_split([w for w, c in entities_by_type[k]], 1 - prop_test)
    print('\nTrain:')
    [print(k, len(v[0])) for k, v in entities_by_type.items()]
    print('\nTest:')
    [print(k, len(v[1])) for k, v in entities_by_type.items()]

    [print(k, v[1]) for k, v in entities_by_type.items()]
    train = []
    test = []
    difficult_questions_pos = []
    difficult_questions_neg_all = []
    for tuple, v in data:
        belong_to_test = [idx in entities_by_type[typ][1] for typ, idx in tuple]
        if not any(belong_to_test):
            train.append((tuple, v))
        else:
            test.append((tuple, v))
            if all(belong_to_test) and all([typ in target_types[i] for i, (typ, id) in enumerate(tuple)]):
                if v == 1.0:
                    difficult_questions_pos.append((tuple, v))
                else:
                    difficult_questions_neg_all.append((tuple, v))
    print(len(difficult_questions_pos), len(difficult_questions_neg_all))
    n_neg = len(difficult_questions_pos) * (1.0 / prop_positives - 1)
    difficult_questions_neg = random_split(difficult_questions_neg_all, n=n_neg)[0]
    difficult_questions = difficult_questions_pos + difficult_questions_neg
    shuffle(difficult_questions)

    print(difficult_questions)
    return train, test, difficult_questions

if __name__ == "__main__":
    # dat, vc = load_area_aspects_data(None, 'train', max_n_words_per_area=10, max_n_words_per_aspect=10)

    data = load_area_aspects_rawdata(x1=True)
    train, test, difficult_questions = story_generator(data, ('Area', 'Aspect'), prop_test=0.33)
    print('n_train:', len(train))
    print('n_test:', len(test))
    print('n_questions:', len(difficult_questions))
    # print(set([a for (a, t), v in y]))
    # print(set([t for (a, t), v in y]))

    # print(vc)
    # print(len(dat))
