# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 13:49:41 2016

@author: Johannes
@author: Guillaume
@author: Sebastian
"""
from collections import defaultdict
from collections import Counter
import numpy as np
import re
import warnings
# from lei.utils.bucketing import make_input_output_buckets


_EOD_TOK = '<EOD/>'


def split_sentence(s):
    s = s.replace('?', ' ?')
    return s.split()


def tokenize_strings(strings, indexer=None):
    if not indexer:
        indexer = Indexer()
    else:
        indexer.remember_state()
    res = []
    for x in strings:
        res.append([indexer.string_to_int(tok) for tok in split_sentence(x)])
    return res, indexer


def make_slot_strings(n):
    return ['<SLOT%d>' % i for i in range(n)]


def tokenize_input_outputs_thr(pairs, count_thrs, indexer=None, eos_id=None, go_id=None, word_tokenizer=split_sentence,
                               slot_strings=None, slot_pattern=None):
    """

    Args:
        pairs: list of (input, output) strings tokenized by the word_tokenizer function
        count_thrs: (thr_in, thr_out) are the threshold on the number of occurrences for input and output token
        indexer:
        eos_id:
        go_id:
        split:

    Returns:

    >>> pairs = [('the cat on the mat', 'cat mat'), ('the dog barks', 'dog barks'), ('a cat and dog', 'cat dog')]
    >>> a, b = tokenize_input_outputs_thr(pairs, 2)
    >>> a
    [([7, 5, 0, 7, 1], [5, 0]), ([7, 6, 0], [6, 0]), ([0, 5, 1, 6], [5, 6])]
    >>> b
    Indexer(<SLOT0>, <SLOT1>, <SLOT2>, <SLOT3>, <SLOT4>, cat, dog, the) with references [0, 7]
    """
    if isinstance(count_thrs, int):
        count_thrs = (count_thrs, count_thrs)
    thr_in, thr_out = count_thrs
    # initialize indexer
    if not indexer:
        indexer = Indexer()
    if eos_id:
        end = [indexer.string_to_int(eos_id)]
    else:
        end = []
    if go_id:
        start = [indexer.string_to_int(go_id)]
    else:
        start = []
    if slot_pattern:  # make sure the vocabulary do not contain tokens that match the slot pattern
        output_toks = [[x for x in word_tokenizer(s_out) if not slot_pattern.match(x)] for s_in, s_out in pairs]
        input_toks = [[x for x in word_tokenizer(s_in) if not slot_pattern.match(x)] for s_in, s_out in pairs]
    else:
        output_toks = [[x for x in word_tokenizer(s_out)] for s_in, s_out in pairs]
        input_toks = [[x for x in word_tokenizer(s_in)] for s_in, s_out in pairs]

    if slot_strings is None:
        slot_strings = make_slot_strings(np.max([len(x) for x in input_toks]))
    for slot_string in slot_strings:
        indexer.string_to_int(slot_string)

    # outputs
    counts_out = Counter([x for s in output_toks for x in s])
    for k, v in counts_out.items():
        if v >= thr_out:
            indexer.string_to_int(k)
    indexer.remember_state()

    # inputs
    counts_in = Counter([x for s in input_toks for x in s])
    for k, v in counts_in.items():
        if v >= thr_in:
            indexer.string_to_int(k)

    indexer.freeze()

    return tokenize_with_slots(pairs, word_tokenizer, indexer, slot_strings, start, end, slot_pattern), indexer


def tokenize_with_slots(pairs, word_tokenizer, indexer, slot_strings, start, end, slot_pattern,
                        input_only=False, return_local_vocab=False):
    def delexicalize(toks, slot_indexer0):
        new_toks = [delexicalize_patterns(tok, slot_indexer0) for tok in toks]
        return [delexicalize_rare_tokens(tok, slot_indexer0) for tok in new_toks]

    def delexicalize_patterns(tok, slot_indexer0):
        if slot_pattern and slot_pattern.match(tok):
            idx = slot_indexer0.string_to_int(tok)
            if idx < len(slot_strings):
                return slot_strings[idx]
            else:
                return slot_strings[-1]
        else:
            return tok

    def delexicalize_rare_tokens(tok, slot_indexer0):
        if tok not in indexer.index:
            idx = slot_indexer0.string_to_int(tok)
            if idx < len(slot_strings):
                return slot_strings[idx]
            else:
                return slot_strings[-1]
        else:
            return tok

    inputs = []
    local_vocabs = []
    if input_only:
        for x, _ in pairs:
            slot_indexer = Indexer()
            inputs.append([indexer.string_to_int(x) for x in delexicalize(word_tokenizer(x), slot_indexer)])
            local_vocabs.append(slot_indexer)
        return [x for x in zip(inputs, local_vocabs)]
    else:
        outputs = []
        for x, y in pairs:
            slot_indexer = Indexer()
            inputs.append([indexer.string_to_int(x) for x in delexicalize(word_tokenizer(x), slot_indexer)])
            outputs.append(start + [indexer.string_to_int(x) for x in delexicalize(word_tokenizer(y), slot_indexer)] + end)
            local_vocabs.append(slot_indexer)
        if return_local_vocab:
            return [x for x in zip(inputs, outputs, local_vocabs)]
        else:
            return [x for x in zip(inputs, outputs)]


def tokenize_input_outputs(pairs, indexer=None, eos_id=None, go_id=None, split=split_sentence):
    """
    List tokenizer

    Args:
        pairs: input/output pairs. Each input or output is a space-separated string that will be tokenized using split
        indexer: initial indexer

    Notes
        The outputs are tokenized first to guarantee a compact vocabulary

    Returns:
        list of output vocabulary

    Examples:
        >>> z, voc = tokenize_input_outputs([('one plus two', '1 + 2'), ('two + 3', '2 plus 3')])
        >>> [x for x in z]
        [([5, 3, 6], [0, 1, 2]), ([6, 1, 4], [2, 3, 4])]
        >>> voc
        Indexer(1, +, 2, plus, 3, one, two) with references [0, 5]
    """
    if not indexer:
        indexer = Indexer()
    else:
        indexer.remember_state()
    if eos_id:
        end = [indexer.string_to_int(eos_id)]
    else:
        end = []
    if go_id:
        start = [indexer.string_to_int(go_id)]
    else:
        start = []
    outputs = []
    for _, y in pairs:
        outputs.append(start + [indexer.string_to_int(tok) for tok in split(y)] + end)
    indexer.remember_state()
    inputs = []
    for x, _ in pairs:
        inputs.append([indexer.string_to_int(tok) for tok in split(x)])
    return [x for x in zip(inputs, outputs)], indexer


# def tokenize_bucketize_input_outputs(pairs, max_buckets=None, bucket_sizes=None, indexer=None, count_thrs=3,
#                                      eos_id=None, go_id=None, slot_strings=None):
#     toks, voc = tokenize_input_outputs_thr(pairs, count_thrs, indexer=indexer, eos_id=eos_id, go_id=go_id,
#                                            slot_strings=slot_strings)
#     if not bucket_sizes:
#         bucket_sizes = make_input_output_buckets(toks, max_buckets=max_buckets)
#     return bucketize(toks, bucket_sizes), voc, bucket_sizes


def bucketize(pairs, bucket_sizes, trim=True):
    """
    Put data into (input, output) buckets
    Args:
        pairs: (input, output) pairs where input and outputs are lists
        bucket_sizes: sizes of the buckets
        trim: boolean that indicates whether large sequences should be trimmed to fit in the largest bucket

    Returns:
        A tuple with as many elements as they a buckets. Each element is a list of (input, output) pair

    Examples:
        >>> p = [([5, 3], [0, 0, 1, 2]), ([6, 1, 0, 4], [3, 4]), ([1], [2, 2]), ([1, 2, 3, 4], [1, 2, 3, 4])]
        >>> bucks = bucketize(p, [(3, 3), (3, 6), (6, 6)])
        >>> bucks[0]
        [([1], [2, 2])]
        >>> bucks[1]
        [([5, 3], [0, 0, 1, 2])]
        >>> bucks[2]
        [([6, 1, 0, 4], [3, 4]), ([1, 2, 3, 4], [1, 2, 3, 4])]
    """
    bucks = [[] for _ in range(len(bucket_sizes))]
    for x, y in pairs:
        found = False
        for i, (in_sz, out_sz) in enumerate(bucket_sizes):
            if len(x) <= in_sz and len(y) <= out_sz:
                bucks[i].append((x, y))
                found = True
                break
        if not found:
            if trim:
                max_in, max_out = bucket_sizes[-1]
                bucks[-1].append((x[:max_in + 1], y[:max_out + 1]))
            else:
                raise BaseException('A sample of length (%d, %d) does not fit in any of these buckets %s'
                                    % (len(pair[0]), len(pair[1]), str(bucket_sizes)))
    return tuple(bucks)



class Indexer(object):
    """
    Mapping from strings to integers and vice versa.

    New words get new indices, starting at 0
    >>> index = Indexer()
    >>> index('cat')
    0

    The next new word gets the next index
    >>> index('the')
    1

    They keep their indices when called again
    >>> index('cat')
    0

    The index remembers the inverse mapping
    >>> index.inv(0)
    'cat'

    Batch calls:
    >>> index.ints('the','cat')
    [1, 0]

    >>> index.strings(1, 0)
    ['the', 'cat']

    Freeze the dictionary
    >>> index.freeze()
    >>> index('dog')
    Traceback (most recent call last):
     ...
    KeyError: 'dog not indexed yet and indexer is frozen'

    >>> index.remember_state()
    Traceback (most recent call last):
    ...
    ValueError: Cannot remember the state of an Indexer that is frozen.
    >>> index.freeze(False)
    >>> index('dog')
    2
    >>> index('giraffe')
    3
    >>> index('mouse')
    4
    >>> index.remember_state()
    >>> index('bee')
    5
    >>> index.reference_ids
    [0, 5]

    """

    def __init__(self, first_words=()):
        self._index = {}
        self._index_to_string = []
        self._frozen = False
        self._reference_ids = [0]
        for w in first_words:
            self.string_to_int(w)

    def freeze(self, frozen = True):
        self._frozen = frozen

    def remember_state(self):
        if self._frozen:
            raise ValueError('Cannot remember the state of an Indexer that is frozen.')
        self._reference_ids.append(len(self._index_to_string))

    def string_to_int(self, string):
        if string in self._index:
            return self._index[string]
        else:
            if not self.frozen:
                result = len(self._index_to_string)
                self._index[string] = result
                self._index_to_string.append(string)
                return result
            else:
                raise KeyError('{} not indexed yet and indexer is frozen'.format(string))

    def int_to_string(self, int):
        return self._index_to_string[int]

    def inv(self, string):
        return self.int_to_string(string)

    def __call__(self, string):
        return self.string_to_int(string)

    def __iter__(self):
        return self._index.__iter__()

    def items(self):
        return self._index.items()

    def ints(self, *strings):
        return [self.string_to_int(string) for string in strings]

    def strings(self, *ints):
        return [self.int_to_string(i) for i in ints]

    def __len__(self):
        return len(self._index_to_string)

    @property
    def theindex(self):
        print('Deprecated method "theindex". Use index instead')
        return self._index

    @property
    def index(self):
        return self._index

    @property
    def index(self):
        return self._index

    @property
    def reference_ids(self):
        return self._reference_ids

    @property
    def frozen(self):
        return self._frozen

    @property
    def frozen(self):
        return self._frozen

    def __str__(self):
        l = len(self._index_to_string)
        if l > 20:
            a = min(l, 10)
            b = max(len(self._index_to_string) - 10, 0)
            mid = ', ..., '
        else:
            a, b = l, l
            mid = ''
        return "%s(%s%s%s)" % (self.__class__.__name__,
                               ', '.join([str(x) for x in self._index_to_string[:a]]),
                               mid, ', '.join([str(x) for x in self._index_to_string[b:]]))

    def __repr__(self):
        s = str(self)
        return s + ' with references %s' % str(self._reference_ids)


class IndexerWithCounts(Indexer):
    """
    Same as indexer but counts the number of occurrences of the words before the object is frozen
    """
    def __init__(self, first_words=()):
        self.counts = []
        super().__init__(first_words)

    def string_to_int(self, string):
        if string in self.index:
            idx = self.index[string]
            if not self.frozen:
                self.counts[idx] += 1
            return idx
        else:
            if not self.frozen:
                result = len(self._index_to_string)
                self.index[string] = result
                self._index_to_string.append(string)
                self.counts.append(1)
                return result
            else:
                raise KeyError('{} not indexed yet and indexer is frozen'.format(string))

    def __repr__(self):
        l = len(self._index_to_string)
        if l > 20:
            a = min(l, 10)
            b = max(len(self._index_to_string) - 10, 0)
            mid = ', ..., '
        else:
            a, b = l, l
            mid = ''
        beg = ', '.join(['%s(%d)' % (e, c) for e, c in zip(self._index_to_string[:a], self.counts[:a])])
        end = ', '.join(['%s(%d)' % (e, c) for e, c in zip(self._index_to_string[b:], self.counts[b:])])

        return "%s(%s%s%s)" % (self.__class__.__name__, beg, mid, end)


class Vocabulary(Indexer):
    def __init__(self, first_words=()):
        super().__init__(first_words)
        warnings.warn('Use Index instead of Vocabulary', DeprecationWarning)

    def __str__(self):
        return 'Vocabulary ' + str(self.word2idx)

    @property
    def word2idx(self):
        return self.index

    @property
    def idx2word(self):
        return self.index_to_string

    def get_idx(self, word):
        return self.string_to_int(word)

    def str2tok(self, sentence, update=True):
        if update:
            return [self.get_idx(w) for w in sentence_split(sentence)]
        else:
            return [self.word2idx[w] for w in sentence_split(sentence)]

    def get_word(self, idx):
        return self.index_to_string[int(idx)]

    def toks2str(self, ut):
        return ' '.join([self.index_to_string[t] for t in ut]).replace(' ?', '?').replace(' .', '.')


# utility functions

def sentence_split(s):
    s = re.sub(' +', ' ', s.replace('?', ' ?').replace('.', ' .'))
    return s.split(' ')

"""
1) GlobalDict One dictionary for EVERY possible entry in any of the types/domains
"""


def convert_String_Tuples_to_Integer_Tuples(string_tuples, String2Int, interactions):
    int_representations = []
    for string_tuple in string_tuples:
        unary = []
        for i_s, s in enumerate(string_tuple):
            s = str(i_s) + ">>" + s
            unary += [ String2Int[s] ]

        binary = []
        for interaction in interactions:
            i1,i2 = interaction[1]
            type_indicator = str(i1) + "," +  str(i2)
            s = string_tuple[i1] + "::::" + string_tuple[i2]
            s = type_indicator + ">>" + s
            binary += [ String2Int[s] ]
        int_representations += [unary + binary]
    return int_representations



#interactions = [(3,(0,1)), (4,(1,2))]: interactions between types.
#format example: (3,(0,1)). 3: column where this interaction will be located
#(0,1) types that interact in this column.
def DefineGlobalDict(string_tuples, interactions = None):
    String2Int = defaultdict(int)
    Int2String = {}

    dictStrings = set([])

    L = len(string_tuples)
    for i_st, str_tuple in enumerate(string_tuples):
        if not i_st % 1000:
            print ("Defining Global Dict.. ", i_st, "/", L)
        for i_type, s in enumerate(str_tuple):
            # collect entities for each type, with no type interaction.
            string = str(i_type) + ">>" + s
            dictStrings.add(string)

            # now allow for type interaction for every
        for interaction in interactions:
            interaction = interaction[1]    #first element is column
            type_indicator = str(interaction[0]) + "," +  str(interaction[1])
            s = str_tuple[interaction[0]] + "::::" + str_tuple[interaction[1]]
            string = type_indicator + ">>" + s
            #if string not in dictStrings:
            #    dictStrings += [string]
            dictStrings.add(string)


    for i_string, string in enumerate(list(dictStrings)):
        String2Int[string] = i_string + 1  #start indexing at 1; 0 reserved for not in dict.
        Int2String[i_string + 1] = string
    return String2Int, Int2String




# demo of usage
if 0:
    interactions = [(3,(0,1)), (4,(1,2))]
    string_tuples = [["ent1", "rel1", "ent2"],
                     ["ent3", "rel2", "ent4"],
                     ["ent5", "rel3", "ent6"],
                     ["ent7", "rel4", "ent8"],
                     ["ent9", "rel5", "ent10"],
                     ["ent11", "rel6", "ent12"]]
    t1 = string_tuples[0]


    String2Int, Int2String = DefineGlobalDict(string_tuples, interactions)

    convert_String_Tuples_to_Integer_Tuples([t1], String2Int, interactions)

