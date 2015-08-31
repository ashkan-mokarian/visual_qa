'''
provides data manipulation function such as pruning to just a single answer,
changing sequences from list format to matrices
'''

import cPickle
import os

import numpy

from data_preprocess import daquar_preprocess


def seq2matrix(seqs):
    """return the matrix form of a list, with padding by 0

    :seqs: list of lists which are sequences
    :returns: narray(maxlen, num_samples)

    """
    lengths = [len(s) for s in seqs]
    n_samples = len(seqs)
    maxlen = numpy.max(lengths)

    x = numpy.zeros((maxlen, n_samples)).astype('int64')
    for idx, s in enumerate(seqs):
        x[:lengths[idx], idx] = s
    return x


def structured_preprocess(path):
    """loads the already preprocessed data,  from predefined files text_data.pkl
    and dict.pkl

    :path: path to folder which contains dict.pkl and text_data.pkl
    :returns: ((train_x, train_y), (test_x, test_y)), dictionary

    """
    text_data_file = path + 'text_data.pkl'
    dict_file = path + 'dict.pkl'
    assert os.path.exists(text_data_file), 'could not find %s to load \
        structured data, must follow specific rules of preprocessing' % \
        text_data_file
    print 'Loading data from file %s' % text_data_file
    with open(text_data_file) as f:
        train_set = cPickle.load(f)
        test_set = cPickle.load(f)
    dictionary = None
    if os.path.exists(dict_file):
        with open(dict_file) as f:
            dictionary = cPickle.load(f)
    return (train_set, test_set), dictionary


datasets = {
    'daquar_text_only': daquar_preprocess.get_data,
    'structured': structured_preprocess
}


def load_data(dataset_name=None, path=None, valid_portion=0.1,
              sort_by_len=True):
    """Loads the dataset, and return train, valid, test samples. if path given,
    omits the default ones and looks for structure in the path, i.e.
    text_data.pkl and dict.pkl files which are preprocessed forms of the raw
    dataset data

    :dataset_name: default datasets name, as given in datasets dictionary
    :path: if not None, omits any default dataset paths and looks for
    text_data.pkl, dict.pkl files
    :valid_portion: TODO
    :sort_by_len: supposedly, shows faster performance?
    :returns: (train_x, train_y), (valid_x, valid_y),(test_x, test_y),dictionary

    """
    assert dataset_name is not None or path is not None, \
        'no dataset specified, either use a default one or assign a path'
    if path is not None:
        (train_set, test_set), dictionary = \
            datasets['structured'](path)
    else:
        (train_set, test_set), dictionary = \
            datasets[dataset_name]()

    # splitting training set into validation set
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = numpy.random.permutation(n_samples)
    n_train = int(numpy.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    # train_set = (train_set_x, train_set_y)
    # valid_set = (valid_set_x, valid_set_y)
    test_set_x, test_set_y = test_set

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    if sort_by_len:
        sorted_index = len_argsort(test_set_x)
        test_set_x = [test_set_x[i] for i in sorted_index]
        test_set_y = [test_set_y[i] for i in sorted_index]

        sorted_index = len_argsort(valid_set_x)
        valid_set_x = [valid_set_x[i] for i in sorted_index]
        valid_set_y = [valid_set_y[i] for i in sorted_index]

        sorted_index = len_argsort(train_set_x)
        train_set_x = [train_set_x[i] for i in sorted_index]
        train_set_y = [train_set_y[i] for i in sorted_index]

    train = (train_set_x, train_set_y)
    valid = (valid_set_x, valid_set_y)
    test = (test_set_x, test_set_y)

    return train, valid, test, dictionary


if __name__ == '__main__':
    train, valid, test, dictionary = load_data(dataset_name='daquar_text_only')
    import pdb; pdb.set_trace()  # XXX BREAKPOINT
