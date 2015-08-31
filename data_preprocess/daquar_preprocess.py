"""
preprocess daquar qa test & train raw text files
single answer mode
"""

import numpy
import cPickle

import os

from subprocess import Popen, PIPE

# tokenizer.perl is from
# Moses: https://github.com/moses-smt/mosesdecoder/tree/master/scripts/tokenizer
tokenizer_cmd = ['./tokenizer.perl', '-l', 'en', '-q', '-']


def tokenize(sentences):

    print 'Tokenizing..',
    text = "\n".join(sentences)
    tokenizer = Popen(tokenizer_cmd, stdin=PIPE, stdout=PIPE)
    tok_text, _ = tokenizer.communicate(text)
    toks = tok_text.split('\n')[:-1]
    print 'Done'

    return toks


def build_dict(path):
    sentences = []
    # currdir = os.getcwd()
    # os.chdir('%s/pos/' % path)
    # for ff in glob.glob("*.txt"):
    with open(os.path.join(path, 'qa.894.raw.train.txt'), 'r') as f:
        for line in f:
            sentences.append(line.strip())
    # os.chdir('%s/neg/' % path)
    # for ff in glob.glob("*.txt"):
    # with open(ff, 'r') as f:
            # sentences.append(f.readline().strip())
    # os.chdir(currdir)

    # sentences = tokenize(sentences)

    print 'Building dictionary..',
    wordcount = dict()
    for ss in sentences:
        words = ss.strip().lower().split()
        for w in words:
            if w.startswith('image'):
                w = 'image'
            if w not in wordcount:
                wordcount[w] = 1
            else:
                wordcount[w] += 1

    counts = wordcount.values()
    keys = wordcount.keys()

    sorted_idx = numpy.argsort(counts)[::-1]

    worddict = dict()

    for idx, ss in enumerate(sorted_idx):
        worddict[keys[ss]] = idx+2  # leave 0 and 1 (UNK)

    print numpy.sum(counts), ' total words ', len(keys), ' unique words'

    return worddict


def grab_data(path, dictionary):
    sentences = []
    # currdir = os.getcwd()
    # os.chdir(path)
    # for ff in glob.glob("*.txt"):
    with open(path, 'r') as f:
        for line in f:
            sentences.append(line.strip())
    # os.chdir(currdir)
    # sentences = tokenize(sentences)

    seqs = [None] * len(sentences)
    for idx, ss in enumerate(sentences):
        words = ss.strip().lower().split()
        seqs[idx] = []
        for w in words:
            if w.startswith('image'):
                w = 'image'
            if w in dictionary:
                seqs[idx].append(dictionary[w])
            else:
                seqs[idx].append(1)
        # seqs[idx] = [dictionary[w] if w in dictionary else 1 for w in words]

    return seqs


def xy_split(xy):
    """splits into x and y since evans are ques and odds are single word answer

    :xy: list of lists
    :returns: x, y

    """
    x = []
    y = []
    for i in range(0, len(xy), 2):
        x.append(xy[i])
        y.append(xy[i+1])
    return x, y


def get_data(path='./data/daquar_only_text/', load=True):
    if load is True:
        if os.path.exists(path + 'text_data.pkl') and \
                os.path.exists(path + 'dict.pkl'):
            print 'Loading data from previously saved preprocessed daquar dataset ...'
            with open(path + 'text_data.pkl') as f:
                train_set = cPickle.load(f)
                test_set = cPickle.load(f)
            with open(path + 'dict.pkl') as f:
                dictionary = cPickle.load(f)
            return (train_set, test_set), dictionary

    print 'Preprocessing daquar raw text dataset ...'
    dictionary = build_dict(path)

    train_whole = grab_data(path+'qa.894.raw.train.txt', dictionary)
    train_x, train_y = xy_split(train_whole)

    test_whole = grab_data(path+'qa.894.raw.test.txt', dictionary)
    test_x, test_y = xy_split(test_whole)

    f = open(path + 'text_data.pkl', 'wb')
    cPickle.dump((train_x, train_y), f, -1)
    cPickle.dump((test_x, test_y), f, -1)
    f.close()

    f = open(path + 'dict.pkl', 'wb')
    cPickle.dump(dictionary, f, -1)
    f.close()

    return ((train_x, train_y), (test_x, test_y)), dictionary

if __name__ == '__main__':
    path = '../data/daquar_only_text/'
    (train, test), dictionary = get_data(path)
