import numpy
import theano


def seqs2matrix(seqs, dim=None):
    """receives the preprocessed data, computes the max length of a sequence,
    and return matrice maxlen*num_samples

    :seqs
    :dim
    :returns: TODO

    """
    lengths = [len(s) for s in seqs]

    n_samples = len(seqs)
    maxlen = numpy.max(lengths)

    if dim is not None:
        assert dim >= maxlen, 'you might lose data if having %s dimensions' % dim
        maxlen = dim

    x = numpy.zeros((maxlen, n_samples)).astype('int64')
    x_mask = numpy.zeros((maxlen, n_samples)).astype(theano.config.floatX)
    for idx, s in enumerate(seqs):
        x[:lengths[idx], idx] = s
        x_mask[:lengths[idx], idx] = 1.

    return x, x_mask
