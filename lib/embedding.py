import numpy
from theano import config
from collections import OrderedDict


def init_params(options):
    """Global (not LSTM) parameter. For the embedding and the classifier.

    :options: TODO
    :returns: TODO

    """
    params = OrderedDict()
    # embedding
    randn = numpy.random.rand(options['n_words'], options['dim_proj'])
    params['Wemb'] = (0.01 * randn).astype(config.floatX)

    # classifier
    params['U'] = 0.01 * numpy.random.randn(options['dim_proj'],
                                            options['ydim']).astype(config.floatX)
    params['b'] = numpy.zeros((options['ydim'],)).astype(config.floatX)

    return params
