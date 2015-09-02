from collections import OrderedDict
import numpy
import theano
from theano import config
import theano.tensor as tensor


def _p(pp, name):
    return '%s_%s' % (pp, name)


def numpy_floatX(data):
    return numpy.asarray(data, dtype=config.floatX)


def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """Used to shuffle the dataset at each iteration.

    :n: TODO
    :minibatch_size: TODO
    :shuffle: TODO
    :returns: TODO

    """
    idx_list = numpy.arange(n, dtype='int32')

    if shuffle:
        numpy.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)


def zpp(params, tparams):
    """When we reload the model. Needed for the GPU stuff.

    :params: TODO
    :tparams: TODO
    :returns: TODO

    """
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)


def unzip(zipped):
    """When we pickle the model. Needed for the GPU stuff.

    :zipped: TODO
    :returns: TODO

    """
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params


def load_params(path, params):
    """loads saved model parameters, needs the keys of the desired model first

    :path: TODO
    :params: TODO
    :returns: TODO

    """
    pp = numpy.load(path)
    for kk, vv in params.iteritems():
        if kk not in pp:
            raise Warning('%s is not in the archive' % kk)
        params[kk] = pp[kk]
    return params


def init_tparams(params):
    """transforms the parameters in theano shared variables

    :params: TODO
    :returns: TODO

    """
    tparams = OrderedDict()
    for kk, vv in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams


def dropout_layer(state_before, use_noise, trng):
    """TODO: Docstring for dropout_layer.

    :state_before: TODO
    :use_noise: TODO
    :trng: TODO
    :returns: TODO

    """
    proj = tensor.switch(use_noise,
                         (state_before *
                          trng.binomial(state_before.shape,
                                        p=0.5, n=1,
                                        dtype=state_before.dtype)),
                         state_before * 0.5)
    return proj
