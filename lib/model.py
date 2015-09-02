'''Just a sample model as in LSTM tutorial on deep learning.
Need to find out another way to generalize this'''

import numpy
import theano
from theano import config
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import theano.tensor as tensor
import utils
from lstm import get_layer

SEED = 123
numpy.random.seed(SEED)


def build_model(tparams, options):
    """first model - blind single answer qa without masking the response at ?

    :tparams: TODO
    :options: TODO
    :returns: TODO

    """
    trng = RandomStreams(SEED)

    # Used for dropout
    use_noise = theano.shared(utils.numpy_floatX(0.))

    x = tensor.matrix('x', dtype='int64')
    mask = tensor.matrix('mask', dtype=config.floatX)
    y = tensor.vector('y', dtype='int64')

    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    emb = tparams['Wemb'][x.flatten()].reshape([n_timesteps,
                                                n_samples,
                                                options['dim_proj']])
    proj = get_layer(options['encoder'])[1](tparams, emb, options,
                                            prefix=options['encoder'],
                                            mask=mask)

    if options['use_dropout']:
        proj = utils.dropout_layer(proj, use_noise, trng)

    pred = tensor.nnet.softmax(tensor.dot(proj, tparams['U']) + tparams['b'])

    f_pred_prob = theano.function([x, mask], pred, name='f_pred_prob')
    f_pred = theano.function([x, mask], pred.argmax(axis=1), name='f_pred')

    off = 1e-8
    if pred.dtype == 'float16':
        off = 1e-6

    cost = -tensor.log(pred[tensor.arange(n_samples), y] + off).mean()

    return use_noise, x, mask, y, f_pred_prob, f_pred, cost
