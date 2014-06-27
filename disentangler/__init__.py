import theano
import numpy
from theano import tensor as T

from PIL import Image

import os
import sys
import gzip
import cPickle
import operator as op

import random

from theano.tensor.shared_randomstreams import RandomStreams
rng_theano = RandomStreams()

sigm = T.nnet.sigmoid
tanh = T.tanh
ce = T.nnet.binary_crossentropy

class Layer():
    def __init__(self, n_in, n_out, act):
        self.act = act
        self.W = self.init_weight(n_in, n_out, act)
        self.b = self.init_bias(n_out)
        self.params = [self.W, self.b]

    def init_weight(self, n_in, n_out, act):
        a = numpy.sqrt(6. / (n_in + n_out))
        if act == sigm:
            a *= 4.
        return theano.shared(numpy.random.uniform(size=(n_in, n_out), low=-a, high=a))

    def init_bias(self, n_out):
        return theano.shared(numpy.zeros(n_out,))

    def __call__(self, inp):
        return self.act(T.dot(inp, self.W) + self.b)

class MLP():
    def __init__(self, n_in, n_out, hls, acts):
        self.layers = [Layer(*args) for args in zip([n_in]+hls, hls+[n_out], acts)]
        self.params = reduce(op.add, map(lambda l: l.params, self.layers))

    def __call__(self, inp):
        return reduce(lambda x, fn: fn(x), self.layers, inp)

def sample_bernoulli(pz):
    return rng_theano.binomial(n=1, p=pz, dtype='float32')

class Disentangler():
    def __init__(self, dimX, dimZ, hls, acts, lr):
        pzx = MLP(dimX, dimZ, hls, acts)
        x = T.fmatrix('x')
        pz = pzx(x)
        z = sample_bernoulli(pz)
        logpzz = (-ce(pz, z)).sum(axis=1).mean(axis=0)
        pzx_grads = T.grad(logpzz, pzx.params, consider_constant=[z])
        updates = map(lambda (param, grad): (param, param - lr * grad), zip(pzx.params, pzx_grads))
        self.train_fn = theano.function([x], logpzz, updates=updates)

    def train(self, D, epochs, mbsz):
        ind = range(D.shape[0])
        for e in xrange(epochs):
            random.shuffle(ind)
            cost = 0.0
            #self.dump_samples(D, e)
            for b in xrange(mbsz):
                bs = D[ind[mbsz * b: mbsz * (b+1)]]
                cs = self.train_fn(bs)
                cost += cs
            print e, cost / mbsz

if __name__ == '__main__':
    model = Disentangler(784, 100, [1200, 1200], [tanh, tanh, sigm], 0.01)
    with gzip.open(os.environ['MNIST']) as f:
        D = cPickle.load(f)[0][0]
    model.train(D, 500, 100)





