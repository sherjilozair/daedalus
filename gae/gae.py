import theano
import numpy
from theano import tensor as T
import time
from PIL import Image

import os
import sys
import gzip
import cPickle
import operator as op

import random
from theano.scalar.basic import UnaryScalarOp, same_out_nocomplex
from theano.tensor.shared_randomstreams import RandomStreams
rng_theano = RandomStreams()
rng_numpy = numpy.random

sigm = T.nnet.sigmoid
tanh = T.tanh
ce = lambda x, y: T.nnet.binary_crossentropy(x, y)

NOISE = 0.0
DIMZ = 200

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


class Multinomial():
    def __init__(self, dim):
        self.W = theano.shared(numpy.random.normal(size=(dim, ), scale=0.1))
        self.params = [self.W]

    def get_cost(self, x):
        return ce(sigm(self.W), x)


def corrupt_SnP(x, corruption_level):
    if corruption_level == 0.0:
        return x
    a = rng_theano.binomial(size=x.shape, n=1, p=1-corruption_level)
    b = rng_theano.binomial(size=x.shape, n=1, p=0.5)
    #b = rng_theano.uniform(size=x.shape, low=0.0, high=1.0)
    return x * a + T.eq(a, 0) * b

class GAE():
    def __init__(self, dimX, dimH, sizes, acts):
        self.dimX = dimX
        self.encoder = MLP(dimX, dimH, sizes, acts)
        self.decoder = MLP(dimH, dimX, sizes, acts)
        self.toplayer = Multinomial(dimH)
        self.X = T.fmatrix('X')
        self.lr = T.scalar('lr')
        self.H = self.encoder(self.X)
        self.SH = (T.ident(2.*self.H-1.)+1.)/2.
        self.CSH = corrupt_SnP(self.SH, NOISE)
        self.RX = self.decoder(self.CSH)
        cost_recons = ce(self.RX, self.X).sum(axis=1).mean(axis=0)
        cost_prior = self.toplayer.get_cost(self.SH).sum(axis=1).mean(axis=0)

        #cost_derivative = ((self.H - self.H**2).sum(axis=1)/784).mean(axis=0)
        cost = cost_recons + cost_prior# + cost_derivative
        cost = cost / 784.
        params = self.encoder.params + self.decoder.params + self.toplayer.params
        grads = T.grad(cost, params)
        updates = map(lambda (param, grad): (param, param - self.lr * grad), zip(params, grads))
        self.train_fn = theano.function([self.X, self.lr], [cost_recons, cost_prior], updates=updates)
        self.encode_fn = theano.function([self.X], self.SH)

    def train(self, dataset, epochs, mbsz, lr, lr_scale):
        for e in xrange(epochs):
            rng_numpy.shuffle(dataset)
            t1 = time.clock()
            costs = 0.
            for i in xrange(0, len(dataset), mbsz):
                cost = self.train_fn(dataset[i:i+mbsz], lr)
                cost = numpy.array(cost)
                #print cost
                costs += cost
            t2 = time.clock()
            if self.dimX == 784:
                self.dump_recons(dataset, 15, 15, e)
                self.dump_samples(15, 15, e)
            print e, (t2 - t1), costs / float(len(range(0, len(dataset), mbsz))), lr
            lr *= lr_scale

    def dump_recons(self, D, x, y, name):
        h = self.encoder(D[:x*y])
        sh = (T.ident(2.*h-1.)+1.)/2.
        DD = self.decoder(sh).eval()
        draw_mnist(DD, 'recons/', x, y, name)

    def dump_samples(self, x, y, name):
        w = sigm(self.toplayer.W).eval()
        r = numpy.random.uniform(size=(x*y, w.shape[0]))
        h = (r < w).astype('float32')
        s = self.decoder(h).eval()
        draw_mnist(s, 'samples/', x, y, name)



def draw_mnist(samples, output_dir, num_samples, num_chains, name):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    all = Image.new("RGB", (28*num_samples, 28*num_chains))
    for i in xrange(num_samples):
        for j in xrange(num_chains):
            pic = samples[i*num_chains+j].reshape(28, 28) * 255
            im = Image.fromarray(pic)
            all.paste(im, (28*i, 28*j))
    all.save(os.path.join(output_dir, 'samples_%d.png' % name))


if __name__ == '__main__':
    model1 = GAE(784, 200, [1200], [tanh, sigm])
    model2 = GAE(200, 200, [500], [tanh, sigm])
    model3 = GAE(200, 200, [500], [tanh, sigm])
    D = numpy.load("../mnist.npy")
    model1.train(D, 200, 100, 1., .995)
    H1 = model1.encode_fn(D).astype('float32')
    model2.train(H1, 200, 100, 1., .995)
    H2 = model2.encode_fn(H1).astype('float32')
    model3.train(H2, 200, 100, 1., .995)
    H3 = model3.encode_fn(H2).astype('float32')
    





