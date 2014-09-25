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

class SplSgn(UnaryScalarOp):
    def impl(self, x):
        #casting to output type is handled by filter
        return numpy.sign(x)

    def grad(self, (x, ), (gz, )):
        if x.type in continuous_types:
            return gz,
        else:
            return None,

    def c_code(self, node, name, (x, ), (z, ), sub):
        #casting is done by compiler
        #TODO: use copysign
        type = node.inputs[0].type
        if type in float_types:
            return "%(z)s = (%(x)s >= 0) ? (%(x)s == 0) ? 0.0 : 1.0 : -1.0;" % locals()
        if type in int_types:
            return "%(z)s = (%(x)s >= 0) ? (%(x)s == 0) ? 0 : 1 : -1;" % locals()
        raise TypeError()  # complex has no sgn

    def c_code_cache_version(self):
        s = super(SplSgn, self).c_code_cache_version()
        if s:
            return (3,) + s
        else:  # if parent is unversioned, we are too
            return s

splsgn = SplSgn(same_out_nocomplex, name='splsgn')

sigm = T.nnet.sigmoid
tanh = T.tanh
ce = lambda x, y: T.nnet.binary_crossentropy(x, y)

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
        self.W = theano.shared(numpy.random.normal(size=(dim, ), scale=0.00001))
        self.params = [self.W]

    def get_cost(self, x):
        return ce(sigm(self.W), x)


def corrupt_SnP(x, corruption_level):
    if corruption_level == 0.0:
        return x
    a = rng_theano.binomial(size=x.shape, n=1, p=1-corruption_level)
    #b = rng_theano.binomial(size=x.shape, n=1, p=0.5)
    b = rng_theano.uniform(size=x.shape, low=0.0, high=1.0)
    return x * a + T.eq(a, 0) * b

class GAE():
    def __init__(self, dimX, dimH):
        self.encoder = MLP(dimX, dimH, [1200], [tanh, sigm])
        self.decoder = MLP(dimH, dimX, [1200], [tanh, sigm])
        self.toplayer = Multinomial(dimH)
        self.X = T.fmatrix('X')
        self.lr = T.scalar('lr')
        self.H = self.encoder(self.X)
        self.SH = T.sgn(self.H)
        self.RX = self.decoder(self.SH)
        cost_recons = (ce(self.RX, self.X).sum(axis=1)/784).mean(axis=0)
        cost_prior = (self.toplayer.get_cost(self.SH).sum(axis=1)/784).mean(axis=0)

        #cost_derivative = ((self.H - self.H**2).sum(axis=1)/784).mean(axis=0)
        cost = cost_recons + cost_prior# + cost_derivative
        params = self.encoder.params + self.decoder.params + self.toplayer.params
        grads = T.grad(cost, params)
        updates = map(lambda (param, grad): (param, param - self.lr * grad), zip(params, grads))
        self.train_fn = theano.function([self.X, self.lr], [cost_recons, cost_prior], updates=updates)

    def train(self, dataset, mbsz, epochs, lr):
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
            self.dump_recons(dataset, 15, 15, e)
            self.dump_samples(15, 15, e)
            print e, (t2 - t1), costs / float(len(range(0, len(dataset), mbsz))), lr
            lr *= 0.99

    def dump_recons(self, D, x, y, name):
        DD = self.decoder(self.encoder(D[:x*y])).eval()
        draw_mnist(DD, 'recons/', x, y, name)

    def dump_samples(self, x, y, name):
        w = sigm(model.toplayer.W).eval()
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
    model = GAE(784, 200)
    D = numpy.load("../mnist.npy")
    D = (D > 0.5).astype('float32')
    model.train(D, 100, 100, 1.)





