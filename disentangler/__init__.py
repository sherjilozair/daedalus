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
    def __init__(self, dimX, dimZ, hls, acts):
    	self.dimZ = dimZ
        self.f = MLP(dimX, dimZ, hls, acts)
        self.g = MLP(dimZ, dimX, hls, acts)
        self.params = self.f.params + self.g.params
        x = T.fmatrix('x')
        lr = T.scalar('lr')
        alpha = T.scalar('alpha')
        z = self.f(x)
        rx = self.g(z)
        cost_uniformity = self.uniformity(z)
        cost_recons = ce(rx, x).mean(axis=1).mean(axis=0)
        cost = cost_uniformity + alpha * cost_recons
        grads = T.grad(cost, self.params)
        updates = map(lambda (param, grad): (param, param - lr * grad), zip(self.params, grads))
        self.train_fn = theano.function([x, lr, alpha], [cost_uniformity, cost_recons], updates=updates)

        z = T.fmatrix('z')
        self.sample_fn = theano.function([z], self.g(z))
        self.infer_fn = theano.function([x], self.f(x))

    def uniformity(self, z):
        n = z.shape[0]
        d = z.shape[1]
        t1 = (4./3.)**d
        t2 = -2. * (1. + 2. * z - 2. * z**2).prod(axis=1).sum(axis=0) / n
        t3 = (2.**d) * (1 - abs(z[:, numpy.newaxis, :] - z)).prod(axis=2).sum() / (n**2)
        t4 = t1 - 2. + (2.**d)
        return (t1 + t2 + t3)/t4

    def train(self, D, epochs, mbsz, lr_init, lr_scale, alpha):
        ind = range(D.shape[0])
        lr = lr_init
        for e in xrange(epochs):
            cPickle.dump(self, open("models/model_%s.pkl" % e, 'w'))
            random.shuffle(ind)
            cost1 = 0.0
            cost2 = 0.0
            self.dump_samples(e)
            for b in xrange(mbsz):
                bs = D[ind[mbsz * b: mbsz * (b+1)]]
                [cs1, cs2] = self.train_fn(bs, lr, alpha)
                cost1 += cs1
                cost2 += cs2
            lr *= lr_scale
            print "f g", e, cost1 / mbsz, cost2 / mbsz, lr
        
    def dump_samples(self, name):
        C = 10
        T = 10
        samples = numpy.zeros((C * T, D.shape[1]))
        for t in xrange(T):
            z = numpy.random.uniform(low=0., high=1., size=(C, self.dimZ)).astype('float32')
            b = self.sample_fn(z)
            samples[C*t:C*(t+1)] = b
           
        draw_mnist(samples, 'samples/', T, C, name)

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
    model = Disentangler(784, 15, [1200, 1200], [tanh, tanh, sigm])
    with gzip.open(os.environ['MNIST']) as f:
        D = (cPickle.load(f)[0][0] > 0.5).astype('float32')
    model.train(D, 500, 100, 0.000001, 0.99, 20000000)





