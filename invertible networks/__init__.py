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
    def __init__(self, dim, act):
        self.act = act
        self.dim = dim
        interval = numpy.sqrt(3./dim**4)
        if act == sigm:
            interval *= 4.
        self.L = self.init_L(dim, interval)
        self.D = self.init_D(dim)
        self.b = self.init_b(dim)
        self.Lmask = numpy.tril(numpy.ones((dim, dim)), -1)
        self.params = [self.L, self.D, self.b]

    def init_L(self, dim, interval):
        w = numpy.random.randn(dim, dim) * interval
        l = numpy.tril(w, -1)
        return theano.shared(l)

    def init_D(self, dim):
        D = numpy.log(numpy.exp(1.)-1.) * numpy.ones((dim,))
        return theano.shared(D)

    def init_b(self, n_out):
        return theano.shared(numpy.zeros(n_out,))

    def get_W(self):
        L = T.eye(self.dim) + T.switch(self.Lmask, self.L, 0.)
        D = T.diag(T.nnet.softplus(self.D))
        W = T.dot(T.dot(L, D), L.T)
        return W

    def __call__(self, inp):
        return self.act(T.dot(inp, self.get_W()) + self.b)

class MLP():
    def __init__(self, dim, acts):
        numl = len(acts)
        self.layers = [Layer(*args) for args in zip([dim]*numl, acts)]
        self.params = reduce(op.add, map(lambda l: l.params, self.layers))

    def __call__(self, inp):
        return reduce(lambda x, fn: fn(x), self.layers, inp)

class InvertibleNetwork():
    def __init__(self, dim, acts):
        self.mlp = MLP(dim, acts)
        self.params = self.mlp.params
        self.x = T.fmatrix('x')
        self.zx = self.mlp(self.x)
        self.zxnn = self.nearest_neighbour(self.zx)
        self.objective = ((self.zx - self.zxnn)**2).min(axis=1).mean()
        #self.cost = self.uniformity(self.zx)
        self.grads = T.grad(self.objective, self.params)#, consider_constant=[self.zxnn])
        self.lr = T.scalar('lr')
        self.updates = map(lambda (param, grad): (param, param + self.lr * grad), zip(self.params, self.grads))
        self.train_fn = theano.function([self.x, self.lr], self.objective, updates=self.updates)
        self.infer_fn = theano.function([self.x], self.zx)

    def train(self, D, epochs, mbsz, lr):
        ind = range(D.shape[0])
        num_batches = D.shape[0] / mbsz
        prevcostn = 0.
        inc = 0
        print nearest_neighbour(numpy.random.randn(100, 784))
        for e in xrange(epochs):
            random.shuffle(ind)
            cost = 0.
            for b in xrange(num_batches):
                bs = D[ind[mbsz * b: mbsz * (b+1)]]
                cs = self.train_fn(bs, lr)
                cost += cs
                print "minibatch", e, b, cs, lr
            costn = cost / num_batches
            if (costn < prevcostn):
                inc += 1
                if inc > 0:
                    inc = 0
                    lr *= 0.5
            prevcostn = costn
            print "epoch", e, costn, lr, inc, nearest_neighbour(numpy.random.randn(100, 784))

    def nearest_neighbour(self, z):
        d = ((z - z[:, numpy.newaxis, :])**2).min(axis=2)
        ind = [(d + (d.max()+1)* T.eye(d.shape[0])).argmin(axis=0)]
        #ind = rng_theano.random_integers(size=(z.shape[0],), high=z.shape[0]-1)
        #ind = numpy.arange(100)
        #numpy.random.shuffle(ind)
        znn = z[ind]
        return znn

    def uniformity(self, z):
        n = z.shape[0]
        d = z.shape[1]
        t1 = (4./3.)**d
        t2 = -2. * (1. + 2. * z - 2. * z**2).prod(axis=1).sum(axis=0) / n
        t3 = (2.**d) * (1 - abs(z[:, numpy.newaxis, :] - z)).prod(axis=2).sum() / (n**2)
        t4 = t1 - 2. + (2.**d)
        return (t1 + t2 + t3)/t4

    def dump_samples(self, name):
        C = 10
        T = 10
        samples = numpy.zeros((C * T, D.shape[1]))
        for t in xrange(T):
            z = numpy.random.uniform(low=0., high=1., size=(C, 100)).astype('float32')
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

def nearest_neighbour(z):
    d = ((z - z[:, numpy.newaxis, :])**2).min(axis=2)
    ind = [(d + (d.max()+1)* numpy.eye(d.shape[0])).argmin(axis=0)]
    znn = z[ind]
    return ((z - znn)**2).min(axis=1).mean()

if __name__ == '__main__':
    model = InvertibleNetwork(784, [tanh]*1 + [sigm])
    D = numpy.load("../mnist.npy")
    #D = (D > 0.5).astype('float32')
    #D = D - numpy.mean(D, axis=0)
    #D = D / numpy.max(abs(D), axis=0)
    #D = (D + 1.) / 2.
    #D = D.astype('float32')
    model.train(D, 5, 100, .001)
















