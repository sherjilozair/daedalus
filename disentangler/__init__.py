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
    def __init__(self, dimX, dimZ, hls, acts, lvl):
        self.lvl = lvl
    	self.dimZ = dimZ
        self.f = MLP(dimX, dimZ, hls, acts)
        self.g = MLP(dimZ, dimX, hls, acts)
        self.params = self.f.params + self.g.params
        x = T.fmatrix('x')
        lr = T.scalar('lr')
        alpha = T.scalar('alpha')
        beta = T.scalar('beta')
        z = self.f(x)
        cz = salt_and_pepper(z, self.lvl)
        rx = self.g(cz)

        cost_uniformity = ((self.nearest_neighbour(z) - z)**2).min(axis=1).mean()
        cost_recons = ce(rx, x).mean(axis=1).mean(axis=0)
        cost = - beta * cost_uniformity + alpha * cost_recons
        grads = T.grad(cost, self.params)
        updates = map(lambda (param, grad): (param, param - lr * grad), zip(self.params, grads))
        self.train_fn = theano.function([x, lr, alpha, beta], [beta * cost_uniformity, alpha * cost_recons], updates=updates)

        z = T.fmatrix('z')
        self.sample_fn = theano.function([z], self.g(z), allow_input_downcast=True)
        self.infer_fn = theano.function([x], self.f(x), allow_input_downcast=True)

    def uniformity(self, z):
        n = z.shape[0]
        d = z.shape[1]
        t1 = (4./3.)**d
        t2 = -2. * (1. + 2. * z - 2. * z**2).prod(axis=1).sum(axis=0) / n
        t3 = (2.**d) * (1 - abs(z[:, numpy.newaxis, :] - z)).prod(axis=2).sum() / (n**2)
        t4 = t1 - 2. + (2.**d)
        return (t1 + t2 + t3)
    
    def nearest_neighbour(self, z):
        d = ((z - z[:, numpy.newaxis, :])**2).min(axis=2)
        ind = [(d + (d.max()+1)* T.eye(d.shape[0])).argmin(axis=0)]
        #ind = rng_theano.random_integers(size=(z.shape[0],), high=z.shape[0]-1)
        #ind = numpy.arange(100)
        #numpy.random.shuffle(ind)
        znn = z[ind]
        return znn
    
    def train(self, D, epochs, mbsz, lr_init, lr_scale, alpha, beta):
        ind = range(D.shape[0])
        lr = lr_init
        num_batches = D.shape[0] / mbsz
        for e in xrange(epochs):
            cPickle.dump(self, open("models/model_%s.pkl" % e, 'w'))
            random.shuffle(ind)
            cost1 = 0.0
            cost2 = 0.0
            self.dump_samples(D, e)
            self.dump_recons(D, e)
            for b in xrange(num_batches):
                bs = D[ind[mbsz * b: mbsz * (b+1)]]
                [cs1, cs2] = self.train_fn(bs, lr, alpha, beta)
                cost1 += cs1
                cost2 += cs2
                #print "f g minibatch", e, b, cs1, cs2, lr
            lr *= lr_scale
            print "f g", e, cost1 / num_batches, cost2 / num_batches, lr
        
    def dump_samples(self, D, name):
        C = 10
        T = 10
        samples = numpy.zeros((C * T, D.shape[1]))
        #z = numpy.random.uniform(size=(C, self.dimZ))
        #px = self.sample_fn(z)
        #x = sample_bernoulli_fn(px)
        px = x = D[:C]
        for t in xrange(T):
            samples[C*t:C*(t+1)] = px
            z = self.infer_fn(x)
            zz = salt_and_pepper_fn(z, self.lvl)
            px = self.sample_fn(zz)
            x = sample_bernoulli_fn(px)
        draw_mnist(samples, 'samples/', T, C, name)

    def dump_recons(self, D, name):
        C = 10
        T = 10
        samples = numpy.zeros((C * T, D.shape[1]))
        for t in xrange(T):
            z = self.infer_fn(D[C*t:C*(t+1)])
            px = self.sample_fn(z)
            samples[C*t:C*(t+1)] = px
        draw_mnist(samples, 'recons/', T, C, name)

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
    d = ((z - z[:, numpy.newaxis, :])**2).sum(axis=2)
    ind = [(d + (d.max()+1)* numpy.eye(d.shape[0])).argmin(axis=0)]
    znn = z[ind]
    return ((z - znn)**2).sum(axis=1).mean()

def salt_and_pepper(z, lvl):
    if lvl == 0.0:
        return z
    field = rng_theano.binomial(n=1,p=1-lvl, size=z.shape)
    r = rng_theano.uniform(size=z.shape, low=0., high=1.)
    return z * field + r * (1-field)

z = T.fmatrix('z')
lvl = T.scalar('lvl')
salt_and_pepper_fn = theano.function([z, lvl], salt_and_pepper(z, lvl), allow_input_downcast=True)
px = T.fmatrix('px')
sample_bernoulli_fn = theano.function([px], sample_bernoulli(px), allow_input_downcast=True)

if __name__ == '__main__':
    #model = Disentangler(784, 15, [1200, 1200], [tanh, tanh, sigm])
    #with gzip.open(os.environ['MNIST']) as f:
    #    D = (cPickle.load(f)[0][0] > 0.5).astype('float32')
    #model.train(D, 500, 100, 0.1, 0.99, 10.)

    model = Disentangler(784, 100, [1200, 1200], [tanh, tanh, sigm], 0.05)
    D = numpy.load("../mnist.npy")
    D = (D > 0.5).astype('float32')
    #D = D - numpy.mean(D, axis=0)
    #D = D / numpy.max(abs(D), axis=0)
    #D = (D + 1.) / 2.
    #D = D.astype('float32')
    model.train(D, 100, 100, .5, 0.99, 1., 0.)






