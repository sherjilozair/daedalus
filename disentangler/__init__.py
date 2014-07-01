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
    def __init__(self, dimX, dimZ, hls, acts, lr, K):
        f = MLP(dimX, dimZ, hls, acts)
        x = T.fmatrix('x')
        z = f(x)
        costs_f = self.get_moments_error(z, K)
        cost_f = sum(costs_f)

        grads_f = T.grad(cost_f, f.params)
        updates_f = map(lambda (param, grad): (param, param - lr * grad), zip(f.params, grads_f))
        self.trainf_fn = theano.function([x], costs_f, updates=updates_f)

        g = MLP(dimZ, dimX, hls, acts)
        rx = g(z)
        cost_g = ce(rx, x).mean(axis=1).mean(axis=0)
        grads_g = T.grad(cost_g, g.params)
        updates_g = map(lambda (param, grad): (param, param - lr * grad), zip(g.params, grads_g))
        self.traing_fn = theano.function([x], cost_g, updates=updates_g)

        z = T.fmatrix('z')
        self.sample_fn = theano.function([z], g(z))

    def get_moments_costs(self, z, K):
        costs = [(((z**i).mean(axis=0) - (1./(i+1)))**2).mean() for i in xrange(1, K+1)] 
        return costs

    def train(self, D, epochsf, epochsg, mbsz):
        ind = range(D.shape[0])
        for e in xrange(epochsf):
            random.shuffle(ind)
            cost = [0] * len(10)
            #self.dump_samples(e)
            for b in xrange(mbsz):
                bs = D[ind[mbsz * b: mbsz * (b+1)]]
                costs = self.trainf_fn(bs)
                cost += sum(costs)/len(costs)
            print "f", e, cost / mbsz

        for e in xrange(epochsg):
            random.shuffle(ind)
            cost = 0.0
            self.dump_samples(e)
            for b in xrange(mbsz):
                bs = D[ind[mbsz * b: mbsz * (b+1)]]
                cs = self.traing_fn(bs)
                cost += cs
            print "g", e, cost / mbsz

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

if __name__ == '__main__':
    model = Disentangler(784, 100, [1200, 1200], [tanh, tanh, sigm], 0.01, 10)
    with gzip.open(os.environ['MNIST']) as f:
        D = (cPickle.load(f)[0][0] > 0.5).astype('float32')
    model.train(D, 30, 100, 100)





