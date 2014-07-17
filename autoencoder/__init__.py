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

class Model():
    def __init__(self, dimX, dimZ, hls, acts):
    	self.dimZ = dimZ
        self.f = MLP(dimX, dimZ, hls, acts)
        self.g = MLP(dimZ, dimX, hls, acts)
        self.generator = MLP(dimZ, dimX, [1200, 1200], [tanh, tanh, sigm])
        self.params = self.f.params + self.g.params + self.generator.params
        x = T.fmatrix('x')
        lr = T.scalar('lr')
        noise = T.scalar('noise')
        z = self.f(x)
        rx = self.g(z)
        cost_recons = ce(rx, x).mean(axis=1).mean(axis=0)
        
        cz = rng_theano.uniform(low=0, high=1, size=z.shape)
        nz = self.nearest_neighbour_of_in(cz, z) # nn of cz in z
        xnz = self.g(nz)
        rxx = self.generator(cz)
        cost_gen = ce(rxx, xnz).mean(axis=1).mean(axis=0)
        cost = (cost_recons + cost_gen) / 2.
        grads = T.grad(cost, self.params, consider_constant=[xnz, cz])
        updates = map(lambda (param, grad): (param, param - lr * grad), zip(self.params, grads))
        nnd = self.nearest_neighbour_distances(z)
        self.train_fn = theano.function([x, lr], [cost_recons, cost_gen, nnd.mean(), nnd.std()], updates=updates)

        z = T.fmatrix('z')
        self.sample_fn = theano.function([z], self.g(z), allow_input_downcast=True)
        self.infer_fn = theano.function([x], self.f(x), allow_input_downcast=True)    
        self.generator_fn = theano.function([z], self.generator(z), allow_input_downcast=True)

    def train(self, D, epochs, mbsz, lr_init, lr_scale):
        ind = range(D.shape[0])
        lr = lr_init
        num_batches = D.shape[0] / mbsz
        noisev = 0.999999
        for e in xrange(epochs):
            if e % 5 == 0:
                cPickle.dump(self, open("models/model_%s.pkl" % e, 'w'))
            random.shuffle(ind)
            cost = 0.0
            self.dump_samples(D, e)
            self.dump_recons(D, e)

            for b in xrange(num_batches):
                bs = D[ind[mbsz * b: mbsz * (b+1)]]
                noise = 1. - noisev
                cs = self.train_fn(bs, lr)
                cs = numpy.array(cs)
                cost += cs
                #print "f g minibatch", e, b, cs1, cs2, lr
            lr *= lr_scale
            noisev *= 0.99
            print e, cost / num_batches, noise, lr
        
    def dump_recons(self, D, name):
        C = 20
        T = 20
        samples = numpy.zeros((C * T, D.shape[1]))
        for t in xrange(T):
            z = self.infer_fn(D[C*t:C*(t+1)])
            px = self.sample_fn(z)
            samples[C*t:C*(t+1)] = px
        draw_mnist(samples, 'recons/', T, C, name)

    def dump_samples(self, D, name):
        C = 20
        T = 20
        samples = numpy.zeros((C * T, D.shape[1]))
        for t in xrange(T):
            z = numpy.random.uniform(0, 1, (C, self.dimZ))
            px = self.generator_fn(z)
            samples[C*t:C*(t+1)] = px
        draw_mnist(samples, 'samples/', T, C, name)

    def nearest_neighbour_distances(self, z):
        d = ((z - z[:, numpy.newaxis, :])**2).sum(axis=2)
        ind = [(d + (d.max()+1)* T.eye(d.shape[0])).argmin(axis=0)]
        znn = z[ind]
        return ((z - znn)**2).sum(axis=1)

    def nearest_neighbour_of_in(self, cz, z):
        d = ((z - cz[:, numpy.newaxis, :])**2).sum(axis=2)
        ind = d.argmin(axis=1)
        return z[ind]

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
    model = Model(784, 100, [1200], [tanh, sigm])
    D = numpy.load("../mnist.npy")
    D = (D > 0.5).astype('float32')
    model.train(D, 100, 100, 1., 0.99)






