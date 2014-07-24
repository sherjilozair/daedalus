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

from theano.tensor.shared_randomstreams import RandomStreams
rng_theano = RandomStreams()

sigm = T.nnet.sigmoid
tanh = T.tanh
ces = lambda x, y: T.nnet.binary_crossentropy(x, y).mean(axis=1).mean(axis=0)

def corrupt(z, noise):
    return z + rng_theano.normal(avg=0, std=noise, size=z.shape)

def ce(p, t):
    p = (p + 1.)/2.
    t = (t + 1.)/2.
    return ces(p, t)

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

def cor(x, lvl):
    return x + rng_theano.normal(std=lvl, size=x.shape)

class Model():
    def __init__(self):
        self.dimX = 784
        self.dimH = 100
        self.dimZ = 100
        self.noise = 0.1

        self.f1 = MLP(self.dimX, self.dimH, [], [sigm])
        self.g1 = MLP(self.dimH, self.dimX, [1200], [tanh, sigm])

        self.f2 = MLP(self.dimH, self.dimZ, [], [sigm])
        self.g2 = MLP(self.dimZ, self.dimH, [200], [tanh, sigm])

        p1 = self.f1.params + self.g1.params
        p2 = self.f2.params + self.g2.params

        x = T.fmatrix('x')
        lr = T.scalar('lr')

        h0 = x
        h1 = self.f1(h0)
        h2 = self.f2(h1)
        hh2 = h2
        hh1 = self.g2(hh2)
        hh0 = self.g1(hh1)

        h0_h1 = self.g1(cor(h1, 0.1))
        c1 = ce(h0_h1, h0) + ce(hh0, h0)
        grad_g1 = T.grad(c1, self.g1.params, consider_constant=[h1, hh1, h0])

        h1_h2 = self.g1(cor(h2, 0.1))
        c2 = ce(h1_h2, h1) + ce(hh1, h1)
        grad_g2 = T.grad(c2, self.g2.params, consider_constant=[h2, hh2, h1])

        hh1_hh0 = self.f1(cor(hh0, 0.1))
        c3 = ce(hh1_hh0, hh1) + ce(h1, hh1)
        grad_f1 = T.grad(c3, self.f1.params, consider_constant=[hh1, hh0, h0])

        hh2_hh1 = self.f2(cor(hh1, 0.1))
        c4 = ce(hh2_hh1, hh2) + ce(h2, hh2)
        grad_f2 = T.grad(c4, self.f1.params, consider_constant=[hh2, hh1, h1])

        grads = grad_g1 + grad_g2 + grad_f1 + grad_f2
        params = []
        for nn in [g1, g2, f1, f2]:
            params += nn.params

        updates = map(lambda (param, grad): (param, param - lr * grad), zip(params, grads))

        self.train_fn = theano.function([x, lr], [c1, c2, c3, c4], updates=updates)
        #i = T.fmatrix('input')
        #self.encode_fn = theano.function([i], self.f(i), allow_input_downcast=True)
        #self.decode_fn = theano.function([i], self.g(i), allow_input_downcast=True)
        #self.denoise_fn = theano.function([i], self.d(i), allow_input_downcast=True)

    def train(self, D, epochs, mbsz, lr_init, lr_scale):
        ind = range(D.shape[0])
        lr = lr_init
        num_batches = D.shape[0] / mbsz
        noisev = 0.999999
        t1 = time.clock()
        for e in xrange(epochs):
            if e % 5 == 0:
                cPickle.dump(self, open("models/model_%s.pkl" % e, 'w'))
            random.shuffle(ind)
            cost = 0.0
            #self.dump_samples(D, e)
            #self.dump_recons(D, e)

            for b in xrange(num_batches):
                bs = D[ind[mbsz * b: mbsz * (b+1)]]
                noise = 1. - noisev
                cs = self.train_fn(bs, lr)
                cs = numpy.array(cs)
                cost += cs
                #print "minibatch", e, b, cs, lr
            t2 = time.clock()
            dur = t2 - t1
            t1 = t2
            lr *= lr_scale
            noisev *= 0.99
            print e, cost / num_batches, noise, lr, dur

    def dump_recons(self, D, name):
        C = 20
        T = 20
        samples = numpy.zeros((C * T, D.shape[1]))
        for t in xrange(T):
            z = self.encode_fn(D[C*t:C*(t+1)])
            px = self.decode_fn(z)
            samples[C*t:C*(t+1)] = px
        draw_mnist(samples, 'recons/', T, C, name)

    def dump_samples(self, D, name):
        C = 20
        T = 20
        samples = numpy.zeros((C * T, D.shape[1]))
        z = self.encode_fn(D[:C])
        for t in xrange(T):
            cz = z + numpy.random.normal(0., self.noise, size=z.shape)
            z = self.denoise_fn(cz)
            px = self.decode_fn(z)
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
    model = Model()
    D = numpy.load("../mnist.npy")
    D = (D > 0.5).astype('float32')
    model.train(D, 100, 100, 1., 0.99)






