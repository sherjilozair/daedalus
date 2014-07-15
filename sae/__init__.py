import theano
import numpy
from theano import tensor as T
import random

from PIL import Image

import os
import sys
import gzip
import cPickle
import operator as op
import time

sigm = T.nnet.sigmoid
tanh = T.tanh
ce = T.nnet.binary_crossentropy

from theano.tensor.shared_randomstreams import RandomStreams
rng_theano = RandomStreams()


class Layer():
    def __init__(self, n_in, n_out, act=tanh):
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

class SAE():
    def __init__(self, dimX, hls, dimZ):
        self.dimX = dimX
        self.dimZ = dimZ
        self.hls = list(hls)
        self.encoder = [Layer(*args) for args in zip([dimX]+hls, hls+[dimZ])]
        rhls = list(reversed(hls))
        self.decoder = list(reversed([Layer(*args) for args in zip([dimZ]+rhls, rhls+[dimX])]))
        self.params = reduce(op.add, map(lambda l: l.params, self.encoder + self.decoder))
        x = T.fmatrix('x')
        lr = T.scalar('lr')
        h = [None] * (len(self.encoder) + 1)
        r = [None] * len(self.decoder)
        h[0] = x
        cost = 0.0
        costs = [None] * len(self.encoder)
        for l in xrange(len(self.encoder)):
            print l
            h[l+1] = self.encoder[l](h[l])
            r[l] = self.decoder[l](h[l+1])
            costs[l] = self.dis(r[l], h[l])
        self.cost = sum(costs)
        self.grads = T.grad(self.cost, self.params)
        self.updates = map(lambda (param, grad): (param, param - lr * grad), zip(self.params, self.grads))
        self.train_fn = theano.function([x, lr], costs, updates=self.updates, allow_input_downcast=True)
    
        h = x
        for l in xrange(len(self.encoder)):
            h = self.encoder[l](h)
        for l in reversed(xrange(len(self.decoder))):
            h = self.decoder[l](h)
        self.recons_fn = theano.function([x], h, allow_input_downcast=True)


    def train(self, D, epochs, mbsz, lr_init, lr_scale):
        ind = range(D.shape[0])
        lr = lr_init
        num_batches = D.shape[0] / mbsz
        t1 = time.clock()
        for e in xrange(epochs):
            #cPickle.dump(self, open("models/model_%s.pkl" % e, 'w'))
            random.shuffle(ind)
            cost = 0.0
            #self.dump_samples(D, e)
            self.dump_recons(D, e)
            for b in xrange(num_batches):
                bs = D[ind[mbsz * b: mbsz * (b+1)]]
                #z = numpy.random.normal(0, 1, (mbsz, self.dimZ)).astype('float32')
                cs = self.train_fn(bs, lr)
                cs = numpy.array(cs)
                cost += cs
                #print "minibatch", e, b, cs, lr
            t2 = time.clock()
            duration = t2 - t1
            t1 = t2
            lr *= lr_scale
            print e, cost / num_batches, lr, duration
 
    def dis(self, x1, x2):
        return ((x1 - x2)**2).mean(axis=1).mean(axis=0)

    def dump_recons(self, D, name):
        C = 10
        T = 10
        samples = numpy.zeros((C * T, D.shape[1]))
        for t in xrange(T):
            px = self.recons_fn(D[C*t:C*(t+1)])
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


if __name__ == '__main__':
    model = SAE(784, [1200, 1200], 100)
    D = numpy.load("../mnist.npy")
    D = (D > 0.5).astype('float32')
    model.train(D, 100, 100, 0.25, 0.99)


