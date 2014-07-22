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
ce = lambda x, y: T.nnet.binary_crossentropy(x, y).mean(axis=1).mean(axis=0)

def corrupt(z, noise):
    return z + rng_theano.normal(avg=0, std=noise, size=z.shape)

def ces(p, t):
    p = (p + 1.)/2.
    t = (t + 1.)/2.
    return ce(p, t)

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
        return theano.shared(numpy.zeros(n_out,) + 0.5)

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



# (1) h_{l-1} is a target for g_l(corrupt(h_l))     --> modify g_l
# {makes g_l invert f_l on h's and be contractive}
# (2) h_{l-1} is a target for g_l(\hat{h}_l)   --> modify g_l
# {makes g_l generate h's, to maximize log P(h_{l-1})}
# (3) \hat{h}_l is a target for f_l(corrupt(\hat{h}_{l-1}))  --> modify f_l
# {makes f_l invert g_l on \hat{h}'s and be contractive}
# (4) \hat{h}_l is a target for f_l(h_{l-1})  --> modify f_l
# {makes f_l produce h_l that are likely under P(h_l)


class Model():
    def __init__(self):
        self.dimX = 784
        self.hls = [500, 300, 100]
        x = T.fmatrix('x')
        lr = T.scalar('lr')











        f1 = MLP(784, 500, [1200], [tanh, sigm])
        g1 = MLP(500, 784, [1200], [tanh, sigm])
        params1 = f1.params + g1.params

        cost1 = ce(g1(f1(x)), x)
        grads1 = T.grad(cost1, params1)
        updates1 = map(lambda (param, grad): (param, param - lr * grad), zip(params1, grads1))

        h1 = f1(x)
        f2 = MLP(500, 500, [700], [tanh, sigm])
        g2 = MLP(500, 500, [700], [tanh, sigm])
        params2 = f2.params + g2.params

        cost2 = ce(g2(f2(h1)), h1)
        grads2 = T.grad(cost2, params2, consider_constant=[h1])
        updates2 = map(lambda (param, grad): (param, param - lr * grad), zip(params2, grads2))

        updates = updates1 + updates2




        self.train_fn = theano.function([x, lr], [cost1, cost2], updates=updates)

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






