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

class VGSN():
    def __init__(self, dimX, hls, acts, lr):
        t = MLP(dimX, dimX, hls, acts)
        s = MLP(dimX, dimX, hls, acts)
        
        x = T.fmatrix("x")
        tx = t(x)
        stx = s(tx)

        cs = ce(stx, x).mean(axis=1).mean(axis=0)
        sgrads = T.grad(cs, s.params, consider_constant=[tx])
        supdates = map(lambda (param, grad): (param, param - lr * grad), zip(s.params, sgrads))
        self.train_s = theano.function([x], cs, updates=supdates)

        xp = T.fmatrix("x'")
        sxp = s(xp)
        tsxp = t(sxp)

        ct = ce(tsxp, xp).mean(axis=1).mean(axis=0)
        tgrads = T.grad(ct, t.params, consider_constant=[sxp])
        tupdates = map(lambda (param, grad): (param, param - lr * grad), zip(t.params, tgrads))
        self.train_t = theano.function([xp], ct, updates=tupdates)

        self.t_fn = theano.function([x], t(x))

    def train(self, D, epochs, mbsz):
        ind_s = range(D.shape[0])
        ind_t = range(D.shape[0])
        for e in xrange(epochs):
            random.shuffle(ind_s)
            random.shuffle(ind_t)
            cost = 0.0
            self.dump_samples(D, e)
            for b in xrange(mbsz):
                bs = D[ind_s[mbsz * b: mbsz * (b+1)]]
                bt = D[ind_t[mbsz * b: mbsz * (b+1)]]
                cs = self.train_s(bs)
                ct = self.train_t(bt)
                cost += cs + ct
            print e, cost / (2 * mbsz)

    def dump_samples(self, D, name):
        C = 10
        T = 10
        b = D[:C]
        samples = numpy.zeros((C * T), D.shape[1])
        for t in xrange(T):
            samples[C*t:C*(t+1)] = b
            b = self.t_fn(b)
        draw_mnist(samples, 'samples/', T, C, name)

def draw_mnist(samples, output_dir, num_samples, num_chains, name):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    all = Image.new("RGB", (28*num_samples, 28*num_chains))
    for i in xrange(num_samples):
        for j in xrange(num_chains):
            pic = (samples[i, j].reshape(28, 28)) * 255
            im = Image.fromarray(pic)
            all.paste(im, (28 * i, 28 * j))
    all.save(os.path.join(output_dir, 'samples_%d.png' % name))



if __name__ == '__main__':
    with gzip.open(os.environ['MNIST']) as f:
        D = cPickle.load(f)[0][0]
    vgsn = VGSN(784, [1200, 1200], [tanh, tanh, sigm], 0.01)
    vgsn.train(D, 500, 100)














